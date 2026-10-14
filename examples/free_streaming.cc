/* ---------------------------------------------------------------------
 *
 * hyperforest example: steady free streaming in phase space
 *
 *   v . grad_x f + sigma f = q(x, v)   in X x V,
 *   f = g                               on {x in dX, v . n < 0},
 *
 * X a deal.II coarse mesh in d = 1, 2, 3 dimensions (the unit cube cut
 * into 2^d trees, or a ball), V = [-1, 1]^d, so the phase space has
 * D = 2 d = 2, 4 or 6 dimensions. Upwind DG with Q_p(x) x Q_p(v) on a
 * hyperforest of 2^D-trees, assembled element by element (strong form:
 * every element integrates its own inflow faces), GMRES with a block
 * Jacobi preconditioner, MPI parallel.
 *
 * The example walks through the library:
 *   PhaseSpace            coarse spatial mesh x velocity box -> trees
 *   Forest                adapt / balance / partition / ghost layer
 *   face_neighbors        upwind fluxes across conforming, hanging and
 *                         tree faces (reference-coordinate maps)
 *   make_partitioner      LinearAlgebra::distributed::Vector with ghosts
 *   TensorDG              the tensor-product basis, solution transfer
 *   relation, transfer_fixed   data transfer under adapt and partition
 *
 * test=linear: f = (1 + a.x)(1 + b.v) lies in the discrete space, so the
 *              solution is exact up to round-off on any mesh, conforming
 *              or not (refinement=random refines and coarsens at random).
 * test=smooth: a Gaussian bump; L2 error and convergence order.
 *
 * usage: mpirun -np 4 ./free_streaming dim=2 test=smooth refinement=uniform cycles=4
 *
 * ---------------------------------------------------------------------
 */

#include <hyperforest/dealii.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace FreeStreaming
{
  using namespace dealii;
  namespace hf     = hyperforest;
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  struct Parameters
  {
    std::string  test       = "linear";  // linear | smooth
    std::string  refinement = "random";  // random | uniform | adaptive
    std::string  mesh       = "cube";    // cube (2^d trees) | ball (d >= 2)
    unsigned int degree     = 1;
    unsigned int level      = 1; // initial uniform level of the trees
    unsigned int cycles     = 4;
    double       sigma      = 1.0;
    double       tol        = 1e-12;
  };


  template <int d>
  class Problem
  {
  public:
    static constexpr int D = 2 * d;
    using CellD            = hf::Cell<D>;

    explicit Problem(const Parameters &prm);
    void run();

    void vmult(VectorType &dst, const VectorType &src) const;        // A src
    void precondition(VectorType &dst, const VectorType &src) const; // block Jacobi

  private:
    struct Geo
    {
      int       tree;
      Point<d>  lo;
      double    h;
      Point<d>  vlo; // velocity box of the element
      Point<d>  vh;  // and its edges
    };

    double exact(const Point<d> &x, const Tensor<1, d> &v) const;
    double source(const Point<d> &x, const Tensor<1, d> &v) const;
    Geo    geo(const CellD &c) const;
    void   map(const Geo &g, const double *xi, Point<d> &x, Tensor<2, d> &J) const;
    Tensor<1, d> velocity(const Geo &g, const double *xiv) const;

    void         setup();
    void         assemble();
    unsigned int solve();
    void         estimate(std::vector<double> &eta) const;
    void         refine(const unsigned int cycle);
    void         errors(double &l2, double &l2_exact, double &max_nodal) const;

    MPI_Comm           comm;
    ConditionalOStream pcout;
    const Parameters   prm;

    Triangulation<d>                   coarse;
    std::unique_ptr<hf::PhaseSpace<d, d>> ps;
    std::unique_ptr<hf::Forest<D>>     forest;
    const hf::TensorDG<d, d>           fe;
    const unsigned int                 n;

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    VectorType                                         solution, rhs;

    // element rows: inverse and matrix of the diagonal block, off-diagonal
    // blocks to face neighbours (column: local element, or -1 - ghost)
    std::vector<double>  diag, diag_inv;
    std::vector<int64_t> row_start;
    std::vector<int32_t> off_col;
    std::vector<double>  off_val;
  };


  // ---------------------------------------------------------- the data
  template <int d>
  double Problem<d>::exact(const Point<d> &x, const Tensor<1, d> &v) const
  {
    static const double a[3] = {0.3, 0.2, 0.1}, b[3] = {0.4, -0.3, 0.2};
    if (prm.test == "linear")
    {
      double ax = 1, bv = 1;
      for (int k = 0; k < d; ++k)
      {
        ax += a[k] * x[k];
        bv += b[k] * v[k];
      }
      return ax * bv;
    }
    double r2 = 0;
    for (int k = 0; k < d; ++k)
      r2 += (x[k] - 0.5) * (x[k] - 0.5);
    return std::exp(-r2 / 0.08) * std::exp(-v.norm_square()) * (1 + 0.3 * v[0]);
  }

  // q = v . grad_x f + sigma f
  template <int d>
  double Problem<d>::source(const Point<d> &x, const Tensor<1, d> &v) const
  {
    static const double a[3] = {0.3, 0.2, 0.1}, b[3] = {0.4, -0.3, 0.2};
    const double        f = exact(x, v);
    double              vgrad = 0;
    if (prm.test == "linear")
    {
      double bv = 1;
      for (int k = 0; k < d; ++k)
      {
        bv += b[k] * v[k];
        vgrad += v[k] * a[k];
      }
      vgrad *= bv;
    }
    else
      for (int k = 0; k < d; ++k)
        vgrad += v[k] * f * (-2.0 * (x[k] - 0.5) / 0.08);
    return vgrad + prm.sigma * f;
  }


  // ---------------------------------------------------------- geometry
  template <int d>
  typename Problem<d>::Geo Problem<d>::geo(const CellD &c) const
  {
    Geo g;
    ps->spatial_box(c, g.tree, g.lo, g.h);
    std::array<double, d> lo, hi;
    ps->angular_box(c, lo, hi);
    for (int k = 0; k < d; ++k)
    {
      g.vlo[k] = lo[k];
      g.vh[k]  = hi[k] - lo[k];
    }
    return g;
  }

  // x(xi) and J = dx/dxi of the spatial factor (xi in [0,1]^d)
  template <int d>
  void Problem<d>::map(const Geo &g, const double *xi, Point<d> &x, Tensor<2, d> &J) const
  {
    Point<d> zeta;
    for (int k = 0; k < d; ++k)
      zeta[k] = g.lo[k] + g.h * xi[k];
    ps->space.map(g.tree, zeta, x, J);
    J *= g.h;
  }

  template <int d>
  Tensor<1, d> Problem<d>::velocity(const Geo &g, const double *xiv) const
  {
    Tensor<1, d> v;
    for (int k = 0; k < d; ++k)
      v[k] = g.vlo[k] + g.vh[k] * xiv[k];
    return v;
  }


  // ---------------------------------------------------------- setup
  template <int d>
  Problem<d>::Problem(const Parameters &prm)
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , prm(prm)
    , fe(prm.degree, prm.degree)
    , n(fe.n)
  {
    if (prm.mesh == "ball" && d >= 2)
      GridGenerator::hyper_ball(coarse); // trees with rotated neighbours
    else
      GridGenerator::subdivided_hyper_cube(coarse, 2, 0.0, 1.0);
    std::array<int, d>    nv;
    std::array<double, d> lo, hi;
    std::array<bool, d>   periodic;
    for (int k = 0; k < d; ++k)
    {
      nv[k]       = 1;
      lo[k]       = -1.0;
      hi[k]       = 1.0;
      periodic[k] = false;
    }
    ps     = std::make_unique<hf::PhaseSpace<d, d>>(coarse, nv, lo, hi, periodic);
    forest = std::make_unique<hf::Forest<D>>(comm, ps->connectivity, prm.level);
  }

  template <int d>
  void Problem<d>::setup()
  {
    forest->build_ghost_layer();
    partitioner = hf::make_partitioner(*forest, n);
    VectorType old;
    old.swap(solution);
    solution.reinit(partitioner);
    if (old.size() == solution.size())
      solution.copy_locally_owned_data_from(old);
    rhs.reinit(partitioner);
  }


  // ---------------------------------------------------------- assembly
  // Element i: volume (v . grad f + sigma f) w and q w; on every spatial
  // face the inflow part -(v.n) (f - f_upwind) w, with f_upwind from the
  // face neighbours (one per piece of the face; hanging faces have several
  // pieces) or the boundary data g. Velocity faces carry no flux.
  template <int d>
  void Problem<d>::assemble()
  {
    const int32_t N = forest->n_local();
    diag.assign(size_t(N) * n * n, 0.0);
    diag_inv.assign(size_t(N) * n * n, 0.0);
    row_start.assign(N + 1, 0);
    off_col.clear();
    off_val.clear();
    rhs = 0;

    const QGauss<1>    g1(prm.degree + 1);
    const unsigned int m = g1.size();
    unsigned int       mx = 1;
    for (int k = 0; k < d; ++k)
      mx *= m;
    const unsigned int nX = fe.nX, nS = fe.nS;
    // reference values at the tensor Gauss points of one factor
    std::vector<std::array<double, d>> qp(mx);
    std::vector<double>                qw(mx), vx(mx * nX), vs(mx * nS);
    std::vector<Tensor<1, d>>          gx(mx * nX);
    for (unsigned int q = 0; q < mx; ++q)
    {
      unsigned int r = q;
      qw[q]          = 1;
      for (int k = 0; k < d; ++k)
      {
        qp[q][k] = g1.point(r % m)[0];
        qw[q] *= g1.weight(r % m);
        r /= m;
      }
      fe.X.values(qp[q].data(), &vx[q * nX], &gx[q * nX]);
      fe.S.values(qp[q].data(), &vs[q * nS]);
    }

    std::vector<double>              A(n * n), G(nX * nX), vc(n), vg(n), vxf(nX), vsf(nS);
    std::vector<hf::FaceNeighbor<D>> nbs;
    for (int32_t i = 0; i < N; ++i)
    {
      const CellD &c  = forest->local_cell(i);
      const Geo    g  = geo(c);
      double       vj = 1; // velocity Jacobian
      for (int k = 0; k < d; ++k)
        vj *= g.vh[k];
      std::fill(A.begin(), A.end(), 0.0);
      double *b = rhs.begin() + size_t(i) * n;

      // volume
      std::vector<Point<d>>     xq(mx);
      std::vector<double>       JxW(mx);
      std::vector<Tensor<1, d>> grad(mx * nX);
      for (unsigned int q = 0; q < mx; ++q)
      {
        Tensor<2, d> J;
        map(g, qp[q].data(), xq[q], J);
        JxW[q]                   = determinant(J) * qw[q];
        const Tensor<2, d> JinvT = transpose(invert(J));
        for (unsigned int j = 0; j < nX; ++j)
          grad[q * nX + j] = JinvT * gx[q * nX + j];
      }
      for (unsigned int a = 0; a < mx; ++a) // velocity points
      {
        const Tensor<1, d> v  = velocity(g, qp[a].data());
        const double       wa = qw[a] * vj;
        const double *     psi = &vs[a * nS];
        std::fill(G.begin(), G.end(), 0.0);
        for (unsigned int q = 0; q < mx; ++q)
          for (unsigned int ix = 0; ix < nX; ++ix)
          {
            const double wi = vx[q * nX + ix] * JxW[q];
            for (unsigned int jx = 0; jx < nX; ++jx)
              G[ix * nX + jx] += wi * (v * grad[q * nX + jx] + prm.sigma * vx[q * nX + jx]);
          }
        for (unsigned int ix = 0; ix < nX; ++ix)
          for (unsigned int is = 0; is < nS; ++is)
            for (unsigned int jx = 0; jx < nX; ++jx)
            {
              const double gij = wa * psi[is] * G[ix * nX + jx];
              for (unsigned int js = 0; js < nS; ++js)
                A[(ix * nS + is) * n + jx * nS + js] += gij * psi[js];
            }
        for (unsigned int q = 0; q < mx; ++q)
        {
          const double qq = source(xq[q], v) * JxW[q] * wa;
          for (unsigned int ix = 0; ix < nX; ++ix)
            for (unsigned int is = 0; is < nS; ++is)
              b[ix * nS + is] += qq * vx[q * nX + ix] * psi[is];
        }
      }

      // inflow faces
      for (int f = 0; f < 2 * d; ++f)
      {
        forest->face_neighbors(i, f, nbs);
        const int    nrm  = f / 2;
        const double sign = (f & 1) ? 1.0 : -1.0;
        const size_t n_pieces = nbs.empty() ? 1 : nbs.size();
        for (size_t p = 0; p < n_pieces; ++p)
        {
          std::array<double, D> lo, hi;
          for (int k = 0; k < D; ++k)
          {
            lo[k] = nbs.empty() ? 0.0 : nbs[p].lo[k];
            hi[k] = nbs.empty() ? 1.0 : nbs[p].hi[k];
          }
          lo[nrm] = hi[nrm] = f & 1;
          std::vector<double> B(nbs.empty() ? 0 : n * n, 0.0);
          bool                any = false;
          unsigned int        nq  = 1;
          for (int k = 0; k < D - 1; ++k)
            nq *= m;
          for (unsigned int q = 0; q < nq; ++q)
          {
            std::array<double, D> xi;
            double                w = 1;
            unsigned int          r = q;
            for (int k = 0; k < D; ++k)
              if (k == nrm)
                xi[k] = f & 1;
              else
              {
                xi[k] = lo[k] + (hi[k] - lo[k]) * g1.point(r % m)[0];
                w *= g1.weight(r % m) * (hi[k] - lo[k]);
                r /= m;
              }
            Point<d>     x;
            Tensor<2, d> J;
            map(g, xi.data(), x, J);
            const Tensor<2, d> cof = cofactor(J); // area-weighted normal: column nrm
            Tensor<1, d>       Nv;
            for (int k = 0; k < d; ++k)
              Nv[k] = sign * cof[k][nrm];
            const Tensor<1, d> v  = velocity(g, xi.data() + d);
            const double       vn = v * Nv;
            if (vn >= 0)
              continue; // outflow: no term in the strong form
            const double ww = -vn * w * vj;
            fe.X.values(xi.data(), vxf.data());
            fe.S.values(xi.data() + d, vsf.data());
            for (unsigned int ix = 0; ix < nX; ++ix)
              for (unsigned int is = 0; is < nS; ++is)
                vc[ix * nS + is] = vxf[ix] * vsf[is];
            for (unsigned int a = 0; a < n; ++a)
              for (unsigned int bb = 0; bb < n; ++bb)
                A[a * n + bb] += ww * vc[a] * vc[bb];
            if (nbs.empty())
            {
              const double gval = exact(x, v);
              for (unsigned int a = 0; a < n; ++a)
                b[a] += ww * gval * vc[a];
              continue;
            }
            const auto xn = nbs[p].map(xi); // the same point in the neighbour
            fe.X.values(xn.data(), vxf.data());
            fe.S.values(xn.data() + d, vsf.data());
            for (unsigned int ix = 0; ix < nX; ++ix)
              for (unsigned int is = 0; is < nS; ++is)
                vg[ix * nS + is] = vxf[ix] * vsf[is];
            for (unsigned int a = 0; a < n; ++a)
              for (unsigned int bb = 0; bb < n; ++bb)
                B[a * n + bb] -= ww * vc[a] * vg[bb];
            any = true;
          }
          if (any)
          {
            off_col.push_back(nbs[p].ghost ? -nbs[p].index - 1 : nbs[p].index);
            off_val.insert(off_val.end(), B.begin(), B.end());
          }
        }
      }
      row_start[i + 1] = off_col.size();
      std::copy(A.begin(), A.end(), diag.begin() + size_t(i) * n * n);
      FullMatrix<double> inv(n, n);
      for (unsigned int a = 0; a < n; ++a)
        for (unsigned int bb = 0; bb < n; ++bb)
          inv(a, bb) = A[a * n + bb];
      inv.gauss_jordan();
      for (unsigned int a = 0; a < n; ++a)
        for (unsigned int bb = 0; bb < n; ++bb)
          diag_inv[size_t(i) * n * n + a * n + bb] = inv(a, bb);
    }
  }


  // ---------------------------------------------------------- solver
  template <int d>
  void Problem<d>::vmult(VectorType &dst, const VectorType &src) const
  {
    src.update_ghost_values(); // ghost dofs after the owned ones, in ghost order
    const int32_t N = forest->n_local();
    const double *s = src.begin();
    for (int32_t i = 0; i < N; ++i)
    {
      double *      y = dst.begin() + size_t(i) * n;
      const double *A = &diag[size_t(i) * n * n];
      for (unsigned int a = 0; a < n; ++a)
      {
        double sum = 0;
        for (unsigned int bb = 0; bb < n; ++bb)
          sum += A[a * n + bb] * s[size_t(i) * n + bb];
        y[a] = sum;
      }
      for (int64_t k = row_start[i]; k < row_start[i + 1]; ++k)
      {
        const int32_t col = off_col[k];
        const double *x   = col >= 0 ? s + size_t(col) * n : s + (size_t(N) + size_t(-col - 1)) * n;
        const double *B   = &off_val[size_t(k) * n * n];
        for (unsigned int a = 0; a < n; ++a)
        {
          double sum = 0;
          for (unsigned int bb = 0; bb < n; ++bb)
            sum += B[a * n + bb] * x[bb];
          y[a] += sum;
        }
      }
    }
    src.zero_out_ghost_values();
  }

  template <int d>
  void Problem<d>::precondition(VectorType &dst, const VectorType &src) const
  {
    for (int32_t i = 0; i < forest->n_local(); ++i)
    {
      const double *M = &diag_inv[size_t(i) * n * n];
      for (unsigned int a = 0; a < n; ++a)
      {
        double sum = 0;
        for (unsigned int bb = 0; bb < n; ++bb)
          sum += M[a * n + bb] * src.local_element(size_t(i) * n + bb);
        dst.local_element(size_t(i) * n + a) = sum;
      }
    }
  }

  template <int d>
  struct Operator
  {
    const Problem<d> &p;
    void              vmult(VectorType &dst, const VectorType &src) const
    {
      p.vmult(dst, src);
    }
  };
  template <int d>
  struct Preconditioner
  {
    const Problem<d> &p;
    void              vmult(VectorType &dst, const VectorType &src) const
    {
      p.precondition(dst, src);
    }
  };

  template <int d>
  unsigned int Problem<d>::solve()
  {
    if (prm.test == "linear")
      solution = 0; // the transferred solution would already be exact
    SolverControl                           control(2000, prm.tol * rhs.l2_norm());
    SolverGMRES<VectorType>::AdditionalData data(100, true); // right preconditioning
    SolverGMRES<VectorType>                 gmres(control, data);
    gmres.solve(Operator<d>{*this}, solution, rhs, Preconditioner<d>{*this});
    return control.last_step();
  }


  // ---------------------------------------------------------- adaptivity
  // eta_i^2 = h sum over spatial faces of \int [f]^2 (neighbour values from
  // the ghost layer)
  template <int d>
  void Problem<d>::estimate(std::vector<double> &eta) const
  {
    solution.update_ghost_values();
    const int32_t                    N = forest->n_local();
    const double *                   u = solution.begin();
    const QGauss<1>                  g1(prm.degree + 1);
    const unsigned int               m = g1.size();
    std::vector<hf::FaceNeighbor<D>> nbs;
    eta.assign(N, 0.0);
    for (int32_t i = 0; i < N; ++i)
    {
      const Geo g = geo(forest->local_cell(i));
      for (int f = 0; f < 2 * d; ++f)
      {
        forest->face_neighbors(i, f, nbs);
        for (const auto &nb : nbs)
        {
          const double *ug = nb.ghost ? u + (size_t(N) + nb.index) * n : u + size_t(nb.index) * n;
          unsigned int  nq = 1;
          for (int k = 0; k < D - 1; ++k)
            nq *= m;
          for (unsigned int q = 0; q < nq; ++q)
          {
            std::array<double, D> xi;
            double                w = 1;
            unsigned int          r = q;
            for (int k = 0; k < D; ++k)
              if (k == f / 2)
                xi[k] = f & 1;
              else
              {
                xi[k] = nb.lo[k] + (nb.hi[k] - nb.lo[k]) * g1.point(r % m)[0];
                w *= g1.weight(r % m) * (nb.hi[k] - nb.lo[k]);
                r /= m;
              }
            const auto   xn   = nb.map(xi);
            const double jump = fe.value(u + size_t(i) * n, xi.data()) - fe.value(ug, xn.data());
            eta[i] += g.h * w * jump * jump;
          }
        }
      }
    }
    solution.zero_out_ghost_values();
  }

  template <int d>
  void Problem<d>::refine(const unsigned int cycle)
  {
    const int32_t       N = forest->n_local();
    std::vector<int8_t> flags(N, 0);
    if (prm.refinement == "uniform")
      std::fill(flags.begin(), flags.end(), 1);
    else if (prm.refinement == "random")
    {
      // refine ~15% of the leaves, coarsen ~30% of the families
      const auto hash = [&](const CellD &c) {
        uint64_t h = 1469598103934665603ull ^ (cycle * 7919);
        for (int k = 0; k < D; ++k)
          h = (h ^ uint32_t(c.x[k])) * 1099511628211ull;
        h = ((h ^ c.tree) * 1099511628211ull ^ c.level) * 1099511628211ull;
        return (h >> 17) % 100;
      };
      for (int32_t i = 0; i < N; ++i)
      {
        const CellD &c = forest->local_cell(i);
        if (c.level > int(prm.level) && hash(c.parent()) < 30)
          flags[i] = -1;
        else if (hash(c) < 15)
          flags[i] = 1;
      }
    }
    else // adaptive: maximum strategy on the jump indicator
    {
      std::vector<double> eta;
      estimate(eta);
      double mx = 0;
      for (const double e : eta)
        mx = std::max(mx, e);
      mx = Utilities::MPI::max(mx, comm);
      for (int32_t i = 0; i < N; ++i)
        flags[i] = eta[i] > 0.1 * mx;
    }

    // adapt + 2:1 balance keep the partition; transfer the solution, then
    // repartition and move the data along
    const std::vector<CellD> old_cells = forest->local_cells();
    std::vector<double>      old_data(solution.begin(), solution.begin() + size_t(N) * n);
    forest->adapt(flags);
    forest->balance();
    const auto          rel = hf::Forest<D>::relation(old_cells, forest->local_cells());
    std::vector<double> new_data(size_t(forest->n_local()) * n);
    for (int32_t j = 0; j < forest->n_local(); ++j)
      fe.transfer(forest->local_cell(j), rel[j], old_cells, old_data.data(), &new_data[size_t(j) * n]);
    const auto          old_gfc = forest->partition();
    std::vector<double> moved(size_t(forest->n_local()) * n);
    hf::transfer_fixed(comm, old_gfc, forest->global_first_cell(), new_data.data(), moved.data(), n);
    solution.reinit(0);
    setup();
    std::copy(moved.begin(), moved.end(), solution.begin());
  }


  // ---------------------------------------------------------- errors
  template <int d>
  void Problem<d>::errors(double &l2, double &l2_exact, double &max_nodal) const
  {
    l2 = l2_exact = max_nodal = 0;
    const QGauss<1>    g1(prm.degree + 2);
    const unsigned int m = g1.size();
    unsigned int       nq = 1;
    for (int k = 0; k < D; ++k)
      nq *= m;
    for (int32_t i = 0; i < forest->n_local(); ++i)
    {
      const Geo     g = geo(forest->local_cell(i));
      const double *u = solution.begin() + size_t(i) * n;
      double        vj = 1;
      for (int k = 0; k < d; ++k)
        vj *= g.vh[k];
      for (unsigned int q = 0; q < nq; ++q)
      {
        double       xi[D], w = 1;
        unsigned int r = q;
        for (int k = 0; k < D; ++k)
        {
          xi[k] = g1.point(r % m)[0];
          w *= g1.weight(r % m);
          r /= m;
        }
        Point<d>     x;
        Tensor<2, d> J;
        map(g, xi, x, J);
        const double ex = exact(x, velocity(g, xi + d)), e = fe.value(u, xi) - ex;
        const double ww = w * determinant(J) * vj;
        l2 += ww * e * e;
        l2_exact += ww * ex * ex;
      }
      for (unsigned int ix = 0; ix < fe.nX; ++ix) // nodal values
        for (unsigned int is = 0; is < fe.nS; ++is)
        {
          double xi[D];
          fe.X.node(ix, xi);
          fe.S.node(is, xi + d);
          Point<d>     x;
          Tensor<2, d> J;
          map(g, xi, x, J);
          max_nodal = std::max(max_nodal, std::abs(u[ix * fe.nS + is] - exact(x, velocity(g, xi + d))));
        }
    }
    l2        = std::sqrt(Utilities::MPI::sum(l2, comm));
    l2_exact  = std::sqrt(Utilities::MPI::sum(l2_exact, comm));
    max_nodal = Utilities::MPI::max(max_nodal, comm);
  }


  template <int d>
  void Problem<d>::run()
  {
    pcout << "free streaming in " << d << "D x " << d << "D phase space (hyperforest<" << D << ">), "
          << ps->connectivity->n_trees() << " trees, Q" << prm.degree << " x Q" << prm.degree << " ("
          << n << " dofs per element), " << Utilities::MPI::n_mpi_processes(comm)
          << " MPI ranks, test " << prm.test << ", refinement " << prm.refinement << std::endl;
    pcout << "cycle   elements        dofs  max_level  GMRES    L2_rel_error  max_nodal_error     EOC"
          << std::endl;
    setup();
    double prev = 0;
    for (unsigned int cycle = 0; cycle < prm.cycles; ++cycle)
    {
      if (cycle > 0)
        refine(cycle);
      assemble();
      const unsigned int its = solve();
      double             l2, l2e, mx;
      errors(l2, l2e, mx);
      const double rel = l2 / l2e;
      pcout << std::setw(5) << cycle << std::setw(11) << forest->n_global() << std::setw(12)
            << forest->n_global() * n << std::setw(11) << forest->global_max_level() << std::setw(7)
            << its << std::scientific << std::setprecision(3) << std::setw(16) << rel << std::setw(17)
            << mx;
      if (prm.refinement == "uniform" && cycle > 0 && prev > 0)
        pcout << std::fixed << std::setprecision(2) << std::setw(8) << std::log2(prev / rel);
      pcout << std::defaultfloat << std::endl;
      prev = rel;
    }
  }
} // namespace FreeStreaming


int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  try
  {
    std::map<std::string, std::string> arg = {{"dim", "2"},        {"test", "linear"}, {"refinement", "random"},
                                              {"mesh", "cube"},    {"degree", "1"},    {"level", "1"},
                                              {"cycles", "4"},     {"sigma", "1"},     {"tol", "1e-12"}};
    for (int i = 1; i < argc; ++i)
    {
      const std::string a   = argv[i];
      const auto        pos = a.find('=');
      if (pos == std::string::npos || !arg.count(a.substr(0, pos)))
        throw std::runtime_error("unknown argument " + a +
                                 "; usage: free_streaming [dim=1|2|3] [test=linear|smooth]"
                                 " [refinement=random|uniform|adaptive] [mesh=cube|ball] [degree=1]"
                                 " [level=1] [cycles=4] [sigma=1] [tol=1e-12]");
      arg[a.substr(0, pos)] = a.substr(pos + 1);
    }
    FreeStreaming::Parameters prm;
    prm.test       = arg["test"];
    prm.refinement = arg["refinement"];
    prm.mesh       = arg["mesh"];
    prm.degree     = std::stoi(arg["degree"]);
    prm.level      = std::stoi(arg["level"]);
    prm.cycles     = std::stoi(arg["cycles"]);
    prm.sigma      = std::stod(arg["sigma"]);
    prm.tol        = std::stod(arg["tol"]);
    if (arg["dim"] == "1")
      FreeStreaming::Problem<1>(prm).run();
    else if (arg["dim"] == "3")
      FreeStreaming::Problem<3>(prm).run();
    else
      FreeStreaming::Problem<2>(prm).run();
  }
  catch (std::exception &exc)
  {
    std::cerr << "Exception: " << exc.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return 0;
}
