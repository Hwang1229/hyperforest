/* ---------------------------------------------------------------------
 *
 * hyperforest <-> deal.II
 *
 * deal.II's Triangulation / DoFHandler / FEValues exist for dim <= 3 only.
 * For a phase space X x S (dim X = dx <= 3, dim S = ds <= 3, D = dx + ds)
 * the forest of hyperforest carries the D-dimensional mesh, and deal.II
 * supplies everything that lives on the factors:
 *
 *   connectivity_from_coarse_mesh(tria)  trees = coarse cells of a deal.II
 *                                        mesh (any quad / hex mesh; face
 *                                        orientations from the vertices)
 *   CoarseGeometry<dim>                  the d-linear map of every tree
 *   PhaseSpace<dx, ds>                   coarse spatial mesh x angular brick,
 *                                        the product connectivity and the
 *                                        geometry of the factors of a cell
 *   make_partitioner(forest, n)          a Utilities::MPI::Partitioner for
 *                                        LinearAlgebra::distributed::Vector
 *                                        with n dofs per leaf (DG); ghost
 *                                        dofs = dofs of the ghost leaves
 *   make_partitioner(forest, n_i, lay)   the same with n_i dofs on leaf i
 *                                        (hp), offsets in a DofLayout
 *   TensorDG<dx, ds>                     the basis Q_px(X) x Q_ps(S) (the
 *                                        tensor product of FE_DGQ<dx> and
 *                                        FE_DGQ<ds>, same numbering; px,
 *                                        ps >= 0), with solution transfer
 *                                        for adapt()
 *   transfer_hp(...)                     transfer between leaves of
 *                                        different degrees (hp)
 *
 * ---------------------------------------------------------------------
 */
#ifndef hyperforest_dealii_h
#define hyperforest_dealii_h

#include <hyperforest/forest.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <memory>
#include <vector>

namespace hyperforest
{
  // ------------------------------------------------------------------
  // connectivity of the coarse cells of a deal.II mesh
  // ------------------------------------------------------------------
  // Tree t = coarse cell with index t. deal.II numbers the vertices of a
  // quad / hex lexicographically and face f = 2 d + s is normal to axis d,
  // as in hyperforest; the transform across a face is read off from the
  // global vertex numbers. Boundary faces get their deal.II boundary id.
  template <int dim>
  Connectivity<dim> connectivity_from_coarse_mesh(const dealii::Triangulation<dim> &tria)
  {
    Connectivity<dim> c;
    c.resize(tria.n_cells(0));
    constexpr unsigned int nv = 1u << dim;
    for (auto cell = tria.begin(0); cell != tria.end(0); ++cell)
    {
      const int t = cell->index();
      for (unsigned int f = 0; f < 2 * dim; ++f)
      {
        if (cell->at_boundary(f))
        {
          c.boundary_ids[t][f] = cell->face(f)->boundary_id();
          continue;
        }
        const auto     nb = cell->neighbor(f);
        TreeFace<dim> &tf = c.faces[t][f];
        tf.tree           = nb->index();
        tf.face           = cell->neighbor_face_no(f);
        const auto find   = [&](const unsigned int g) {
          for (unsigned int j = 0; j < nv; ++j)
            if (nb->vertex_index(j) == g)
              return int(j);
          throw std::runtime_error("connectivity_from_coarse_mesh: faces do not match");
        };
        const int          n  = f / 2;
        const unsigned int i0 = (f & 1u) << n;
        const int          j0 = find(cell->vertex_index(i0));
        for (int d = 0; d < dim; ++d)
        {
          tf.perm[d] = d;
          tf.flip[d] = 0;
        }
        tf.perm[n] = tf.face / 2;
        for (int d = 0; d < dim; ++d)
          if (d != n)
          {
            const int diff = j0 ^ find(cell->vertex_index(i0 | (1u << d)));
            int       e    = 0;
            while ((1 << e) < diff)
              ++e;
            if (diff != (1 << e) || e == tf.face / 2)
              throw std::runtime_error("connectivity_from_coarse_mesh: bad face orientation");
            tf.perm[d] = e;
            tf.flip[d] = (j0 >> e) & 1;
          }
      }
    }
    c.check();
    return c;
  }


  // ------------------------------------------------------------------
  // d-linear geometry of the trees
  // ------------------------------------------------------------------
  template <int dim>
  class CoarseGeometry
  {
  public:
    static constexpr unsigned int nv = 1u << dim;

    CoarseGeometry() = default;
    explicit CoarseGeometry(const dealii::Triangulation<dim> &tria)
    {
      vertices.resize(tria.n_cells(0));
      for (auto cell = tria.begin(0); cell != tria.end(0); ++cell)
        for (unsigned int i = 0; i < nv; ++i)
          vertices[cell->index()][i] = cell->vertex(i);
    }

    // x(zeta) and J = dx/dzeta for zeta in [0,1]^dim of tree t
    void map(const int t, const dealii::Point<dim> &zeta, dealii::Point<dim> &x,
             dealii::Tensor<2, dim> &J) const
    {
      x = dealii::Point<dim>();
      J = dealii::Tensor<2, dim>();
      for (unsigned int i = 0; i < nv; ++i)
      {
        double                 N = 1;
        dealii::Tensor<1, dim> dN;
        for (int d = 0; d < dim; ++d)
          dN[d] = 1;
        for (int d = 0; d < dim; ++d)
        {
          const bool   b  = (i >> d) & 1;
          const double v  = b ? zeta[d] : 1 - zeta[d];
          const double dv = b ? 1 : -1;
          N *= v;
          for (int e = 0; e < dim; ++e)
            dN[e] *= e == d ? dv : v;
        }
        const dealii::Point<dim> &v = vertices[t][i];
        for (int a = 0; a < dim; ++a)
        {
          x[a] += N * v[a];
          for (int e = 0; e < dim; ++e)
            J[a][e] += dN[e] * v[a];
        }
      }
    }
    dealii::Point<dim> map(const int t, const dealii::Point<dim> &zeta) const
    {
      dealii::Point<dim>     x;
      dealii::Tensor<2, dim> J;
      map(t, zeta, x, J);
      return x;
    }

    std::vector<std::array<dealii::Point<dim>, nv>> vertices;
  };


  // ------------------------------------------------------------------
  // phase space = coarse spatial mesh x brick of angular boxes
  // ------------------------------------------------------------------
  template <int dx, int ds>
  class PhaseSpace
  {
  public:
    static constexpr int D = dx + ds;

    // The angular domain [lo, hi] is cut into n[0] x ... boxes (periodic
    // along the axes with periodic[d]). With angle_major the trees are
    // numbered angle by angle, so that ranks own direction sectors.
    PhaseSpace(const dealii::Triangulation<dx> &coarse_space,
               const std::array<int, ds> &      n_ang,
               const std::array<double, ds> &   lo,
               const std::array<double, ds> &   hi,
               const std::array<bool, ds> &     periodic,
               const bool                       angle_major = true)
      : space(coarse_space)
      , n_ang(n_ang)
      , ang_lo(lo)
      , ang_hi(hi)
    {
      const Connectivity<dx> cx = connectivity_from_coarse_mesh(coarse_space);
      const Connectivity<ds> cs = Connectivity<ds>::brick(n_ang, periodic);
      connectivity = std::make_shared<Connectivity<D>>(product(cx, cs, angle_major, &index));
      connectivity->check();
    }

    int tree_x(const int t) const
    {
      return index.ta(t);
    }
    int tree_s(const int t) const
    {
      return index.tb(t);
    }

    // the spatial factor of c: tree, lower corner and edge in the reference
    // coordinates [0,1]^dx of the coarse cell
    void spatial_box(const Cell<D> &c, int &tree, dealii::Point<dx> &lo, double &h) const
    {
      tree = tree_x(c.tree);
      h    = double(c.length()) / root_len;
      for (int d = 0; d < dx; ++d)
        lo[d] = double(c.x[d]) / root_len;
    }
    // the angular factor of c as a box [lo, hi] in angular coordinates
    void angular_box(const Cell<D> &c, std::array<double, ds> &lo, std::array<double, ds> &hi) const
    {
      int       t = tree_s(c.tree);
      const double h = double(c.length()) / root_len;
      for (int d = 0; d < ds; ++d)
      {
        const int    i = t % n_ang[d];
        t /= n_ang[d];
        const double w = (ang_hi[d] - ang_lo[d]) / n_ang[d];
        lo[d]          = ang_lo[d] + (i + double(c.x[dx + d]) / root_len) * w;
        hi[d]          = lo[d] + h * w;
      }
    }
    // the factors as cells of the dx- / ds-dimensional trees
    Cell<dx> spatial_cell(const Cell<D> &c) const
    {
      Cell<dx> k;
      k.tree  = tree_x(c.tree);
      k.level = c.level;
      for (int d = 0; d < dx; ++d)
        k.x[d] = c.x[d];
      return k;
    }
    Cell<ds> angular_cell(const Cell<D> &c) const
    {
      Cell<ds> s;
      s.tree  = tree_s(c.tree);
      s.level = c.level;
      for (int d = 0; d < ds; ++d)
        s.x[d] = c.x[dx + d];
      return s;
    }

    CoarseGeometry<dx>               space;
    std::array<int, ds>              n_ang;
    std::array<double, ds>           ang_lo, ang_hi;
    ProductIndex                     index;
    std::shared_ptr<Connectivity<D>> connectivity;
  };


  // ------------------------------------------------------------------
  // dof partitioner for LinearAlgebra::distributed::Vector
  // ------------------------------------------------------------------
  // Dofs of leaf g (global number) are [g n, (g+1) n). Ghost leaves are
  // sorted by global number, so the ghost dofs of ghost j are stored at
  // local_element(n_local * n + j * n + k). Needs the ghost layer.
  template <int D>
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  make_partitioner(const Forest<D> &forest, const unsigned int n)
  {
    const auto &gfc = forest.global_first_cell();
    const auto  N   = dealii::types::global_dof_index(gfc.back()) * n;
    dealii::IndexSet owned(N), ghost(N);
    owned.add_range(dealii::types::global_dof_index(gfc[forest.rank()]) * n,
                    dealii::types::global_dof_index(gfc[forest.rank() + 1]) * n);
    for (int32_t j = 0; j < forest.n_ghosts(); ++j)
    {
      const auto g = dealii::types::global_dof_index(forest.ghost_global_index(j));
      ghost.add_range(g * n, (g + 1) * n);
    }
    owned.compress();
    ghost.compress();
    return std::make_shared<dealii::Utilities::MPI::Partitioner>(owned, ghost, forest.mpi_comm());
  }

  // Variable number of dofs per leaf (hp): local leaf i has n_dofs[i] dofs,
  // numbered along the curve (rank offsets by MPI_Exscan). The ghost leaves
  // learn the first dof and the count of their owner's leaf by
  // exchange_ghost_data; ghosts are sorted by global leaf number and dof
  // numbers grow with it, so the dofs of ghost j are stored at
  // local_element(n_owned() + ghost_start[j] + k), k < ghost_start[j+1] -
  // ghost_start[j]. Needs the ghost layer.
  struct DofLayout
  {
    std::vector<int64_t> local_start; // leaf i: owned dofs [local_start[i], local_start[i+1])
    std::vector<int64_t> ghost_start; // ghost j: [ghost_start[j], ghost_start[j+1]) after the owned ones
    int64_t              first    = 0; // global number of the first owned dof
    int64_t              n_global = 0;
    int64_t              n_owned() const
    {
      return local_start.back();
    }
    unsigned int n_local_dofs(const int32_t i) const
    {
      return local_start[i + 1] - local_start[i];
    }
    unsigned int n_ghost_dofs(const int32_t j) const
    {
      return ghost_start[j + 1] - ghost_start[j];
    }
  };

  template <int D>
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  make_partitioner(const Forest<D> &forest, const std::vector<unsigned int> &n_dofs, DofLayout &layout)
  {
    if (int32_t(n_dofs.size()) != forest.n_local())
      throw std::runtime_error("make_partitioner: one dof count per local leaf");
    const MPI_Comm comm = forest.mpi_comm();
    layout.local_start.assign(n_dofs.size() + 1, 0);
    for (size_t i = 0; i < n_dofs.size(); ++i)
      layout.local_start[i + 1] = layout.local_start[i] + n_dofs[i];
    int64_t mine = layout.n_owned();
    layout.first = 0;
    MPI_Exscan(&mine, &layout.first, 1, MPI_INT64_T, MPI_SUM, comm);
    if (forest.rank() == 0)
      layout.first = 0;
    MPI_Allreduce(&mine, &layout.n_global, 1, MPI_INT64_T, MPI_SUM, comm);

    std::vector<int64_t> lg(2 * n_dofs.size()), gg(2 * size_t(forest.n_ghosts()));
    for (size_t i = 0; i < n_dofs.size(); ++i)
    {
      lg[2 * i]     = layout.first + layout.local_start[i];
      lg[2 * i + 1] = n_dofs[i];
    }
    forest.exchange_ghost_data(lg.data(), gg.data(), 2);
    const auto       N = dealii::types::global_dof_index(layout.n_global);
    dealii::IndexSet owned(N), ghost(N);
    owned.add_range(layout.first, layout.first + mine);
    layout.ghost_start.assign(forest.n_ghosts() + 1, 0);
    for (int32_t j = 0; j < forest.n_ghosts(); ++j)
    {
      ghost.add_range(gg[2 * j], gg[2 * j] + gg[2 * j + 1]);
      layout.ghost_start[j + 1] = layout.ghost_start[j] + gg[2 * j + 1];
      if (j > 0 && gg[2 * j] < gg[2 * j - 2])
        throw std::runtime_error("make_partitioner: ghost leaves not sorted");
    }
    owned.compress();
    ghost.compress();
    return std::make_shared<dealii::Utilities::MPI::Partitioner>(owned, ghost, comm);
  }


  // ------------------------------------------------------------------
  // tensor-product DG basis Q_px(X) x Q_ps(S)
  // ------------------------------------------------------------------
  // Lagrange polynomials of degree p on the p+1 Gauss-Lobatto nodes of
  // [0,1] (the nodes of FE_DGQ<dim>(p); equidistant for p <= 2),
  // lexicographic (axis 0 fastest) as in FE_DGQ. Hyper dof k = ix * nS + is.
  inline void lagrange_1d(const unsigned int p, const double *nodes, const double t, double *v,
                          double *dv = nullptr)
  {
    if (p == 0) // the constant, node 1/2
    {
      v[0] = 1;
      if (dv)
        dv[0] = 0;
      return;
    }
    if (p == 1)
    {
      v[0] = 1 - t;
      v[1] = t;
      if (dv)
      {
        dv[0] = -1;
        dv[1] = 1;
      }
      return;
    }
    for (unsigned int i = 0; i <= p; ++i)
    {
      double num = 1, den = 1, der = 0;
      for (unsigned int j = 0; j <= p; ++j)
        if (j != i)
        {
          // derivative of the product by the product rule
          der = der * (t - nodes[j]) + num;
          num *= t - nodes[j];
          den *= nodes[i] - nodes[j];
        }
      v[i] = num / den;
      if (dv)
        dv[i] = der / den;
    }
  }

  template <int dim>
  class TensorLagrange
  {
  public:
    static constexpr int dimension = dim;
    explicit TensorLagrange(const unsigned int p)
      : p(p)
      , n(dealii::Utilities::pow(p + 1, dim))
    {
      if (p > max_p)
        throw std::runtime_error("TensorLagrange: degree must be in [0, 8]");
      if (p == 0) // FE_DGQ<dim>(0): one node at the centre
        nodes[0] = 0.5;
      else
      {
        const dealii::QGaussLobatto<1> gl(p + 1);
        for (unsigned int i = 0; i <= p; ++i)
          nodes[i] = gl.point(i)[0];
      }
      for (unsigned int i = 0; i < n; ++i)
      {
        unsigned int r = i;
        for (int d = 0; d < dim; ++d)
        {
          index[i][d] = r % (p + 1);
          r /= p + 1;
        }
      }
    }
    // values (and reference gradients) of all n basis functions at xi
    void values(const double *xi, double *v, dealii::Tensor<1, dim> *grad = nullptr) const
    {
      double l[dim][max_p + 1], dl[dim][max_p + 1];
      for (int d = 0; d < dim; ++d)
        lagrange_1d(p, nodes.data(), xi[d], l[d], grad ? dl[d] : nullptr);
      for (unsigned int i = 0; i < n; ++i)
      {
        double prod = 1;
        for (int d = 0; d < dim; ++d)
          prod *= l[d][index[i][d]];
        v[i] = prod;
        if (grad)
          for (int e = 0; e < dim; ++e)
          {
            double g = 1;
            for (int d = 0; d < dim; ++d)
              g *= d == e ? dl[d][index[i][d]] : l[d][index[i][d]];
            grad[i][e] = g;
          }
      }
    }
    // the reference node of basis function i
    void node(const unsigned int i, double *xi) const
    {
      for (int d = 0; d < dim; ++d)
        xi[d] = nodes[index[i][d]];
    }
    // basis functions not vanishing on face f = 2 d + s (all others do;
    // p = 0: the constant)
    std::vector<unsigned int> face_dofs(const int f) const
    {
      std::vector<unsigned int> r;
      for (unsigned int i = 0; i < n; ++i)
        if (index[i][f / 2] == ((f & 1) ? p : 0u))
          r.push_back(i);
      return r;
    }
    // the 1D index of basis function i along axis d
    unsigned int index_1d(const unsigned int i, const int d) const
    {
      return index[i][d];
    }
    // 1D change of basis to the orthonormal Legendre polynomials
    // L_k(t) = sqrt(2k + 1) P_k(2t - 1) of [0, 1]: u = sum_i c_i l_i has
    // coefficients a_k = sum_i T[k (p + 1) + i] c_i (exact; per axis for the
    // tensor product), so that \int_{[0,1]} u^2 = sum_k a_k^2
    std::vector<double> legendre_transform() const
    {
      const unsigned int         m = p + 1;
      dealii::FullMatrix<double> V(m, m); // V(i, k) = L_k(node_i)
      for (unsigned int i = 0; i < m; ++i)
      {
        const double x = 2 * nodes[i] - 1;
        double       P0 = 1, P1 = x;
        for (unsigned int k = 0; k < m; ++k)
        {
          const double Pk = k == 0 ? 1.0 : (k == 1 ? x : 0.0);
          double       v  = Pk;
          if (k >= 2)
          {
            v  = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k;
            P0 = P1;
            P1 = v;
          }
          V(i, k) = std::sqrt(2.0 * k + 1) * v;
        }
      }
      V.gauss_jordan();
      std::vector<double> T(m * m);
      for (unsigned int k = 0; k < m; ++k)
        for (unsigned int i = 0; i < m; ++i)
          T[k * m + i] = V(k, i);
      return T;
    }

    static constexpr unsigned int max_p = 8;
    const unsigned int            p, n;

  private:
    std::array<double, max_p + 1>                                                     nodes;
    std::array<std::array<unsigned int, dim>, dealii::Utilities::pow(max_p + 1, dim)> index;
  };


  template <int dx, int ds>
  class TensorDG
  {
  public:
    static constexpr int D = dx + ds;

    TensorDG(const unsigned int px, const unsigned int ps)
      : X(px)
      , S(ps)
      , nX(X.n)
      , nS(S.n)
      , n(nX * nS)
    {
      // reference mass matrices (exact Gauss rules) and their inverses
      const auto mass = [](const auto &B, const unsigned int deg, dealii::FullMatrix<double> &M) {
        constexpr int             dim = std::remove_reference_t<decltype(B)>::dimension;
        const dealii::QGauss<dim> q(deg + 1);
        std::vector<double>       v(B.n);
        M.reinit(B.n, B.n);
        for (unsigned int k = 0; k < q.size(); ++k)
        {
          double xi[dim];
          for (int d = 0; d < dim; ++d)
            xi[d] = q.point(k)[d];
          B.values(xi, v.data());
          for (unsigned int i = 0; i < B.n; ++i)
            for (unsigned int j = 0; j < B.n; ++j)
              M(i, j) += q.weight(k) * v[i] * v[j];
        }
        M.gauss_jordan();
      };
      mass(X, px, MX_inv);
      mass(S, ps, MS_inv);
    }

    // u(xi) for the coefficients c (n values) at the D-dimensional point xi
    double value(const double *c, const double *xi) const
    {
      double vx[max_n], vs[max_n];
      X.values(xi, vx);
      S.values(xi + dx, vs);
      double u = 0;
      for (unsigned int i = 0; i < nX; ++i)
        for (unsigned int j = 0; j < nS; ++j)
          u += c[i * nS + j] * vx[i] * vs[j];
      return u;
    }

    // Coefficients of the new leaf `now` from those of the old leaves:
    // interpolation if it lies in an old leaf, L2 projection (in reference
    // coordinates) if it is the union of old leaves.
    void transfer(const Cell<D> &now, const CellRelation &rel, const std::vector<Cell<D>> &old_cells,
                  const double *old_data, double *new_data) const
    {
      transfer(now, rel.kind, rel.n_old, &old_cells[rel.old_first], old_data + std::size_t(rel.old_first) * n,
               new_data);
    }
    // the same for the related old leaves old_cells[0, n_old) with their
    // coefficients at old_data (n per leaf)
    void transfer(const Cell<D> &now, const Relation kind, const int n_old, const Cell<D> *old_cells,
                  const double *old_data, double *new_data) const
    {
      if (kind == Relation::same)
      {
        std::copy(old_data, old_data + n, new_data);
        return;
      }
      if (kind == Relation::refined)
      {
        const NestedMap<D> m = nested_map(now, old_cells[0]);
        for (unsigned int ix = 0; ix < nX; ++ix)
          for (unsigned int is = 0; is < nS; ++is)
          {
            double xi[D];
            X.node(ix, xi);
            S.node(is, xi + dx);
            for (int d = 0; d < D; ++d)
              xi[d] = m.scale * xi[d] + m.shift[d];
            new_data[ix * nS + is] = value(old_data, xi);
          }
        return;
      }
      // b_k = \int_now phi_k u dxi over the old leaves, c = (MX^-1 x MS^-1) b
      std::vector<double>     b(n, 0.0);
      const dealii::QGauss<1> g(std::max(X.p, S.p) + 1);
      const unsigned int      nq = g.size();
      unsigned int            total = 1;
      for (int d = 0; d < D; ++d)
        total *= nq;
      double vx[max_n], vs[max_n];
      for (int k = 0; k < n_old; ++k)
      {
        const Cell<D> &    o = old_cells[k];
        const NestedMap<D> m = nested_map(o, now);
        const double       vol = std::pow(m.scale, D);
        for (unsigned int q = 0; q < total; ++q)
        {
          double       xi_o[D], xi_n[D], w = vol;
          unsigned int r = q;
          for (int d = 0; d < D; ++d)
          {
            xi_o[d] = g.point(r % nq)[0];
            w *= g.weight(r % nq);
            r /= nq;
            xi_n[d] = m.scale * xi_o[d] + m.shift[d];
          }
          const double u = value(old_data + std::size_t(k) * n, xi_o) * w;
          X.values(xi_n, vx);
          S.values(xi_n + dx, vs);
          for (unsigned int ix = 0; ix < nX; ++ix)
            for (unsigned int is = 0; is < nS; ++is)
              b[ix * nS + is] += u * vx[ix] * vs[is];
        }
      }
      for (unsigned int ix = 0; ix < nX; ++ix)
        for (unsigned int is = 0; is < nS; ++is)
        {
          double c = 0;
          for (unsigned int jx = 0; jx < nX; ++jx)
            for (unsigned int js = 0; js < nS; ++js)
              c += MX_inv(ix, jx) * MS_inv(is, js) * b[jx * nS + js];
          new_data[ix * nS + is] = c;
        }
    }

    static constexpr unsigned int max_n = 729; // (max_p + 1)^3
    const TensorLagrange<dx>      X;
    const TensorLagrange<ds>      S;
    const unsigned int       nX, nS, n;
    dealii::FullMatrix<double> MX_inv, MS_inv;
  };


  // hp transfer: the coefficients of the new leaf `now` (basis fe_new) from
  // the old leaves, each with its own basis (old leaf k: *old_fe[k],
  // coefficients old_data + old_start[k]). Interpolation at the new nodes if
  // `now` lies in an old leaf whose space it contains (h refinement and / or
  // a higher degree: exact), otherwise the L2 projection in reference
  // coordinates (union of old leaves, lower degree). With the same basis
  // everywhere this is TensorDG::transfer.
  template <int dx, int ds>
  void transfer_hp(const Cell<dx + ds> &now, const TensorDG<dx, ds> &fe_new, const CellRelation &rel,
                   const std::vector<Cell<dx + ds>> &old_cells, const std::vector<const TensorDG<dx, ds> *> &old_fe,
                   const std::vector<int64_t> &old_start, const double *old_data, double *new_data)
  {
    constexpr int D = dx + ds;
    using FE        = TensorDG<dx, ds>;
    bool same_fe    = true;
    for (int k = 0; k < rel.n_old; ++k)
      same_fe = same_fe && old_fe[rel.old_first + k] == &fe_new;
    if (same_fe)
    {
      fe_new.transfer(now, rel.kind, rel.n_old, &old_cells[rel.old_first], old_data + old_start[rel.old_first],
                      new_data);
      return;
    }
    const FE &o0 = *old_fe[rel.old_first];
    if (rel.kind != Relation::coarsened && fe_new.X.p >= o0.X.p && fe_new.S.p >= o0.S.p)
    {
      const NestedMap<D> m = nested_map(now, old_cells[rel.old_first]);
      for (unsigned int ix = 0; ix < fe_new.nX; ++ix)
        for (unsigned int is = 0; is < fe_new.nS; ++is)
        {
          double xi[D];
          fe_new.X.node(ix, xi);
          fe_new.S.node(is, xi + dx);
          for (int d = 0; d < D; ++d)
            xi[d] = m.scale * xi[d] + m.shift[d];
          new_data[ix * fe_new.nS + is] = o0.value(old_data + old_start[rel.old_first], xi);
        }
      return;
    }
    // b_k = \int_now phi_k u: over the old leaves (coarsened) or over `now`
    // inside the old leaf; c = (MX^-1 x MS^-1) b
    unsigned int pmax = std::max(fe_new.X.p, fe_new.S.p);
    for (int k = 0; k < rel.n_old; ++k)
      pmax = std::max({pmax, old_fe[rel.old_first + k]->X.p, old_fe[rel.old_first + k]->S.p});
    const dealii::QGauss<1> g(pmax + 1);
    const unsigned int      nq    = g.size();
    unsigned int            total = 1;
    for (int d = 0; d < D; ++d)
      total *= nq;
    std::vector<double> b(fe_new.n, 0.0);
    double              vx[FE::max_n], vs[FE::max_n];
    for (int k = 0; k < rel.n_old; ++k)
    {
      const int32_t      o   = rel.old_first + k;
      const bool         sub = rel.kind == Relation::coarsened; // integrate over the old leaf
      const NestedMap<D> m   = sub ? nested_map(old_cells[o], now) : nested_map(now, old_cells[o]);
      const double       vol = sub ? std::pow(m.scale, D) : 1.0;
      for (unsigned int q = 0; q < total; ++q)
      {
        double       xi_q[D], xi_m[D], w = vol;
        unsigned int r = q;
        for (int d = 0; d < D; ++d)
        {
          xi_q[d] = g.point(r % nq)[0];
          w *= g.weight(r % nq);
          r /= nq;
          xi_m[d] = m.scale * xi_q[d] + m.shift[d];
        }
        // sub: xi_q in the old leaf, xi_m in now; otherwise the reverse
        const double *xi_o = sub ? xi_q : xi_m, *xi_n = sub ? xi_m : xi_q;
        const double  u    = old_fe[o]->value(old_data + old_start[o], xi_o) * w;
        fe_new.X.values(xi_n, vx);
        fe_new.S.values(xi_n + dx, vs);
        for (unsigned int ix = 0; ix < fe_new.nX; ++ix)
          for (unsigned int is = 0; is < fe_new.nS; ++is)
            b[ix * fe_new.nS + is] += u * vx[ix] * vs[is];
      }
    }
    for (unsigned int ix = 0; ix < fe_new.nX; ++ix)
      for (unsigned int is = 0; is < fe_new.nS; ++is)
      {
        double c = 0;
        for (unsigned int jx = 0; jx < fe_new.nX; ++jx)
          for (unsigned int js = 0; js < fe_new.nS; ++js)
            c += fe_new.MX_inv(ix, jx) * fe_new.MS_inv(is, js) * b[jx * fe_new.nS + js];
        new_data[ix * fe_new.nS + is] = c;
      }
  }
} // namespace hyperforest

#endif
