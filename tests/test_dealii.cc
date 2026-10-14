// Tests of the deal.II adapter:
//  - connectivity from deal.II coarse meshes with rotated / reflected
//    neighbours (hyper_ball, hyper_shell, gmsh quads): after random
//    refinement and balancing, the physical point of every shared face is
//    the same seen from both sides,
//  - the same for phase space X x S (dx = 2, 3; S = periodic angular box),
//  - ghost values through LinearAlgebra::distributed::Vector,
//  - TensorLagrange against FE_DGQ (degrees 0-3), TensorDG transfer
//    exactness, the Legendre transform, hp transfer between degrees,
//    variable dofs per leaf (partitioner, ghosts, transfer_variable).
#include <hyperforest/dealii.h>

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <fstream>
#include <iostream>

using namespace hyperforest;
using dealii::Point;

static int rank = 0, n_failed = 0;
#define CHECK(cond, msg)                                                                    \
  do                                                                                        \
  {                                                                                         \
    if (!(cond))                                                                            \
    {                                                                                       \
      if (++n_failed < 20)                                                                  \
        std::cerr << "FAILED (rank " << rank << "): " << msg << " [" #cond "]" << std::endl; \
    }                                                                                       \
  } while (0)

template <int D>
void random_refine(Forest<D> &forest, const int rounds, const int pct)
{
  for (int r = 0; r < rounds; ++r)
  {
    std::vector<int8_t> flags(forest.n_local());
    for (int32_t i = 0; i < forest.n_local(); ++i)
    {
      const auto &c = forest.local_cell(i);
      uint64_t    h = 1469598103934665603ull + 77 * r;
      for (int d = 0; d < D; ++d)
        h = (h ^ uint32_t(c.x[d])) * 1099511628211ull;
      h = (h ^ c.tree) * 1099511628211ull;
      flags[i] = (h >> 20) % 100 < uint64_t(pct);
    }
    forest.adapt(flags);
  }
  forest.balance();
  forest.partition();
  forest.build_ghost_layer();
}

// the physical point of the reference point xi of leaf c: spatial part by
// the d-linear map of the coarse cell, angular part (if any) in the box
template <int dx, int ds>
std::array<double, dx + ds> physical(const Cell<dx + ds> &c, const std::array<double, dx + ds> &xi,
                                     const CoarseGeometry<dx> &geo, const PhaseSpace<dx, ds> *ps)
{
  std::array<double, dx + ds> r;
  Point<dx>                   zeta;
  const double                h = double(c.length()) / root_len;
  for (int d = 0; d < dx; ++d)
    zeta[d] = double(c.x[d]) / root_len + h * xi[d];
  const Point<dx> x = geo.map(ps ? ps->tree_x(c.tree) : c.tree, zeta);
  for (int d = 0; d < dx; ++d)
    r[d] = x[d];
  if (ps)
  {
    std::array<double, ds> lo, hi;
    ps->angular_box(c, lo, hi);
    for (int d = 0; d < ds; ++d)
      r[dx + d] = lo[d] + xi[dx + d] * (hi[d] - lo[d]);
  }
  return r;
}

template <int dx, int ds>
void check_faces(const Forest<dx + ds> &forest, const CoarseGeometry<dx> &geo,
                 const PhaseSpace<dx, ds> *ps, const std::string &name)
{
  constexpr int                D = dx + ds;
  std::vector<FaceNeighbor<D>> nbs;
  double                       max_err = 0, area_err = 0;
  int64_t                      n_pairs = 0, n_cross = 0;
  for (int32_t i = 0; i < forest.n_local(); ++i)
  {
    const auto &c = forest.local_cell(i);
    for (int f = 0; f < 2 * D; ++f)
    {
      forest.face_neighbors(i, f, nbs);
      double area = 0;
      for (const auto &nb : nbs)
      {
        const auto &g = nb.ghost ? forest.ghost_cells()[nb.index] : forest.local_cell(nb.index);
        n_cross += g.tree != c.tree;
        double a = 1;
        for (int d = 0; d < D; ++d)
          if (d != f / 2)
            a *= nb.hi[d] - nb.lo[d];
        area += a;
        // a few points of the shared face
        for (int k = 0; k < 3; ++k)
        {
          std::array<double, D> xi;
          for (int d = 0; d < D; ++d)
            xi[d] = nb.lo[d] + (0.17 + 0.31 * k + 0.07 * d) * (nb.hi[d] - nb.lo[d]);
          xi[f / 2] = f & 1;
          const auto xn = nb.map(xi);
          const auto pc = physical<dx, ds>(c, xi, geo, ps), pg = physical<dx, ds>(g, xn, geo, ps);
          for (int d = 0; d < D; ++d)
          {
            double e = std::abs(pc[d] - pg[d]);
            if (d == dx && ps) // periodic phi
              e = std::min(e, std::abs(e - 2 * dealii::numbers::PI));
            max_err = std::max(max_err, e);
          }
        }
        ++n_pairs;
      }
      if (!nbs.empty())
        area_err = std::max(area_err, std::abs(area - 1));
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &area_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_pairs, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_cross, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  CHECK(max_err < 1e-12, name << ": face points differ by " << max_err);
  CHECK(area_err < 1e-12, name << ": faces not covered");
  if (rank == 0)
    std::cout << "  " << name << ": " << forest.n_global() << " leaves, " << n_pairs
              << " face pairs (" << n_cross << " across trees), max point mismatch " << max_err
              << std::endl;
}

template <int dim>
void test_space(dealii::Triangulation<dim> &tria, const std::string &name, const int rounds)
{
  auto conn = std::make_shared<Connectivity<dim>>(connectivity_from_coarse_mesh(tria));
  int  rotated = 0;
  for (int t = 0; t < conn->n_trees(); ++t)
    for (int f = 0; f < 2 * dim; ++f)
      if (conn->neighbor(t, f).tree >= 0)
      {
        bool id = conn->neighbor(t, f).face == (f ^ 1);
        for (int d = 0; d < dim; ++d)
          id = id && conn->neighbor(t, f).flip[d] == 0 && (d == f / 2 || conn->neighbor(t, f).perm[d] == d);
        rotated += !id;
      }
  if (rank == 0)
    std::cout << name << ": " << conn->n_trees() << " trees, " << rotated
              << " tree faces with a rotated / reflected neighbour" << std::endl;
  CoarseGeometry<dim> geo(tria);
  Forest<dim>         forest(MPI_COMM_WORLD, conn, 1);
  random_refine(forest, rounds, 30);
  check_faces<dim, 0>(forest, geo, nullptr, name);
}

template <int dx>
void test_phase_space(dealii::Triangulation<dx> &tria, const std::string &name)
{
  constexpr int D = dx + 2;
  for (const bool angle_major : {true, false})
  {
    PhaseSpace<dx, 2> ps(tria, {4, 2}, {-dealii::numbers::PI, -1.0}, {dealii::numbers::PI, 1.0},
                         {true, false}, angle_major);
    Forest<D> forest(MPI_COMM_WORLD, ps.connectivity, 1);
    random_refine(forest, 2, 12);
    check_faces<dx, 2>(forest, ps.space, &ps,
                       name + (angle_major ? " x S (angle-major)" : " x S (space-major)"));

    // ghost layer extended by arbitrary phase-space points (as for
    // reflected directions), then ghost values through a distributed vector
    {
      std::vector<Cell<D>> pts;
      uint64_t             h = 1234567 + 7919 * rank;
      for (int k = 0; k < 100; ++k)
      {
        Cell<D> p;
        p.level = max_level;
        h       = h * 6364136223846793005ull + 1442695040888963407ull;
        p.tree  = (h >> 33) % ps.connectivity->n_trees();
        for (int d = 0; d < D; ++d)
        {
          h      = h * 6364136223846793005ull + 1442695040888963407ull;
          p.x[d] = (h >> 33) % root_len;
        }
        pts.push_back(p);
      }
      const auto refs = forest.locate_points(pts);
      for (size_t k = 0; k < pts.size(); ++k)
        CHECK(forest.cell(refs[k]).contains(pts[k]), "locate_points");
    }
    const unsigned int n    = 3;
    const auto         part = make_partitioner(forest, n);
    dealii::LinearAlgebra::distributed::Vector<double> v(part);
    for (int32_t i = 0; i < forest.n_local(); ++i)
      for (unsigned int k = 0; k < n; ++k)
        v.local_element(i * n + k) = forest.global_index(i) * 10.0 + k;
    v.update_ghost_values();
    for (int32_t j = 0; j < forest.n_ghosts(); ++j)
      for (unsigned int k = 0; k < n; ++k)
        CHECK(v.local_element((forest.n_local() + j) * n + k) == forest.ghost_global_index(j) * 10.0 + k,
              "ghost value");
  }
}

int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // basis against FE_DGQ
  for (unsigned int p = 1; p <= 3; ++p)
  {
    const dealii::FE_DGQ<2> fe(p);
    const TensorLagrange<2> B(p);
    std::vector<double>              v(B.n);
    std::vector<dealii::Tensor<1, 2>> g(B.n);
    const double                     xi[2] = {0.3, 0.71};
    B.values(xi, v.data(), g.data());
    double err = 0;
    for (unsigned int i = 0; i < B.n; ++i)
      err = std::max({err, std::abs(v[i] - fe.shape_value(i, Point<2>(xi[0], xi[1]))),
                      (g[i] - fe.shape_grad(i, Point<2>(xi[0], xi[1]))).norm()});
    CHECK(err < 1e-12, "TensorLagrange != FE_DGQ, p = " << p);
  }
  {
    const dealii::FE_DGQ<3> fe(2);
    const TensorLagrange<3> B(2);
    std::vector<double>     v(B.n);
    const double            xi[3] = {0.3, 0.71, 0.45};
    B.values(xi, v.data());
    double err = 0;
    for (unsigned int i = 0; i < B.n; ++i)
      err = std::max(err, std::abs(v[i] - fe.shape_value(i, Point<3>(xi[0], xi[1], xi[2]))));
    CHECK(err < 1e-12, "TensorLagrange<3> != FE_DGQ<3>");
  }

  // transfer: a Q1 x Q1 polynomial survives refinement and coarsening
  {
    const TensorDG<2, 2> fe(1, 1);
    Cell<4>              c;
    c.level   = 2;
    c.x       = {0, root_len / 4, root_len / 2, 0};
    const auto u = [](const double *z) {
      return (1 + 2 * z[0] - z[1] + 0.5 * z[0] * z[1]) * (3 - z[2] + 0.25 * z[3] * z[2]);
    };
    std::vector<double> coef(fe.n);
    for (unsigned int ix = 0; ix < fe.nX; ++ix)
      for (unsigned int is = 0; is < fe.nS; ++is)
      {
        double xi[4];
        fe.X.node(ix, xi);
        fe.S.node(is, xi + 2);
        coef[ix * fe.nS + is] = u(xi);
      }
    std::vector<Cell<4>> old_cells{c}, kids;
    std::vector<double>  kid_data(16 * fe.n);
    for (int k = 0; k < 16; ++k)
    {
      kids.push_back(c.child(k));
      fe.transfer(c.child(k), {Relation::refined, 0, 1}, old_cells, coef.data(), &kid_data[k * fe.n]);
    }
    double err = 0;
    for (int k = 0; k < 16; ++k)
    {
      const NestedMap<4> m = nested_map(kids[k], c);
      const double       xi[4] = {0.2, 0.9, 0.4, 0.6};
      double             xo[4];
      for (int d = 0; d < 4; ++d)
        xo[d] = m.scale * xi[d] + m.shift[d];
      err = std::max(err, std::abs(fe.value(&kid_data[k * fe.n], xi) - u(xo)));
    }
    std::vector<double> back(fe.n);
    fe.transfer(c, {Relation::coarsened, 0, 16}, kids, kid_data.data(), back.data());
    for (unsigned int i = 0; i < fe.n; ++i)
      err = std::max(err, std::abs(back[i] - coef[i]));
    CHECK(err < 1e-12, "TensorDG transfer error " << err);
    if (rank == 0)
      std::cout << "TensorDG<2,2> refine + coarsen transfer error " << err << std::endl;
  }

  // degree 0 (FE_DGQ(0)) and the Legendre transform (Parseval)
  {
    const dealii::FE_DGQ<2> fe(0);
    const TensorLagrange<2> B(0);
    double                  v[1];
    const double            xi[2] = {0.3, 0.71};
    B.values(xi, v);
    CHECK(B.n == 1 && std::abs(v[0] - fe.shape_value(0, Point<2>(xi[0], xi[1]))) < 1e-14, "TensorLagrange(0)");
    CHECK(B.face_dofs(0).size() == 1 && B.face_dofs(3).size() == 1, "TensorLagrange(0) face dofs");
    double err = 0;
    for (unsigned int p = 0; p <= 4; ++p)
    {
      const TensorLagrange<1>  L(p);
      const std::vector<double> T = L.legendre_transform();
      std::vector<double>       c(p + 1), lv(p + 1);
      for (unsigned int i = 0; i <= p; ++i)
        c[i] = std::sin(1.0 + 3.0 * i);
      double a2 = 0, u2 = 0;
      for (unsigned int k = 0; k <= p; ++k)
      {
        double a = 0;
        for (unsigned int i = 0; i <= p; ++i)
          a += T[k * (p + 1) + i] * c[i];
        a2 += a * a;
      }
      const dealii::QGauss<1> q(p + 1);
      for (unsigned int k = 0; k < q.size(); ++k)
      {
        const double t = q.point(k)[0];
        L.values(&t, lv.data());
        double u = 0;
        for (unsigned int i = 0; i <= p; ++i)
          u += c[i] * lv[i];
        u2 += q.weight(k) * u * u;
      }
      err = std::max(err, std::abs(a2 - u2));
    }
    CHECK(err < 1e-12, "legendre_transform (Parseval) error " << err);
  }

  // hp transfer: Q1(x) x Q0(s) data refined into children of mixed degrees
  // (interpolation), coarsened back from them (projection), and lowered
  // from Q2 x Q2 to Q1 x Q0 on the same cell (projection): all exact
  {
    const TensorDG<2, 2> f10(1, 0), f11(1, 1), f21(2, 1), f12(1, 2), f22(2, 2);
    const TensorDG<2, 2> *mixed[4] = {&f11, &f21, &f12, &f22};
    Cell<4>               c;
    c.level    = 2;
    c.x        = {root_len / 4, 0, root_len / 2, root_len / 4};
    const auto u = [](const double *z) { return 1 + 2 * z[0] - z[1] + 0.5 * z[0] * z[1]; }; // Q1 in x
    const auto fill = [&](const TensorDG<2, 2> &fe, const NestedMap<4> &m, double *coef) {
      for (unsigned int ix = 0; ix < fe.nX; ++ix)
        for (unsigned int is = 0; is < fe.nS; ++is)
        {
          double xi[4];
          fe.X.node(ix, xi);
          fe.S.node(is, xi + 2);
          for (int d = 0; d < 4; ++d)
            xi[d] = m.scale * xi[d] + m.shift[d];
          coef[ix * fe.nS + is] = u(xi);
        }
    };
    const NestedMap<4> id{1.0, {0, 0, 0, 0}};
    std::vector<double> parent(f10.n);
    fill(f10, id, parent.data());
    std::vector<Cell<4>>                old_cells{c}, kids;
    std::vector<const TensorDG<2, 2> *> old_fe{&f10}, kid_fe;
    std::vector<int64_t>                old_start{0, int64_t(f10.n)}, kid_start{0};
    std::vector<double>                 kid_data;
    double                              err = 0;
    for (int k = 0; k < 16; ++k)
    {
      kids.push_back(c.child(k));
      kid_fe.push_back(mixed[k % 4]);
      kid_data.resize(kid_start.back() + kid_fe.back()->n);
      transfer_hp<2, 2>(kids.back(), *kid_fe.back(), {Relation::refined, 0, 1}, old_cells, old_fe, old_start,
                        parent.data(), &kid_data[kid_start.back()]);
      std::vector<double> exact(kid_fe.back()->n);
      fill(*kid_fe.back(), nested_map(kids.back(), c), exact.data());
      for (unsigned int i = 0; i < exact.size(); ++i)
        err = std::max(err, std::abs(exact[i] - kid_data[kid_start.back() + i]));
      kid_start.push_back(kid_data.size());
    }
    std::vector<double> back(f10.n);
    transfer_hp<2, 2>(c, f10, {Relation::coarsened, 0, 16}, kids, kid_fe, kid_start, kid_data.data(), back.data());
    for (unsigned int i = 0; i < f10.n; ++i)
      err = std::max(err, std::abs(back[i] - parent[i]));
    std::vector<double> high(f22.n), low(f10.n);
    fill(f22, id, high.data());
    const std::vector<const TensorDG<2, 2> *> hfe{&f22};
    const std::vector<int64_t>                hstart{0, int64_t(f22.n)};
    transfer_hp<2, 2>(c, f10, {Relation::same, 0, 1}, old_cells, hfe, hstart, high.data(), low.data());
    for (unsigned int i = 0; i < f10.n; ++i)
      err = std::max(err, std::abs(low[i] - parent[i]));
    CHECK(err < 1e-12, "transfer_hp error " << err);
    if (rank == 0)
      std::cout << "transfer_hp (mixed degrees, refine / coarsen / lower) error " << err << std::endl;
  }

  // variable dofs per leaf: ghost values through the partitioner of
  // make_partitioner(forest, n_i, layout), and transfer_variable along with
  // a weighted partition
  {
    dealii::Triangulation<2> square;
    dealii::GridGenerator::hyper_cube(square);
    auto      conn = std::make_shared<Connectivity<2>>(connectivity_from_coarse_mesh(square));
    Forest<2> forest(MPI_COMM_WORLD, conn, 3);
    random_refine(forest, 2, 30);
    const auto n_of = [](const int64_t g) { return unsigned(1 + g % 5); };
    std::vector<unsigned int> nd(forest.n_local());
    for (int32_t i = 0; i < forest.n_local(); ++i)
      nd[i] = n_of(forest.global_index(i));
    DofLayout  lay;
    const auto part = make_partitioner(forest, nd, lay);
    dealii::LinearAlgebra::distributed::Vector<double> v(part);
    for (int32_t i = 0; i < forest.n_local(); ++i)
      for (unsigned int k = 0; k < nd[i]; ++k)
        v.local_element(lay.local_start[i] + k) = 10.0 * forest.global_index(i) + k;
    v.update_ghost_values();
    int bad = 0;
    for (int32_t j = 0; j < forest.n_ghosts(); ++j)
    {
      const int64_t g = forest.ghost_global_index(j);
      bad += lay.n_ghost_dofs(j) != n_of(g);
      for (unsigned int k = 0; k < lay.n_ghost_dofs(j); ++k)
        bad += v.local_element(lay.n_owned() + lay.ghost_start[j] + k) != 10.0 * g + k;
    }
    std::vector<double> data(v.begin(), v.begin() + lay.n_owned()), w(forest.n_local()), moved;
    std::vector<int>    n_old(nd.begin(), nd.end()), n_new;
    for (int32_t i = 0; i < forest.n_local(); ++i)
      w[i] = double(nd[i]) * nd[i];
    const auto old_gfc = forest.partition(&w);
    transfer_variable(MPI_COMM_WORLD, old_gfc, forest.global_first_cell(), n_old, data.data(), n_new, moved);
    size_t pos = 0;
    for (int32_t i = 0; i < forest.n_local(); ++i)
    {
      bad += unsigned(n_new[i]) != n_of(forest.global_index(i));
      for (int k = 0; k < n_new[i]; ++k)
        bad += moved[pos++] != 10.0 * forest.global_index(i) + k;
    }
    bad += pos != moved.size();
    CHECK(bad == 0, "variable dofs: " << bad << " wrong ghost / moved values");
    if (rank == 0)
      std::cout << "variable dofs per leaf: " << lay.n_global << " dofs, ghosts and transfer ok" << std::endl;
  }

  {
    dealii::Triangulation<2> ball;
    dealii::GridGenerator::hyper_ball(ball);
    test_space(ball, "hyper_ball<2>", 4);
    test_phase_space(ball, "hyper_ball<2>");
  }
  {
    dealii::Triangulation<2> mesh, flat;
    dealii::GridIn<2>        gi;
    gi.attach_triangulation(mesh);
    std::ifstream in(std::string(SOURCE_DIR) + "/tests/meshes/square_with_hole.msh");
    if (in)
    {
      gi.read_msh(in);
      test_space(mesh, "gmsh square_with_hole", 3);
      test_phase_space(mesh, "gmsh square_with_hole");
    }
  }
  {
    dealii::Triangulation<3> ball;
    dealii::GridGenerator::hyper_ball(ball);
    test_space(ball, "hyper_ball<3>", 3);
    dealii::Triangulation<3> shell;
    dealii::GridGenerator::hyper_shell(shell, Point<3>(), 0.5, 1.0, 12);
    test_space(shell, "hyper_shell<3> (12 cells)", 3);
    test_phase_space(ball, "hyper_ball<3>");
  }

  int failed = n_failed;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << (failed ? "FAILED: " + std::to_string(failed) + " checks" : std::string("all checks passed"))
              << std::endl;
  return failed ? 1 : 0;
}
