// Tests of the core forest against brute force on brick connectivities:
// random refinement, coarsening, 2:1 balance, partition (+ data transfer),
// ghost layer and face neighbours (incl. periodic faces), D = 2 ... 5.
// Run with any number of MPI ranks.
#include <hyperforest/forest.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>

using namespace hyperforest;

static int n_failed = 0;
#define CHECK(cond, msg)                                                                \
  do                                                                                    \
  {                                                                                     \
    if (!(cond))                                                                        \
    {                                                                                   \
      ++n_failed;                                                                       \
      if (n_failed < 20)                                                                \
        std::cerr << "FAILED (rank " << rank << "): " << msg << " [" #cond "]" << std::endl; \
    }                                                                                   \
  } while (0)

static int rank = 0;

template <int D>
uint64_t hash_cell(const Cell<D> &c, const uint64_t salt)
{
  uint64_t h = 1469598103934665603ull ^ salt;
  const auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
    h ^= h >> 29;
  };
  mix(c.tree);
  mix(c.level);
  for (int d = 0; d < D; ++d)
    mix(uint32_t(c.x[d]));
  return h;
}

// brick geometry: global integer box of a cell
template <int D>
struct Brick
{
  std::array<int, D>  n;
  std::array<bool, D> periodic;

  void box(const Cell<D> &c, std::array<int64_t, D> &lo, int64_t &h) const
  {
    int t = c.tree;
    for (int d = 0; d < D; ++d)
    {
      const int i = t % n[d];
      t /= n[d];
      lo[d] = int64_t(i) * root_len + c.x[d];
    }
    h = c.length();
  }
  // does b touch face f of a, sharing a (D-1)-dimensional part of it?
  bool across(const Cell<D> &a, const Cell<D> &b, const int f) const
  {
    std::array<int64_t, D> la, lb;
    int64_t                ha, hb;
    box(a, la, ha);
    box(b, lb, hb);
    const int     m = f / 2;
    const int64_t L = int64_t(n[m]) * root_len;
    if (f & 1)
    {
      int64_t e = la[m] + ha;
      if (periodic[m])
        e %= L;
      if (e != lb[m] || (!periodic[m] && la[m] + ha == L))
        return false;
    }
    else
    {
      int64_t e = lb[m] + hb;
      if (periodic[m])
        e %= L;
      if (e != la[m] || (!periodic[m] && la[m] == 0))
        return false;
    }
    for (int d = 0; d < D; ++d)
      if (d != m && (std::max(la[d], lb[d]) >= std::min(la[d] + ha, lb[d] + hb)))
        return false;
    return true;
  }
};

template <int D>
std::vector<Cell<D>> gather_all(const Forest<D> &forest)
{
  const auto &     gfc = forest.global_first_cell();
  std::vector<int> counts(forest.n_ranks()), displs(forest.n_ranks());
  for (int r = 0; r < forest.n_ranks(); ++r)
  {
    counts[r] = int((gfc[r + 1] - gfc[r]) * sizeof(Cell<D>));
    displs[r] = int(gfc[r] * sizeof(Cell<D>));
  }
  std::vector<Cell<D>> all(gfc.back());
  MPI_Allgatherv(forest.local_cells().data(), counts[forest.rank()], MPI_BYTE, all.data(),
                 counts.data(), displs.data(), MPI_BYTE, forest.mpi_comm());
  return all;
}

template <int D>
void check_forest(Forest<D> &forest, const Brick<D> &brick, const std::string &stage)
{
  const auto all = gather_all(forest);
  CHECK(std::is_sorted(all.begin(), all.end()), stage << ": not sorted");
  for (size_t i = 1; i < all.size(); ++i)
    CHECK(!all[i - 1].contains(all[i]), stage << ": overlapping leaves");
  // volume = number of trees
  long double vol = 0;
  for (const auto &c : all)
    vol += std::pow(0.5L, (long double)(D * c.level));
  CHECK(std::abs(vol - forest.connectivity().n_trees()) < 1e-9, stage << ": volume " << (double)vol);

  // ghost layer and face neighbours against brute force
  forest.build_ghost_layer();
  const int64_t                  my0 = forest.global_first_cell()[forest.rank()];
  const int64_t                  my1 = forest.global_first_cell()[forest.rank() + 1];
  std::vector<FaceNeighbor<D>>   nbs;
  int                            n_expected_ghosts = 0;
  std::map<int64_t, int>         needed;
  for (int32_t i = 0; i < forest.n_local(); ++i)
  {
    const Cell<D> &c = forest.local_cell(i);
    for (int f = 0; f < 2 * D; ++f)
    {
      std::vector<int64_t> expected;
      for (size_t j = 0; j < all.size(); ++j)
        if (int64_t(j) != my0 + i && brick.across(c, all[j], f))
          expected.push_back(j);
      forest.face_neighbors(i, f, nbs);
      CHECK(nbs.size() == expected.size(), stage << ": cell " << i << " face " << f << ": "
                                                  << nbs.size() << " neighbours, expected "
                                                  << expected.size());
      CHECK(expected.empty() == forest.at_boundary(i, f), stage << ": boundary flag");
      double area = 0;
      for (const auto &nb : nbs)
      {
        const Cell<D> &g  = nb.ghost ? forest.ghost_cells()[nb.index] : forest.local_cell(nb.index);
        const int64_t  gi = nb.ghost ? forest.ghost_global_index(nb.index) : my0 + nb.index;
        CHECK(all[gi] == g, stage << ": ghost / local index mismatch");
        CHECK(std::find(expected.begin(), expected.end(), gi) != expected.end(),
              stage << ": unexpected neighbour");
        CHECK(std::abs(int(g.level) - int(c.level)) <= 1 || stage.find("unbalanced") != std::string::npos,
              stage << ": not balanced");
        double a = 1;
        for (int d = 0; d < D; ++d)
          if (d != f / 2)
            a *= nb.hi[d] - nb.lo[d];
        area += a;
        // the centre of the shared face, seen from both cells
        std::array<double, D> xi;
        for (int d = 0; d < D; ++d)
          xi[d] = 0.5 * (nb.lo[d] + nb.hi[d]);
        const auto xn = nb.map(xi);
        CHECK(std::abs(xn[nb.face / 2] - (nb.face & 1)) < 1e-14, stage << ": not on the neighbour face");
        std::array<int64_t, D> lc, lg;
        int64_t                hc, hg;
        brick.box(c, lc, hc);
        brick.box(g, lg, hg);
        for (int d = 0; d < D; ++d)
        {
          CHECK(xn[d] >= -1e-14 && xn[d] <= 1 + 1e-14, stage << ": mapped point outside");
          const double L  = double(brick.n[d]) * root_len;
          double       pc = lc[d] + xi[d] * hc, pg = lg[d] + xn[d] * hg;
          double       diff = std::fmod(std::abs(pc - pg), L);
          diff              = std::min(diff, L - diff);
          CHECK(diff < 1e-6, stage << ": face points differ in dim " << d);
        }
      }
      if (!expected.empty())
        CHECK(std::abs(area - 1) < 1e-12, stage << ": neighbour faces cover " << area);
      for (const int64_t j : expected)
        if (j < my0 || j >= my1)
          needed[j] = 1;
    }
  }
  n_expected_ghosts = needed.size();
  for (const auto &kv : needed)
  {
    bool found = false;
    for (int32_t k = 0; k < forest.n_ghosts() && !found; ++k)
      found = forest.ghost_global_index(k) == kv.first;
    CHECK(found, stage << ": missing ghost");
  }
  int extra = forest.n_ghosts() - n_expected_ghosts;
  MPI_Allreduce(MPI_IN_PLACE, &extra, 1, MPI_INT, MPI_SUM, forest.mpi_comm());
  // ghost data exchange: global indices
  std::vector<int64_t> data(forest.n_local()), gdata(forest.n_ghosts());
  for (int32_t i = 0; i < forest.n_local(); ++i)
    data[i] = forest.global_index(i);
  forest.exchange_ghost_data(data.data(), gdata.data(), 1);
  for (int32_t k = 0; k < forest.n_ghosts(); ++k)
    CHECK(gdata[k] == forest.ghost_global_index(k), stage << ": ghost data");
  const int max_l = forest.global_max_level();
  if (rank == 0)
    std::cout << "    " << stage << ": " << forest.n_global() << " leaves, max level " << max_l
              << ", extra ghosts " << extra << std::endl;
}

// locate_points: random points anywhere; the leaves containing them,
// ghost data exchange over the extended ghost layer, face neighbours after
template <int D>
void check_locate(Forest<D> &forest, const Brick<D> &brick, const std::string &stage)
{
  const auto all = gather_all(forest);
  forest.build_ghost_layer();
  const int           n_trees = forest.connectivity().n_trees();
  std::vector<Cell<D>> pts;
  uint64_t            h = 88172645463325252ull ^ (uint64_t(rank) * 0x9E3779B97F4A7C15ull);
  const auto          rnd = [&]() {
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    return h;
  };
  for (int k = 0; k < 200; ++k)
  {
    Cell<D> p;
    p.level = max_level;
    p.tree  = rnd() % n_trees;
    for (int d = 0; d < D; ++d)
      p.x[d] = rnd() % root_len;
    pts.push_back(p);
  }
  const int32_t n_face_ghosts = forest.n_ghosts();
  const auto    refs          = forest.locate_points(pts);
  for (size_t k = 0; k < pts.size(); ++k)
  {
    const Cell<D> &c  = forest.cell(refs[k]);
    const int64_t  gi = refs[k].ghost ? forest.ghost_global_index(refs[k].index) : forest.global_index(refs[k].index);
    CHECK(c.contains(pts[k]), stage << ": located leaf does not contain the point");
    CHECK(all[gi] == c, stage << ": located leaf has a wrong global index");
  }
  for (int32_t k = 1; k < forest.n_ghosts(); ++k)
    CHECK(forest.ghost_global_index(k - 1) < forest.ghost_global_index(k), stage << ": ghosts not sorted");
  std::vector<int64_t> data(forest.n_local()), gdata(forest.n_ghosts());
  for (int32_t i = 0; i < forest.n_local(); ++i)
    data[i] = forest.global_index(i);
  forest.exchange_ghost_data(data.data(), gdata.data(), 1);
  for (int32_t k = 0; k < forest.n_ghosts(); ++k)
    CHECK(gdata[k] == forest.ghost_global_index(k), stage << ": ghost data after locate_points");
  // face neighbours are still found (indices refer to the extended layer)
  std::vector<FaceNeighbor<D>> nbs;
  for (int32_t i = 0; i < forest.n_local(); ++i)
    for (int f = 0; f < 2 * D; ++f)
    {
      forest.face_neighbors(i, f, nbs);
      for (const auto &nb : nbs)
      {
        const Cell<D> &g = nb.ghost ? forest.ghost_cells()[nb.index] : forest.local_cell(nb.index);
        CHECK(brick.across(forest.local_cell(i), g, f), stage << ": face neighbour after locate_points");
      }
    }
  int64_t added = forest.n_ghosts() - n_face_ghosts;
  MPI_Allreduce(MPI_IN_PLACE, &added, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "    locate_points: " << pts.size() << " points per rank, " << added
              << " leaves added to the ghost layers" << std::endl;
}

template <int D>
void run(const std::array<int, D> &n, const std::array<bool, D> &periodic, const int max_lvl,
         const int pct)
{
  Brick<D> brick{n, periodic};
  auto     conn = std::make_shared<Connectivity<D>>(Connectivity<D>::brick(n, periodic));
  conn->check();
  Forest<D> forest(MPI_COMM_WORLD, conn, 1);
  if (rank == 0)
  {
    std::cout << "D = " << D << ", trees";
    for (int d = 0; d < D; ++d)
      std::cout << (d ? " x " : " ") << n[d] << (periodic[d] ? "p" : "");
    std::cout << std::endl;
  }
  for (int round = 0; round < max_lvl - 1; ++round)
  {
    std::vector<int8_t> flags(forest.n_local());
    for (int32_t i = 0; i < forest.n_local(); ++i)
      flags[i] = hash_cell(forest.local_cell(i), round) % 100 < uint64_t(pct) ? 1 : 0;
    forest.adapt(flags);
  }
  check_forest(forest, brick, "random (unbalanced)");

  const auto before = forest.local_cells();
  const int  rounds = forest.balance();
  if (rank == 0)
    std::cout << "    balance: " << rounds << " rounds" << std::endl;
  const auto rel = Forest<D>::relation(before, forest.local_cells());
  for (size_t j = 0; j < rel.size(); ++j)
    CHECK(rel[j].kind != Relation::coarsened && before[rel[j].old_first].contains(forest.local_cell(j)),
          "relation after balance");
  check_forest(forest, brick, "balanced");
  check_locate(forest, brick, "balanced");

  // partition: uniform, then weighted by level; checksum and data transfer
  const uint64_t sum = forest.checksum();
  std::vector<int64_t> data(forest.n_local());
  for (int32_t i = 0; i < forest.n_local(); ++i)
    data[i] = forest.global_index(i);
  std::vector<double> w(forest.n_local());
  for (int32_t i = 0; i < forest.n_local(); ++i)
    w[i] = 1 + forest.local_cell(i).level;
  for (int pass = 0; pass < 2; ++pass)
  {
    const auto           old_gfc = pass == 0 ? forest.partition(&w) : forest.partition();
    std::vector<int64_t> moved(forest.n_local());
    transfer_fixed(MPI_COMM_WORLD, old_gfc, forest.global_first_cell(), data.data(), moved.data(), 1);
    for (int32_t i = 0; i < forest.n_local(); ++i)
      CHECK(moved[i] == forest.global_index(i), "transfer_fixed");
    data.swap(moved);
    CHECK(forest.checksum() == sum, "checksum changed by partition");
    if (pass == 0)
    {
      w.assign(forest.n_local(), 0);
      double local = 0, mx = 0, tot = 0;
      for (int32_t i = 0; i < forest.n_local(); ++i)
        local += 1 + forest.local_cell(i).level;
      MPI_Allreduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&local, &tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (rank == 0)
        std::cout << "    weighted partition: max/mean weight " << mx / (tot / forest.n_ranks())
                  << std::endl;
    }
  }
  check_forest(forest, brick, "repartitioned");

  // coarsen roughly half of the families, rebalance
  {
    std::vector<int8_t> flags(forest.n_local());
    for (int32_t i = 0; i < forest.n_local(); ++i)
      flags[i] = hash_cell(forest.local_cell(i).parent(), 99) % 2 ? -1 : 0;
    const auto old = forest.local_cells();
    forest.adapt(flags);
    const auto rel = Forest<D>::relation(old, forest.local_cells());
    int64_t    n_coarsened = 0;
    for (size_t j = 0; j < rel.size(); ++j)
      if (rel[j].kind == Relation::coarsened)
      {
        ++n_coarsened;
        CHECK(rel[j].n_old == (1 << D), "coarsened family size");
        for (int k = 0; k < rel[j].n_old; ++k)
          CHECK(forest.local_cell(j).contains(old[rel[j].old_first + k]), "coarsened relation");
      }
      else
        CHECK(rel[j].kind == Relation::same && old[rel[j].old_first] == forest.local_cell(j),
              "relation after coarsening");
    MPI_Allreduce(MPI_IN_PLACE, &n_coarsened, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
      std::cout << "    coarsened " << n_coarsened << " families" << std::endl;
    forest.balance();
    forest.partition();
    check_forest(forest, brick, "coarsened + balanced");
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    std::cout << "hyperforest core test on " << size << " ranks" << std::endl;

  // Morton order of cell_from_morton agrees with compare()
  {
    for (uint64_t m = 0; m + 1 < (1u << 12); ++m)
      CHECK(compare(cell_from_morton<4>(0, 3, m), cell_from_morton<4>(0, 3, m + 1)) < 0, "morton order");
    const Cell<4> c = cell_from_morton<4>(0, 3, 1234);
    CHECK(morton_index(c) == 1234, "morton_index");
    for (int k = 0; k < 16; ++k)
      CHECK(c.child(k).child_id() == k && c.child(k).parent() == c, "child / parent");
  }

  run<2>({2, 3}, {false, true}, 6, 25);
  run<3>({2, 1, 2}, {false, true, false}, 5, 20);
  run<4>({1, 1, 4, 1}, {false, false, true, false}, 4, 15);
  run<5>({1, 1, 1, 4, 2}, {false, false, false, true, false}, 3, 10);

  int failed = n_failed;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << (failed ? "FAILED: " + std::to_string(failed) + " checks" : std::string("all checks passed"))
              << std::endl;
  MPI_Finalize();
  return failed ? 1 : 0;
}
