/* ---------------------------------------------------------------------
 *
 * hyperforest: the distributed linear forest.
 *
 * Every rank stores the leaves it owns as one array sorted along the
 * space-filling curve (tree by tree, Morton order within a tree), as in
 * p4est. Rank r owns the leaves between global_first_position[r] and
 * global_first_position[r+1]; leaves are numbered globally along the curve
 * (global_first_cell). Operations (all collective unless noted):
 *
 *   Forest(comm, connectivity, level)  uniform forest, evenly partitioned
 *   adapt(flags)            refine (+1) / coarsen (-1, whole local families)
 *   balance()               2:1 balance across faces
 *   partition(weights)      repartition along the curve (optionally weighted)
 *   build_ghost_layer()     the remote leaves sharing a face with a local
 *                           leaf; exchange_ghost_data() sends cell data
 *   face_neighbors(i, f)    (local, needs the ghost layer) the leaves on the
 *                           other side of face f of local leaf i, with the
 *                           reference-coordinate maps for face integrals
 *   relation(old, new)      (local) how the leaves before and after
 *                           adapt()/balance() are related, for data transfer
 *   transfer_fixed(...)     moves cell data along with partition()
 *   transfer_variable(...)  the same with a variable number of values per
 *                           leaf (hp)
 *
 * Parallelism is MPI only (like p4est); there is no shared state between
 * ranks except through these collectives.
 *
 * ---------------------------------------------------------------------
 */
#ifndef hyperforest_forest_h
#define hyperforest_forest_h

#include <hyperforest/cell.h>
#include <hyperforest/connectivity.h>

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hyperforest
{
  // One leaf on the other side of a face of a local leaf c.
  // Reference coordinates: xi_nb[perm[d]] = scale[d] * xi_c[d] + shift[d].
  // [lo, hi] is the part of the face shared with this neighbour in the
  // reference coordinates of c (lo[n] = hi[n] = side for the normal axis n).
  template <int D>
  struct FaceNeighbor
  {
    int32_t               index; // local index, or ghost index if ghost
    bool                  ghost;
    int8_t                face;  // face of the neighbour on the shared hyperplane
    int8_t                level;
    std::array<int8_t, D> perm;
    std::array<double, D> scale, shift;
    std::array<double, D> lo, hi;

    // the reference point of the neighbour corresponding to the point xi of c
    std::array<double, D> map(const std::array<double, D> &xi) const
    {
      std::array<double, D> r;
      for (int d = 0; d < D; ++d)
        r[perm[d]] = scale[d] * xi[d] + shift[d];
      return r;
    }
  };

  // How a new leaf (after adapt / balance) is related to the old leaves
  enum class Relation : int8_t
  {
    same,      // identical to old leaf old_first
    refined,   // a descendant of old leaf old_first
    coarsened  // the union of old leaves [old_first, old_first + n_old)
  };
  struct CellRelation
  {
    Relation kind;
    int32_t  old_first;
    int32_t  n_old;
  };

  struct LeafRef
  {
    int32_t index = -1; // -1: not found
    bool    ghost = false;
  };


  template <int D>
  class Forest
  {
  public:
    using CellD = Cell<D>;
    static constexpr int n_faces         = 2 * D;
    static constexpr int n_children      = 1 << D;
    static constexpr int n_face_children = 1 << (D - 1);

    Forest(MPI_Comm comm, std::shared_ptr<const Connectivity<D>> conn, const int level)
      : comm(comm)
      , conn(std::move(conn))
    {
      MPI_Comm_rank(comm, &my_rank);
      MPI_Comm_size(comm, &n_procs);
      if (D * level > 62)
        throw std::runtime_error("hyperforest: initial level too large");
      const int64_t per_tree = int64_t(1) << (D * level);
      const int64_t N        = per_tree * this->conn->n_trees();
      const int64_t q = N / n_procs, rem = N % n_procs;
      const int64_t begin = my_rank * q + std::min<int64_t>(my_rank, rem);
      const int64_t end   = begin + q + (my_rank < rem ? 1 : 0);
      cells.reserve(end - begin);
      for (int64_t g = begin; g < end; ++g)
        cells.push_back(cell_from_morton<D>(int32_t(g / per_tree), level, uint64_t(g % per_tree)));
      update_global_info();
    }

    Forest(const Forest &) = default;
    Forest &operator=(const Forest &) = default;

    // ------------------------------------------------------------ queries
    MPI_Comm mpi_comm() const
    {
      return comm;
    }
    int rank() const
    {
      return my_rank;
    }
    int n_ranks() const
    {
      return n_procs;
    }
    const Connectivity<D> &connectivity() const
    {
      return *conn;
    }
    std::shared_ptr<const Connectivity<D>> connectivity_ptr() const
    {
      return conn;
    }
    int32_t n_local() const
    {
      return cells.size();
    }
    int64_t n_global() const
    {
      return gfc.back();
    }
    const std::vector<CellD> &local_cells() const
    {
      return cells;
    }
    const CellD &local_cell(const int32_t i) const
    {
      return cells[i];
    }
    // global number of the leaves of each rank: [gfc[r], gfc[r+1])
    const std::vector<int64_t> &global_first_cell() const
    {
      return gfc;
    }
    int64_t global_index(const int32_t i) const
    {
      return gfc[my_rank] + i;
    }
    const std::vector<CellD> &global_first_position() const
    {
      return positions;
    }
    int global_max_level() const
    {
      int l = 0;
      for (const auto &c : cells)
        l = std::max<int>(l, c.level);
      MPI_Allreduce(MPI_IN_PLACE, &l, 1, MPI_INT, MPI_MAX, comm);
      return l;
    }

    // the rank owning the point p (a cell of max_level)
    int owner(const CellD &p) const
    {
      const auto it = std::upper_bound(positions.begin(), positions.end(), p,
                                       [](const CellD &a, const CellD &b) { return compare(a, b) < 0; });
      return int(it - positions.begin()) - 1;
    }

    // The leaf containing the point p, among the local leaves and the ghosts
    LeafRef find_leaf(const CellD &p) const
    {
      LeafRef r;
      if (owner(p) == my_rank)
        r.index = find_in(cells, p);
      else
      {
        r.ghost = true;
        r.index = find_in(ghosts, p);
      }
      return r;
    }
    const CellD &cell(const LeafRef &r) const
    {
      return r.ghost ? ghosts[r.index] : cells[r.index];
    }

    // The same-size region nc across face f of c and its face nf on the
    // shared hyperplane, with the map of tree coordinates (identity inside
    // a tree). False at the domain boundary.
    bool neighbor_region(const CellD &c, const int f, CellD &nc, int &nf, TreeAffine<D> &T) const
    {
      const CellD q = c.face_neighbor(f);
      if (q.inside_root())
      {
        nc = q;
        nf = f ^ 1;
        T  = TreeAffine<D>::identity();
        return true;
      }
      const TreeFace<D> &tf = conn->neighbor(c.tree, f);
      if (tf.tree < 0)
        return false;
      T  = tf.affine(f);
      nc = T(q, tf.tree);
      nf = tf.face;
      return true;
    }
    bool at_boundary(const int32_t i, const int f) const
    {
      const CellD &c = cells[i];
      return c.touches_root_face(f) && conn->neighbor(c.tree, f).tree < 0;
    }

    // --------------------------------------------------------- refinement
    // flags[i] > 0: refine leaf i once (up to max_level); flags[i] < 0:
    // coarsen if all 2^D siblings are local leaves flagged < 0. The
    // partition is kept (the first point of every rank does not move).
    // Returns true if any rank changed its leaves.
    bool adapt(const std::vector<int8_t> &flags, const int max_lvl = max_level)
    {
      if (flags.size() != cells.size())
        throw std::runtime_error("hyperforest::adapt: wrong flag vector size");
      std::vector<CellD> out;
      out.reserve(cells.size());
      bool         changed = false;
      const size_t n       = cells.size();
      for (size_t i = 0; i < n;)
      {
        const CellD &c = cells[i];
        if (flags[i] < 0 && c.level > 0 && c.child_id() == 0 && i + n_children <= n)
        {
          bool family = true;
          for (int k = 1; k < n_children && family; ++k)
            family = flags[i + k] < 0 && cells[i + k] == c.sibling(k);
          if (family)
          {
            out.push_back(c.parent());
            i += n_children;
            changed = true;
            continue;
          }
        }
        if (flags[i] > 0 && c.level < max_lvl)
        {
          for (int k = 0; k < n_children; ++k)
            out.push_back(c.child(k));
          changed = true;
        }
        else
          out.push_back(c);
        ++i;
      }
      cells.swap(out);
      int any = changed;
      MPI_Allreduce(MPI_IN_PLACE, &any, 1, MPI_INT, MPI_LOR, comm);
      if (any)
        update_global_info();
      return any;
    }

    // 2:1 balance across faces (all 2D faces of the D-cells): leaves sharing
    // a (D-1)-dimensional face differ by at most one level. Only refines.
    // Ripple algorithm: refine every leaf that has a face neighbour two or
    // more levels finer, rebuild the ghost layer, repeat until no rank
    // changes. Returns the number of rounds that refined.
    // After the first round only the leaves next to leaves refined in the
    // previous round, and the leaves next to other ranks, are checked.
    int balance()
    {
      int               rounds = 0;
      std::vector<char> todo(cells.size(), 1);
      CellD             nc;
      int               nf;
      TreeAffine<D>     T;
      while (true)
      {
        build_ghost_layer();
        for (const int32_t k : mirror_indices) // the other side may have refined
          todo[k] = 1;
        std::vector<int8_t> flags(cells.size(), 0);
        for (size_t i = 0; i < cells.size(); ++i)
          if (todo[i] && cells[i].level + 2 <= max_level)
            for (int f = 0; f < n_faces && !flags[i]; ++f)
              if (neighbor_region(cells[i], f, nc, nf, T) && finer_by_two(cells[i], nc, nf))
                flags[i] = 1;
        const std::vector<CellD> before = cells;
        if (!adapt(flags))
          break;
        ++rounds;
        // next round: the new leaves and the local leaves next to them
        todo.assign(cells.size(), 0);
        const auto rel = relation(before, cells);
        for (size_t j = 0; j < cells.size(); ++j)
          if (rel[j].kind == Relation::refined)
          {
            todo[j] = 1;
            for (int f = 0; f < n_faces; ++f)
              if (neighbor_region(cells[j], f, nc, nf, T) && owner(face_point(nc, nf)) == my_rank)
              {
                const int32_t L = find_in(cells, face_point(nc, nf));
                if (L >= 0)
                  todo[L] = 1;
              }
          }
      }
      return rounds;
    }

    // How the leaves `now` (after adapt / balance on the same partition)
    // relate to the leaves `before`
    static std::vector<CellRelation> relation(const std::vector<CellD> &before,
                                              const std::vector<CellD> &now)
    {
      std::vector<CellRelation> rel(now.size());
      size_t                    i = 0;
      for (size_t j = 0; j < now.size(); ++j)
      {
        const CellD &n = now[j];
        if (i >= before.size())
          throw std::runtime_error("hyperforest::relation: leaves do not match");
        const CellD &o = before[i];
        if (n == o)
        {
          rel[j] = {Relation::same, int32_t(i), 1};
          ++i;
        }
        else if (o.contains(n))
        {
          rel[j] = {Relation::refined, int32_t(i), 1};
          if (n.last_descendant() == o.last_descendant())
            ++i;
        }
        else if (n.contains(o))
        {
          const size_t first = i;
          while (i < before.size() && n.contains(before[i]))
            ++i;
          rel[j] = {Relation::coarsened, int32_t(first), int32_t(i - first)};
        }
        else
          throw std::runtime_error("hyperforest::relation: leaves do not match");
      }
      return rel;
    }

    // ---------------------------------------------------------- partition
    // Redistributes the leaves so that every rank gets the same number of
    // leaves (or the same total weight, weights[i] >= 0 for local leaf i).
    // The global order is unchanged. Returns the old global_first_cell,
    // for transfer_fixed().
    std::vector<int64_t> partition(const std::vector<double> *weights = nullptr)
    {
      const std::vector<int64_t> old_gfc = gfc;
      std::vector<int>           send(n_procs, 0), recv(n_procs, 0);
      if (!weights)
      {
        const int64_t N = gfc.back(), q = N / n_procs, rem = N % n_procs;
        const auto    first = [&](const int64_t r) { return r * q + std::min<int64_t>(r, rem); };
        const int64_t o0 = gfc[my_rank], o1 = gfc[my_rank + 1];
        for (int r = 0; r < n_procs; ++r)
          send[r] = int(std::max<int64_t>(0, std::min(o1, first(r + 1)) - std::max(o0, first(r))));
      }
      else
      {
        if (weights->size() != cells.size())
          throw std::runtime_error("hyperforest::partition: wrong weight vector size");
        double local = 0, offset = 0, total = 0;
        for (const double w : *weights)
          local += w;
        MPI_Exscan(&local, &offset, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (my_rank == 0)
          offset = 0;
        MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, comm);
        double cum = offset;
        for (const double w : *weights)
        {
          const double mid = cum + 0.5 * w;
          cum += w;
          int dest = total > 0 ? int(mid / total * n_procs) : 0;
          dest     = std::min(std::max(dest, 0), n_procs - 1);
          ++send[dest];
        }
      }
      MPI_Alltoall(send.data(), 1, MPI_INT, recv.data(), 1, MPI_INT, comm);
      std::vector<CellD> out(std::accumulate(recv.begin(), recv.end(), size_t(0)));
      alltoallv_bytes(cells.data(), send, out.data(), recv, sizeof(CellD));
      cells.swap(out);
      update_global_info();
      return old_gfc;
    }

    // --------------------------------------------------------- ghost layer
    // Collects the remote leaves that share a face with a local leaf (and
    // possibly a few more: a leaf is sent to every rank owning a part of the
    // same-size region across one of its faces).
    void build_ghost_layer()
    {
      std::vector<std::vector<int32_t>> to(n_procs);
      CellD                             nc;
      int                               nf;
      TreeAffine<D>                     T;
      for (size_t i = 0; i < cells.size(); ++i)
        for (int f = 0; f < n_faces; ++f)
        {
          if (!neighbor_region(cells[i], f, nc, nf, T))
            continue;
          const int r0 = owner(nc.first_descendant()), r1 = owner(nc.last_descendant());
          for (int r = r0; r <= r1; ++r)
            if (r != my_rank && gfc[r + 1] > gfc[r] && (to[r].empty() || to[r].back() != int32_t(i)))
              to[r].push_back(i);
        }
      mirror_offsets.assign(n_procs + 1, 0);
      for (int r = 0; r < n_procs; ++r)
        mirror_offsets[r + 1] = mirror_offsets[r] + to[r].size();
      mirror_indices.resize(mirror_offsets.back());
      for (int r = 0; r < n_procs; ++r)
        std::copy(to[r].begin(), to[r].end(), mirror_indices.begin() + mirror_offsets[r]);

      struct Msg
      {
        CellD   c;
        int32_t index;
      };
      std::vector<int> send(n_procs), recv(n_procs);
      for (int r = 0; r < n_procs; ++r)
        send[r] = to[r].size();
      MPI_Alltoall(send.data(), 1, MPI_INT, recv.data(), 1, MPI_INT, comm);
      std::vector<Msg> sbuf(mirror_indices.size()), rbuf(std::accumulate(recv.begin(), recv.end(), size_t(0)));
      for (size_t k = 0; k < mirror_indices.size(); ++k)
        sbuf[k] = {cells[mirror_indices[k]], mirror_indices[k]};
      alltoallv_bytes(sbuf.data(), send, rbuf.data(), recv, sizeof(Msg));

      ghost_offsets.assign(n_procs + 1, 0);
      for (int r = 0; r < n_procs; ++r)
        ghost_offsets[r + 1] = ghost_offsets[r] + recv[r];
      ghosts.resize(rbuf.size());
      ghost_owner.resize(rbuf.size());
      ghost_remote.resize(rbuf.size());
      for (int r = 0; r < n_procs; ++r)
        for (int k = ghost_offsets[r]; k < ghost_offsets[r + 1]; ++k)
        {
          ghosts[k]       = rbuf[k].c;
          ghost_owner[k]  = r;
          ghost_remote[k] = rbuf[k].index;
        }
      ghosts_valid = true;
    }

    // Collective. The leaves containing the given points (cells of
    // max_level, any tree, any owner); leaves of other ranks are added to
    // the ghost layer (and to the mirrors of their owners), so that ghost
    // data exchange and distributed vectors built afterwards include them.
    // For couplings that are not face neighbours, e.g. reflection of a
    // direction at a wall or refraction at an interface in phase space.
    // Ghost indices change: query face neighbours again afterwards.
    // build_ghost_layer() (also called by balance()) drops these ghosts.
    std::vector<LeafRef> locate_points(const std::vector<CellD> &points)
    {
      if (!ghosts_valid)
        throw std::runtime_error("hyperforest: ghost layer not built");
      // requests to the owners (duplicates are harmless)
      std::vector<std::vector<CellD>> req(n_procs);
      for (const CellD &p : points)
      {
        const int r = owner(p);
        if (r != my_rank)
          req[r].push_back(p);
      }
      std::vector<int> send(n_procs), recv(n_procs);
      for (int r = 0; r < n_procs; ++r)
        send[r] = req[r].size();
      MPI_Alltoall(send.data(), 1, MPI_INT, recv.data(), 1, MPI_INT, comm);
      std::vector<CellD> sbuf, rbuf(std::accumulate(recv.begin(), recv.end(), size_t(0)));
      for (const auto &v : req)
        sbuf.insert(sbuf.end(), v.begin(), v.end());
      alltoallv_bytes(sbuf.data(), send, rbuf.data(), recv, sizeof(CellD));

      // owners: the local leaves asked for become mirrors of the asking rank
      struct Msg
      {
        CellD   c;
        int32_t index;
      };
      std::vector<Msg>                  reply(rbuf.size());
      std::vector<std::vector<int32_t>> to(n_procs);
      for (int r = 0; r < n_procs; ++r)
        to[r].assign(mirror_indices.begin() + mirror_offsets[r], mirror_indices.begin() + mirror_offsets[r + 1]);
      for (int r = 0, k = 0; r < n_procs; ++r)
        for (int m = 0; m < recv[r]; ++m, ++k)
        {
          const int32_t i = find_in(cells, rbuf[k]);
          if (i < 0)
            throw std::runtime_error("hyperforest::locate_points: point not owned");
          reply[k] = {cells[i], i};
          to[r].push_back(i);
        }
      mirror_offsets.assign(n_procs + 1, 0);
      mirror_indices.clear();
      for (int r = 0; r < n_procs; ++r)
      {
        std::sort(to[r].begin(), to[r].end());
        to[r].erase(std::unique(to[r].begin(), to[r].end()), to[r].end());
        mirror_indices.insert(mirror_indices.end(), to[r].begin(), to[r].end());
        mirror_offsets[r + 1] = mirror_indices.size();
      }
      std::vector<Msg> answers(sbuf.size());
      alltoallv_bytes(reply.data(), recv, answers.data(), send, sizeof(Msg));

      // askers: merge the new leaves into the ghosts (sorted by global index)
      struct G
      {
        int64_t gi;
        CellD   c;
        int     owner;
        int32_t remote;
      };
      std::vector<G> all;
      all.reserve(ghosts.size() + answers.size());
      for (size_t j = 0; j < ghosts.size(); ++j)
        all.push_back({gfc[ghost_owner[j]] + ghost_remote[j], ghosts[j], ghost_owner[j], ghost_remote[j]});
      for (int r = 0, k = 0; r < n_procs; ++r)
        for (int m = 0; m < send[r]; ++m, ++k)
          all.push_back({gfc[r] + answers[k].index, answers[k].c, r, answers[k].index});
      std::sort(all.begin(), all.end(), [](const G &a, const G &b) { return a.gi < b.gi; });
      all.erase(std::unique(all.begin(), all.end(), [](const G &a, const G &b) { return a.gi == b.gi; }),
                all.end());
      ghosts.resize(all.size());
      ghost_owner.resize(all.size());
      ghost_remote.resize(all.size());
      ghost_offsets.assign(n_procs + 1, 0);
      for (size_t j = 0; j < all.size(); ++j)
      {
        ghosts[j]       = all[j].c;
        ghost_owner[j]  = all[j].owner;
        ghost_remote[j] = all[j].remote;
        ++ghost_offsets[all[j].owner + 1];
      }
      for (int r = 0; r < n_procs; ++r)
        ghost_offsets[r + 1] += ghost_offsets[r];

      std::vector<LeafRef> result(points.size());
      for (size_t k = 0; k < points.size(); ++k)
      {
        result[k] = find_leaf(points[k]);
        if (result[k].index < 0)
          throw std::runtime_error("hyperforest::locate_points: leaf not found");
      }
      return result;
    }

    bool has_ghost_layer() const
    {
      return ghosts_valid;
    }
    int32_t n_ghosts() const
    {
      return ghosts.size();
    }
    const std::vector<CellD> &ghost_cells() const
    {
      return ghosts;
    }
    int ghost_owner_rank(const int32_t j) const
    {
      return ghost_owner[j];
    }
    int32_t ghost_remote_index(const int32_t j) const
    {
      return ghost_remote[j];
    }
    int64_t ghost_global_index(const int32_t j) const
    {
      return gfc[ghost_owner[j]] + ghost_remote[j];
    }
    // local leaves sent to rank r: mirror_indices[mirror_offsets[r] ...]
    const std::vector<int32_t> &mirrors() const
    {
      return mirror_indices;
    }
    const std::vector<int32_t> &mirror_rank_offsets() const
    {
      return mirror_offsets;
    }

    // ghost[j * n + k] = (local data of ghost leaf j on its owner)[k]
    template <typename T>
    void exchange_ghost_data(const T *local, T *ghost, const int n) const
    {
      if (!ghosts_valid)
        throw std::runtime_error("hyperforest: ghost layer not built");
      std::vector<T> sbuf(mirror_indices.size() * size_t(n));
      for (size_t k = 0; k < mirror_indices.size(); ++k)
        std::copy(local + size_t(mirror_indices[k]) * n, local + size_t(mirror_indices[k] + 1) * n,
                  sbuf.begin() + k * n);
      std::vector<MPI_Request> req;
      for (int r = 0; r < n_procs; ++r)
        if (ghost_offsets[r + 1] > ghost_offsets[r])
        {
          req.emplace_back();
          MPI_Irecv(ghost + size_t(ghost_offsets[r]) * n,
                    int((ghost_offsets[r + 1] - ghost_offsets[r]) * n * sizeof(T)), MPI_BYTE, r, 17,
                    comm, &req.back());
        }
      for (int r = 0; r < n_procs; ++r)
        if (mirror_offsets[r + 1] > mirror_offsets[r])
        {
          req.emplace_back();
          MPI_Isend(sbuf.data() + size_t(mirror_offsets[r]) * n,
                    int((mirror_offsets[r + 1] - mirror_offsets[r]) * n * sizeof(T)), MPI_BYTE, r, 17,
                    comm, &req.back());
        }
      MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
    }

    // ----------------------------------------------------- face neighbours
    // The leaves across face f of local leaf i (empty at the boundary): one
    // leaf of the same or a coarser level, or all finer leaves touching the
    // face. Needs the ghost layer.
    void face_neighbors(const int32_t i, const int f, std::vector<FaceNeighbor<D>> &out) const
    {
      out.clear();
      if (!ghosts_valid)
        throw std::runtime_error("hyperforest: ghost layer not built");
      const CellD & c = cells[i];
      CellD         nc;
      int           nf;
      TreeAffine<D> T;
      if (!neighbor_region(c, f, nc, nf, T))
        return;
      const LeafRef L = find_leaf(face_point(nc, nf));
      if (L.index < 0)
        throw std::runtime_error("hyperforest::face_neighbors: incomplete ghost layer");
      if (cell(L).level <= c.level)
        out.push_back(make_neighbor(c, f, T, L, nf));
      else
        collect_finer(c, f, T, nc, nf, out);
    }

    // order-sensitive checksum of the global leaf sequence, independent of
    // the partition
    uint64_t checksum() const
    {
      uint64_t s = 0;
      for (size_t i = 0; i < cells.size(); ++i)
      {
        uint64_t h = 1469598103934665603ull;
        const auto mix = [&](const uint64_t v) {
          h ^= v;
          h *= 1099511628211ull;
        };
        mix(cells[i].tree);
        mix(cells[i].level);
        for (int d = 0; d < D; ++d)
          mix(uint32_t(cells[i].x[d]));
        s += h * uint64_t(global_index(i) + 1);
      }
      MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_UINT64_T, MPI_SUM, comm);
      return s;
    }

  private:
    // Is there a leaf of level >= c.level + 2 touching face nf of the
    // same-size neighbour region nc of c? (Needs the ghost layer.)
    bool finer_by_two(const CellD &c, const CellD &nc, const int nf) const
    {
      // fast path: nc lies in this rank and holds at most 2^D leaves, so
      // none of them is finer than c.level + 1
      if (owner(nc.first_descendant()) == my_rank && owner(nc.last_descendant()) == my_rank)
      {
        const auto less = [](const CellD &a, const CellD &b) { return compare(a, b) < 0; };
        const auto lo   = std::lower_bound(cells.begin(), cells.end(), nc, less);
        const auto hi   = std::upper_bound(lo, cells.end(), nc.last_descendant(), less);
        if (hi - lo <= n_children)
          return false;
      }
      // a leaf of level >= c.level + 2 touches the face iff one of the
      // children of nc on the face is subdivided
      for (int k = 0; k < n_face_children; ++k)
      {
        const CellD   h = nc.child(face_child<D>(nf, k));
        const LeafRef L = find_leaf(face_point(h, nf));
        if (L.index < 0)
          throw std::runtime_error("hyperforest::balance: incomplete ghost layer");
        if (cell(L).level > c.level + 1)
          return true;
      }
      return false;
    }

    static int32_t find_in(const std::vector<CellD> &v, const CellD &p)
    {
      auto it = std::upper_bound(v.begin(), v.end(), p,
                                 [](const CellD &a, const CellD &b) { return compare(a, b) < 0; });
      if (it == v.begin())
        return -1;
      --it;
      return it->contains(p) ? int32_t(it - v.begin()) : -1;
    }

    FaceNeighbor<D> make_neighbor(const CellD &c, const int f, const TreeAffine<D> &T,
                                  const LeafRef &L, const int nf) const
    {
      const CellD &   g = cell(L);
      FaceNeighbor<D> nb;
      nb.index         = L.index;
      nb.ghost         = L.ghost;
      nb.face          = nf;
      nb.level         = g.level;
      const double hc = c.length(), hg = g.length();
      const int    n  = f / 2;
      for (int d = 0; d < D; ++d)
      {
        const int e  = T.perm[d];
        nb.perm[d]   = e;
        nb.scale[d]  = T.a[d] * hc / hg;
        nb.shift[d]  = (double(T.a[d]) * c.x[d] + double(T.b[d]) - g.x[e]) / hg;
        if (d == n)
          nb.lo[d] = nb.hi[d] = f & 1;
        else if (g.level <= c.level)
        {
          nb.lo[d] = 0;
          nb.hi[d] = 1;
        }
        else
        {
          // g's extent along axis e, mapped back: X = (X' - b) / a
          const double x0 = (double(g.x[e]) - T.b[d]) * T.a[d], x1 = (double(g.x[e]) + hg - T.b[d]) * T.a[d];
          nb.lo[d] = (std::min(x0, x1) - c.x[d]) / hc;
          nb.hi[d] = (std::max(x0, x1) - c.x[d]) / hc;
        }
      }
      return nb;
    }

    void collect_finer(const CellD &c, const int f, const TreeAffine<D> &T, const CellD &h,
                       const int nf, std::vector<FaceNeighbor<D>> &out) const
    {
      for (int k = 0; k < n_face_children; ++k)
      {
        const CellD   hh = h.child(face_child<D>(nf, k));
        const LeafRef L  = find_leaf(face_point(hh, nf));
        if (L.index < 0)
          throw std::runtime_error("hyperforest::face_neighbors: incomplete ghost layer");
        if (cell(L).level <= hh.level)
          out.push_back(make_neighbor(c, f, T, L, nf));
        else
          collect_finer(c, f, T, hh, nf, out);
      }
    }

    void update_global_info()
    {
      int64_t n = cells.size();
      gfc.assign(n_procs + 1, 0);
      MPI_Allgather(&n, 1, MPI_INT64_T, gfc.data() + 1, 1, MPI_INT64_T, comm);
      for (int r = 0; r < n_procs; ++r)
        gfc[r + 1] += gfc[r];
      struct First
      {
        CellD p;
        int   has;
      } mine{}, *all = nullptr;
      std::vector<First> v(n_procs);
      all = v.data();
      if (!cells.empty())
      {
        mine.p   = cells[0].first_descendant();
        mine.has = 1;
      }
      MPI_Allgather(&mine, sizeof(First), MPI_BYTE, all, sizeof(First), MPI_BYTE, comm);
      positions.resize(n_procs + 1);
      positions[n_procs]       = CellD();
      positions[n_procs].tree  = conn->n_trees();
      positions[n_procs].level = max_level;
      for (int r = n_procs - 1; r >= 0; --r)
        positions[r] = v[r].has ? v[r].p : positions[r + 1];
      ghosts_valid = false;
    }

    void alltoallv_bytes(const void *sbuf, const std::vector<int> &send, void *rbuf,
                         const std::vector<int> &recv, const size_t size) const
    {
      std::vector<int> sc(n_procs), sd(n_procs, 0), rc(n_procs), rd(n_procs, 0);
      for (int r = 0; r < n_procs; ++r)
      {
        sc[r] = int(send[r] * size);
        rc[r] = int(recv[r] * size);
        if (r > 0)
        {
          sd[r] = sd[r - 1] + sc[r - 1];
          rd[r] = rd[r - 1] + rc[r - 1];
        }
      }
      MPI_Alltoallv(sbuf, sc.data(), sd.data(), MPI_BYTE, rbuf, rc.data(), rd.data(), MPI_BYTE, comm);
    }

    MPI_Comm                               comm;
    int                                    my_rank = 0, n_procs = 1;
    std::shared_ptr<const Connectivity<D>> conn;
    std::vector<CellD>                     cells;
    std::vector<int64_t>                   gfc;
    std::vector<CellD>                     positions;

    bool                 ghosts_valid = false;
    std::vector<CellD>   ghosts;
    std::vector<int>     ghost_owner;
    std::vector<int32_t> ghost_remote, ghost_offsets;
    std::vector<int32_t> mirror_indices, mirror_offsets;
  };


  // Moves data attached to the leaves (n values of type T per leaf) from the
  // partition old_gfc to the partition new_gfc of the same global sequence.
  template <typename T>
  void transfer_fixed(MPI_Comm comm, const std::vector<int64_t> &old_gfc,
                      const std::vector<int64_t> &new_gfc, const T *src, T *dst, const int n)
  {
    int me, P;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &P);
    std::vector<int> sc(P), sd(P), rc(P), rd(P);
    const int64_t    bytes = int64_t(n) * sizeof(T);
    for (int r = 0; r < P; ++r)
    {
      const int64_t s0 = std::max(old_gfc[me], new_gfc[r]), s1 = std::min(old_gfc[me + 1], new_gfc[r + 1]);
      sc[r] = int(std::max<int64_t>(0, s1 - s0) * bytes);
      sd[r] = int(std::max<int64_t>(0, s0 - old_gfc[me]) * bytes);
      const int64_t r0 = std::max(new_gfc[me], old_gfc[r]), r1 = std::min(new_gfc[me + 1], old_gfc[r + 1]);
      rc[r] = int(std::max<int64_t>(0, r1 - r0) * bytes);
      rd[r] = int(std::max<int64_t>(0, r0 - new_gfc[me]) * bytes);
      if (sc[r] == 0)
        sd[r] = 0;
      if (rc[r] == 0)
        rd[r] = 0;
    }
    MPI_Alltoallv(src, sc.data(), sd.data(), MPI_BYTE, dst, rc.data(), rd.data(), MPI_BYTE, comm);
  }


  // The same for a variable number of values per leaf (e.g. dofs of hp
  // elements): old local leaf i has n_old[i] values, stored one leaf after
  // the other in src. On return n_new holds the counts of the new local
  // leaves and dst their values, in the same layout.
  template <typename T>
  void transfer_variable(MPI_Comm comm, const std::vector<int64_t> &old_gfc, const std::vector<int64_t> &new_gfc,
                         const std::vector<int> &n_old, const T *src, std::vector<int> &n_new, std::vector<T> &dst)
  {
    int me, P;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &P);
    n_new.resize(new_gfc[me + 1] - new_gfc[me]);
    transfer_fixed(comm, old_gfc, new_gfc, n_old.data(), n_new.data(), 1);
    std::vector<int64_t> so(n_old.size() + 1, 0), ro(n_new.size() + 1, 0);
    for (size_t i = 0; i < n_old.size(); ++i)
      so[i + 1] = so[i] + n_old[i];
    for (size_t i = 0; i < n_new.size(); ++i)
      ro[i + 1] = ro[i] + n_new[i];
    dst.resize(ro.back());
    std::vector<int> sc(P, 0), sd(P, 0), rc(P, 0), rd(P, 0);
    const auto       bytes = [](const int64_t v) {
      const int64_t b = v * int64_t(sizeof(T));
      if (b > int64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("hyperforest::transfer_variable: message too large");
      return int(b);
    };
    for (int r = 0; r < P; ++r)
    {
      const int64_t s0 = std::max(old_gfc[me], new_gfc[r]), s1 = std::min(old_gfc[me + 1], new_gfc[r + 1]);
      if (s1 > s0)
      {
        sd[r] = bytes(so[s0 - old_gfc[me]]);
        sc[r] = bytes(so[s1 - old_gfc[me]] - so[s0 - old_gfc[me]]);
      }
      const int64_t r0 = std::max(new_gfc[me], old_gfc[r]), r1 = std::min(new_gfc[me + 1], old_gfc[r + 1]);
      if (r1 > r0)
      {
        rd[r] = bytes(ro[r0 - new_gfc[me]]);
        rc[r] = bytes(ro[r1 - new_gfc[me]] - ro[r0 - new_gfc[me]]);
      }
    }
    MPI_Alltoallv(src, sc.data(), sd.data(), MPI_BYTE, dst.data(), rc.data(), rd.data(), MPI_BYTE, comm);
  }
} // namespace hyperforest

#endif
