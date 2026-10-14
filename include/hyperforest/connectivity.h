/* ---------------------------------------------------------------------
 *
 * hyperforest: connectivity of the trees (the coarse "macro" mesh).
 *
 * Every tree is the image of the unit cube [0,1]^D. Two trees sharing a
 * face are related by an affine map of their integer coordinates that is
 * a signed permutation of the axes:
 *     X'[perm[d]] = a[d] X[d] + b[d],   a[d] = +-1.
 * Only face connections are stored (no edge / corner connectivity): the
 * forest offers face neighbours, face ghosts and 2:1 face balance, which
 * is what DG discretisations need.
 *
 * Connectivities are built as
 *   - brick(n, periodic): an n_0 x ... x n_{D-1} array of trees,
 *   - product(A, B): the tensor product of a D1- and a D2-dimensional
 *     connectivity (e.g. a spatial mesh x an angular mesh; the dimensions
 *     of A come first). Face transforms act on the A or the B dimensions
 *     only, so every tree of the product is K x S with K a tree of A and S
 *     a tree of B,
 *   - from a deal.II coarse mesh (see dealii.h).
 *
 * ---------------------------------------------------------------------
 */
#ifndef hyperforest_connectivity_h
#define hyperforest_connectivity_h

#include <hyperforest/cell.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace hyperforest
{
  // X'[perm[d]] = a[d] X[d] + b[d] (continuous tree coordinates)
  template <int D>
  struct TreeAffine
  {
    std::array<int8_t, D>  perm;
    std::array<int8_t, D>  a;
    std::array<int64_t, D> b;

    static TreeAffine identity()
    {
      TreeAffine t;
      for (int d = 0; d < D; ++d)
      {
        t.perm[d] = d;
        t.a[d]    = 1;
        t.b[d]    = 0;
      }
      return t;
    }

    // image of the cell q (the box [x, x + h)) in the other tree
    Cell<D> operator()(const Cell<D> &q, const int32_t new_tree) const
    {
      Cell<D> r;
      r.tree        = new_tree;
      r.level       = q.level;
      const int64_t h = q.length();
      for (int d = 0; d < D; ++d)
        r.x[perm[d]] = int32_t(a[d] > 0 ? q.x[d] + b[d] : b[d] - q.x[d] - h);
      return r;
    }

    TreeAffine inverse() const
    {
      TreeAffine t;
      for (int d = 0; d < D; ++d)
      {
        t.perm[perm[d]] = d;
        t.a[perm[d]]    = a[d];
        t.b[perm[d]]    = -a[d] * b[d];
      }
      return t;
    }
  };


  // Face f = 2 n + s of a tree is glued to face f' of tree `tree`; the
  // tangential axis d of this tree becomes axis perm[d] of the neighbour,
  // reversed if flip[d]. perm[n] is the neighbour's normal axis.
  template <int D>
  struct TreeFace
  {
    int32_t               tree = -1; // -1: domain boundary
    int8_t                face = -1;
    std::array<int8_t, D> perm{};
    std::array<int8_t, D> flip{};

    // the map of integer coordinates from this tree to the neighbour, for
    // cells across face f of this tree
    TreeAffine<D> affine(const int f) const
    {
      TreeAffine<D> t;
      const int     n = f / 2, s = f & 1, np = face / 2, sp = face & 1;
      for (int d = 0; d < D; ++d)
      {
        t.perm[d] = perm[d];
        t.a[d]    = flip[d] ? -1 : 1;
        t.b[d]    = flip[d] ? root_len : 0;
      }
      t.perm[n] = np;
      if (s == 1 && sp == 0)
      {
        t.a[n] = 1;
        t.b[n] = -int64_t(root_len);
      }
      else if (s == 1 && sp == 1)
      {
        t.a[n] = -1;
        t.b[n] = 2 * int64_t(root_len);
      }
      else if (s == 0 && sp == 0)
      {
        t.a[n] = -1;
        t.b[n] = 0;
      }
      else
      {
        t.a[n] = 1;
        t.b[n] = root_len;
      }
      return t;
    }
  };


  template <int D>
  class Connectivity
  {
  public:
    static constexpr int n_faces = 2 * D;

    int n_trees() const
    {
      return faces.size();
    }
    const TreeFace<D> &neighbor(const int t, const int f) const
    {
      return faces[t][f];
    }
    int boundary_id(const int t, const int f) const
    {
      return boundary_ids[t][f];
    }

    void resize(const int n)
    {
      faces.assign(n, std::array<TreeFace<D>, 2 * D>());
      boundary_ids.assign(n, std::array<int, 2 * D>());
      for (auto &b : boundary_ids)
        b.fill(0);
    }

    // n[0] x ... x n[D-1] trees, tree index sum_d i_d prod_{e<d} n_e
    // (dimension 0 fastest). Boundary faces get the id f (face number).
    static Connectivity brick(const std::array<int, D> &n, const std::array<bool, D> &periodic)
    {
      Connectivity c;
      int          total = 1;
      std::array<int, D> stride;
      for (int d = 0; d < D; ++d)
      {
        stride[d] = total;
        total *= n[d];
      }
      c.resize(total);
      for (int t = 0; t < total; ++t)
        for (int f = 0; f < 2 * D; ++f)
        {
          const int d = f / 2, s = f & 1;
          const int i = (t / stride[d]) % n[d];
          int       j = s ? i + 1 : i - 1;
          if (j < 0 || j >= n[d])
          {
            if (!periodic[d])
            {
              c.boundary_ids[t][f] = f;
              continue;
            }
            j = (j + n[d]) % n[d];
          }
          TreeFace<D> &tf = c.faces[t][f];
          tf.tree         = t + (j - i) * stride[d];
          tf.face         = f ^ 1;
          for (int e = 0; e < D; ++e)
          {
            tf.perm[e] = e;
            tf.flip[e] = 0;
          }
        }
      return c;
    }

    // throws if the face connections are not symmetric
    void check() const
    {
      for (int t = 0; t < n_trees(); ++t)
        for (int f = 0; f < 2 * D; ++f)
        {
          const TreeFace<D> &tf = faces[t][f];
          if (tf.tree < 0)
            continue;
          const TreeFace<D> &back = faces.at(tf.tree)[tf.face];
          if (back.tree != t || back.face != f)
            throw std::runtime_error("connectivity: face " + std::to_string(f) + " of tree " +
                                     std::to_string(t) + " is not glued back");
          for (int d = 0; d < D; ++d)
            if (d != f / 2 &&
                (back.perm[tf.perm[d]] != d || back.flip[tf.perm[d]] != tf.flip[d]))
              throw std::runtime_error("connectivity: inconsistent face transform at tree " +
                                       std::to_string(t) + ", face " + std::to_string(f));
          // a cell just outside face f maps to a cell just inside face f'
          Cell<D> q;
          q.tree  = t;
          q.level = 1;
          if (f & 1)
            q.x[f / 2] = root_len - q.length();
          q = q.face_neighbor(f);
          const Cell<D> r = tf.affine(f)(q, tf.tree);
          if (!r.inside_root() || !r.touches_root_face(tf.face))
            throw std::runtime_error("connectivity: bad transform at tree " + std::to_string(t));
        }
    }

    std::vector<std::array<TreeFace<D>, 2 * D>> faces;
    std::vector<std::array<int, 2 * D>>         boundary_ids; // faces without neighbour
  };


  // Tree numbering of a product connectivity A x B
  struct ProductIndex
  {
    int  nA = 0, nB = 0;
    bool b_major = false; // true: tree = tb * nA + ta, false: tree = ta * nB + tb

    int tree(const int ta, const int tb) const
    {
      return b_major ? tb * nA + ta : ta * nB + tb;
    }
    int ta(const int t) const
    {
      return b_major ? t % nA : t / nB;
    }
    int tb(const int t) const
    {
      return b_major ? t / nA : t % nB;
    }
  };


  // The tensor product A x B: dimensions 0..D1-1 from A, D1..D1+D2-1 from B.
  // With b_major the trees are numbered B-major, so that a partition of
  // the forest along the space-filling curve splits B first (for phase
  // space with B = angles: ranks own direction sectors).
  template <int D1, int D2>
  Connectivity<D1 + D2> product(const Connectivity<D1> &A,
                                const Connectivity<D2> &B,
                                const bool              b_major,
                                ProductIndex *          index = nullptr)
  {
    constexpr int D = D1 + D2;
    ProductIndex  pi{A.n_trees(), B.n_trees(), b_major};
    Connectivity<D> c;
    c.resize(pi.nA * pi.nB);
    for (int ta = 0; ta < pi.nA; ++ta)
      for (int tb = 0; tb < pi.nB; ++tb)
      {
        const int t = pi.tree(ta, tb);
        for (int f = 0; f < 2 * D; ++f)
        {
          TreeFace<D> &tf = c.faces[t][f];
          for (int d = 0; d < D; ++d)
          {
            tf.perm[d] = d;
            tf.flip[d] = 0;
          }
          if (f < 2 * D1)
          {
            const TreeFace<D1> &af = A.neighbor(ta, f);
            c.boundary_ids[t][f]   = A.boundary_id(ta, f);
            if (af.tree < 0)
              continue;
            tf.tree = pi.tree(af.tree, tb);
            tf.face = af.face;
            for (int d = 0; d < D1; ++d)
            {
              tf.perm[d] = af.perm[d];
              tf.flip[d] = af.flip[d];
            }
          }
          else
          {
            const int           fb = f - 2 * D1;
            const TreeFace<D2> &bf = B.neighbor(tb, fb);
            c.boundary_ids[t][f]   = B.boundary_id(tb, fb);
            if (bf.tree < 0)
              continue;
            tf.tree = pi.tree(ta, bf.tree);
            tf.face = 2 * D1 + bf.face;
            for (int d = 0; d < D2; ++d)
            {
              tf.perm[D1 + d] = D1 + bf.perm[d];
              tf.flip[D1 + d] = bf.flip[d];
            }
          }
        }
      }
    if (index)
      *index = pi;
    return c;
  }
} // namespace hyperforest

#endif
