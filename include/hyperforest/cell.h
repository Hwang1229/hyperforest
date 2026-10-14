/* ---------------------------------------------------------------------
 *
 * hyperforest: a p4est-like parallel forest of 2^D-trees for any D >= 1
 *
 * cell.h: the D-dimensional cell ("hyperoctant") of a tree. A cell is
 * stored by the integer coordinates of its lower corner (anchor), its
 * level and the tree it belongs to, exactly as a p4est quadrant. The root
 * of every tree is [0, root_len)^D; a cell of level l has edge length
 * 2^(max_level - l).
 *
 * Cells are ordered along the Morton (Z-order) curve, tree by tree. The
 * Morton index interleaves the coordinate bits level by level with the
 * bit of dimension D-1 most significant, so that the children of a cell
 * are ordered by their child id c = sum_d b_d 2^d (b_d = upper half in
 * dimension d). The comparison works directly on the coordinates (the
 * "most significant differing bit" trick), so no D * max_level bit key is
 * ever formed and every D has the full max_level levels.
 *
 * ---------------------------------------------------------------------
 */
#ifndef hyperforest_cell_h
#define hyperforest_cell_h

#include <array>
#include <cassert>
#include <cstdint>

namespace hyperforest
{
  // Finest level. Neighbours of cells just outside the root are still
  // representable: coordinates lie in [-root_len, 2 root_len) < 2^31.
  constexpr int     max_level = 30;
  constexpr int32_t root_len  = int32_t(1) << max_level;

  constexpr int32_t cell_length(const int level)
  {
    return int32_t(1) << (max_level - level);
  }

  // msb(a) < msb(b) for a, b >= 0
  inline bool less_msb(const uint32_t a, const uint32_t b)
  {
    return a < b && a < (a ^ b);
  }

  template <int D>
  struct Cell
  {
    static_assert(D >= 1 && D <= 10, "hyperforest supports 1 <= D <= 10");
    static constexpr int dimension       = D;
    static constexpr int n_children      = 1 << D;
    static constexpr int n_faces         = 2 * D;
    static constexpr int n_face_children = 1 << (D - 1);

    std::array<int32_t, D> x{}; // anchor (lower corner)
    int32_t                tree  = 0;
    int8_t                 level = 0;

    int32_t length() const
    {
      return cell_length(level);
    }

    Cell child(const int c) const
    {
      Cell          r = *this;
      const int32_t h = cell_length(level + 1);
      r.level         = level + 1;
      for (int d = 0; d < D; ++d)
        if ((c >> d) & 1)
          r.x[d] += h;
      return r;
    }

    Cell parent() const
    {
      assert(level > 0);
      Cell          r = *this;
      const int32_t h = length();
      r.level         = level - 1;
      for (int d = 0; d < D; ++d)
        r.x[d] &= ~h;
      return r;
    }

    int child_id() const
    {
      if (level == 0)
        return 0;
      const int32_t h = length();
      int           c = 0;
      for (int d = 0; d < D; ++d)
        if (x[d] & h)
          c |= 1 << d;
      return c;
    }

    Cell sibling(const int c) const
    {
      return parent().child(c);
    }

    // the same-size cell across face f = 2 d + side (normal d); it may lie
    // outside the root
    Cell face_neighbor(const int f) const
    {
      Cell r = *this;
      r.x[f / 2] += (f & 1) ? length() : -length();
      return r;
    }

    bool inside_root() const
    {
      for (int d = 0; d < D; ++d)
        if (x[d] < 0 || x[d] >= root_len)
          return false;
      return true;
    }

    // the descendants of level l (>= level) with the smallest / largest
    // Morton index; with l = max_level these are the first and last points
    Cell first_descendant(const int l = max_level) const
    {
      Cell r  = *this;
      r.level = l;
      return r;
    }
    Cell last_descendant(const int l = max_level) const
    {
      Cell          r = *this;
      const int32_t s = length() - cell_length(l);
      r.level         = l;
      for (int d = 0; d < D; ++d)
        r.x[d] += s;
      return r;
    }

    // b lies in this cell (b may be this cell itself)
    bool contains(const Cell &b) const
    {
      if (b.tree != tree || b.level < level)
        return false;
      const uint32_t mask = ~uint32_t(length() - 1);
      for (int d = 0; d < D; ++d)
        if ((uint32_t(b.x[d]) & mask) != uint32_t(x[d]))
          return false;
      return true;
    }

    // does this cell touch face f of the root?
    bool touches_root_face(const int f) const
    {
      return (f & 1) ? x[f / 2] + length() == root_len : x[f / 2] == 0;
    }

    bool operator==(const Cell &b) const
    {
      return tree == b.tree && level == b.level && x == b.x;
    }
    bool operator!=(const Cell &b) const
    {
      return !(*this == b);
    }
  };


  // Total order: by tree, then Morton index of the anchor, then level (an
  // ancestor comes before its descendants). Returns <0, 0, >0.
  template <int D>
  inline int compare(const Cell<D> &a, const Cell<D> &b)
  {
    if (a.tree != b.tree)
      return a.tree < b.tree ? -1 : 1;
    int      best = D - 1;
    uint32_t bx   = uint32_t(a.x[D - 1]) ^ uint32_t(b.x[D - 1]);
    for (int d = D - 2; d >= 0; --d)
    {
      const uint32_t xd = uint32_t(a.x[d]) ^ uint32_t(b.x[d]);
      if (less_msb(bx, xd))
      {
        best = d;
        bx   = xd;
      }
    }
    if (bx == 0)
      return int(a.level) - int(b.level);
    return a.x[best] < b.x[best] ? -1 : 1;
  }

  template <int D>
  inline bool operator<(const Cell<D> &a, const Cell<D> &b)
  {
    return compare(a, b) < 0;
  }


  // The i-th cell of level l of a tree in Morton order (D * l <= 63)
  template <int D>
  inline Cell<D> cell_from_morton(const int32_t tree, const int l, const uint64_t m)
  {
    assert(D * l <= 63);
    Cell<D> c;
    c.tree  = tree;
    c.level = l;
    for (int b = 0; b < l; ++b)
      for (int d = 0; d < D; ++d)
        if ((m >> (b * D + d)) & 1)
          c.x[d] |= int32_t(1) << (max_level - l + b);
    return c;
  }

  // Morton index of c among the cells of its level (D * level <= 63)
  template <int D>
  inline uint64_t morton_index(const Cell<D> &c)
  {
    assert(D * c.level <= 63);
    uint64_t m = 0;
    for (int b = 0; b < c.level; ++b)
      for (int d = 0; d < D; ++d)
        if ((c.x[d] >> (max_level - c.level + b)) & 1)
          m |= uint64_t(1) << (b * D + d);
    return m;
  }

  // The k-th (k < 2^(D-1)) child touching face f = 2 n + s: bit n of the
  // child id is s, the other bits are those of k
  template <int D>
  constexpr int face_child(const int f, const int k)
  {
    const int n = f / 2, s = f & 1;
    return ((k >> n) << (n + 1)) | (s << n) | (k & ((1 << n) - 1));
  }

  // The point (cell of max_level) of c touching face f with the other
  // coordinates at the anchor: a point of c on that face.
  template <int D>
  inline Cell<D> face_point(const Cell<D> &c, const int f)
  {
    Cell<D> p = c.first_descendant();
    if (f & 1)
      p.x[f / 2] += c.length() - 1;
    return p;
  }

  // Affine map between the reference coordinates [0,1]^D of two cells of
  // the same tree, a containing b (or b containing a):
  //   xi_b[d] = scale * xi_a[d] + shift[d]
  template <int D>
  struct NestedMap
  {
    double                scale;
    std::array<double, D> shift;
  };
  template <int D>
  inline NestedMap<D> nested_map(const Cell<D> &a, const Cell<D> &b)
  {
    NestedMap<D> m;
    const double ha = a.length(), hb = b.length();
    m.scale         = ha / hb;
    for (int d = 0; d < D; ++d)
      m.shift[d] = double(a.x[d] - b.x[d]) / hb;
    return m;
  }
} // namespace hyperforest

#endif
