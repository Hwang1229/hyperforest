# hyperforest

A header-only, p4est-like **parallel forest of 2^D-trees for any dimension D**, built for
phase-space problems (kinetic equations, radiative transfer) whose meshes live in more than
three dimensions, with an adapter that lets **deal.II** programs use it directly.

deal.II's `Triangulation`, `DoFHandler` and `FEValues` exist for dim ≤ 3 only. For a phase space
X × S (dim X = dx ≤ 3, dim S = ds ≤ 3, D = dx + ds), hyperforest carries the D-dimensional mesh,
and deal.II supplies everything that lives on the factors: the coarse spatial mesh (any
quadrilateral / hexahedral mesh, also from gmsh), `FE_DGQ`-compatible bases, quadrature, and
`LinearAlgebra::distributed::Vector` with its MPI partitioner.

Tested for D = 2, …, 6 on 1–8 MPI ranks.

## Features

| p4est | hyperforest | |
|---|---|---|
| `p4est_quadrant_t` | `Cell<D>` | integer anchor, level, tree; Morton order by "most significant differing bit" comparison (no D·L-bit keys, 30 levels for every D) |
| `p4est_connectivity_t` | `Connectivity<D>` | face connections with signed-permutation transforms; `brick(n, periodic)`, `product(A, B)` (e.g. space × angle), `connectivity_from_coarse_mesh(tria)` (deal.II meshes, face orientation read from the vertices) |
| `p4est_new` | `Forest<D>(comm, conn, level)` | uniform forest, evenly partitioned |
| `p4est_refine / coarsen` | `adapt(flags)` | refine, or coarsen complete local families |
| `p4est_balance(FACE)` | `balance()` | 2:1 balance across all 2D faces (incremental ripple) |
| `p4est_partition` | `partition(weights)` | optionally weighted; `transfer_fixed`, `transfer_variable` move cell data |
| `p4est_ghost_new(FACE)` | `build_ghost_layer()`, `exchange_ghost_data()` | face ghosts; `locate_points()` adds the remote leaves of arbitrary phase-space points |
| `p4est_iterate` (faces) | `face_neighbors(i, f)` | the leaves across a face (same / coarser / all finer), each with the reference-coordinate map and the shared face piece |
| `p4est_checksum` | `checksum()` | partition independent |
| — | `relation(old, new)` | how leaves before and after adapt / balance relate (for data transfer) |

deal.II adapter (`hyperforest/dealii.h`):

- `PhaseSpace<dx, ds>`: coarse spatial mesh × brick of angular / velocity boxes, the product
  connectivity, the geometry of both factors of a cell;
- `CoarseGeometry<dim>`: the d-linear map of every tree;
- `make_partitioner(forest, n)` (n dofs per leaf) and `make_partitioner(forest, n_dofs, layout)`
  (variable dofs per leaf, hp) for `LinearAlgebra::distributed::Vector`;
- `TensorDG<dx, ds>(px, ps)`: the tensor-product basis Q_px(X) × Q_ps(S) (Gauss–Lobatto Lagrange,
  identical to `FE_DGQ`), solution transfer under refinement / coarsening (`transfer`,
  `transfer_hp`).

## Build and test

Requirements: C++14, MPI; deal.II ≥ 9.5 for the adapter, its tests and the example.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # -DDEAL_II_DIR=... if needed
cmake --build build && (cd build && ctest)
```

| test | checks |
|---|---|
| `forest_np{1,3,4}` | D = 2…5, periodic directions: random refinement, coarsening, balance, weighted partition + data transfer, ghost layer, face neighbours (count, coverage of the face, the same face point from both sides), all against brute-force geometric adjacency |
| `dealii_np{1,4}` | tree connectivity of deal.II meshes with rotated / reflected neighbours (hyper_ball, hyper_shell, an unstructured gmsh mesh): physical face points agree to 1e-16 from both sides after refinement; phase space; ghost values through distributed vectors; basis = `FE_DGQ`; transfer |
| `free_streaming_linear_*` | the example below: a solution in the discrete space is reproduced to round-off on random nonconforming meshes in 2D, 4D and 6D phase space, also across rotated trees |
| `free_streaming_order_p{1,2}` | optimal L2 order p + 1 under uniform refinement |

## Using it from a CMake project

```cmake
add_subdirectory(external/hyperforest)          # git submodule, or
find_package(hyperforest REQUIRED)              # after `cmake --install`
target_link_libraries(my_solver hyperforest::hyperforest)
```

```cpp
#include <hyperforest/dealii.h>
namespace hf = hyperforest;

Triangulation<2> space;                         // any quad mesh (deal.II, gmsh)
GridGenerator::hyper_ball(space);
// phase space = space x (phi, mu) box, periodic in phi; trees numbered angle-major
hf::PhaseSpace<2, 2> ps(space, {4, 1}, {-M_PI, 0.0}, {M_PI, 1.0}, {true, false});
hf::Forest<4> forest(MPI_COMM_WORLD, ps.connectivity, /*level=*/2);

forest.adapt(flags);  forest.balance();  forest.partition();
forest.build_ghost_layer();

hf::TensorDG<2, 2> fe(1, 1);                    // Q1(x) x Q1(s), FE_DGQ numbering
auto partitioner = hf::make_partitioner(forest, fe.n);
LinearAlgebra::distributed::Vector<double> u(partitioner);   // ghost dofs after the owned ones

std::vector<hf::FaceNeighbor<4>> nbs;
for (int i = 0; i < forest.n_local(); ++i)
  for (int f = 0; f < 2 * 4; ++f) {
    forest.face_neighbors(i, f, nbs);           // empty: domain boundary
    for (auto &nb : nbs) {                      // integrate over the piece [nb.lo, nb.hi]
      auto xi_nb = nb.map(xi);                  // reference point of the neighbour
    }
  }
```

## Example: free streaming in 2D / 4D / 6D phase space

`examples/free_streaming.cc` solves v · ∇ₓ f + σ f = q in X × V, V = [−1, 1]^d, with upwind DG
(Q_p × Q_p, strong form, element-by-element assembly), GMRES with block Jacobi, MPI, and
random / uniform / adaptive refinement with solution transfer and repartitioning. It is a compact
tour of the library (about 750 lines with comments).

```bash
mpirun -np 4 ./free_streaming dim=2 test=smooth refinement=uniform cycles=4   # 4D, convergence
mpirun -np 4 ./free_streaming dim=3 test=linear refinement=random cycles=2    # 6D, exactness
```

| case | result |
|---|---|
| linear solution, random refinement + coarsening, 2D / 4D (also hyper_ball trees, p = 2) / 6D | nodal error 1e-13 … 1e-16 (6D, 363k dofs: 5e-11) |
| smooth solution, uniform refinement, 4D, p = 1 | L2 error 3.4e-2 → 8.6e-3 → 2.2e-3 (4.2M dofs), order 1.98 |
| the same, p = 2 | 3.1e-3 → 3.9e-4 (1.3M dofs), order 2.98 |

A full application — a 4D / 5D phase-space DG solver for the radiative transfer equation with
parallel transport sweeps, scattering, graded refractive index, Fresnel / diffuse / BRDF walls and
hp adaptivity — is built on hyperforest in the FlexRT project.

## Roadmap

- adapters for further FEM libraries: MFEM, DOLFINx, Gridap.jl (through a C API for fixed D),
  PETSc DMForest;
- partition for coarsening (families split across ranks are not coarsened yet);
- edge / corner connectivity and balance (currently faces only, which is what DG needs).

## License

To be decided.
