# ImplicitCutIntegration.jl

ImplicitCutIntegration.jl builds on [ImplicitIntegration.jl](https://github.com/maltezfaria/ImplicitIntegration.jl)
to compute geometric moments for level-set defined bodies on tensor-product meshes of arbitrary
dimension. The central entry point is `GeometricMoments`, which returns the sparse capacity
matrices (A, B, V, W), centroid locations, interface centroids, surface measures, and cell types
needed by finite-volume style cut-cell methods.

## Getting started

```julia
julia> using Pkg
julia> Pkg.activate("/path/to/ImplicitCutIntegration.jl")
julia> Pkg.instantiate()
```

`GeometricMoments` expects:

- `body::Function`: a level-set function that accepts `N` scalar arguments.
- `mesh::NTuple{N,<:AbstractVector}`: coordinate vectors for each axis. Each vector must be
	monotonically increasing and contain at least two nodes (cell edges).

The function returns `(A, B, V, W, C_ω, C_γ, Γ, cell_types)` where each entry is generic across the
dimension `N`. All dense data are converted to diagonal sparse matrices to remain lightweight for
post-processing.

## Example

```julia
using ImplicitCutIntegration

# Signed-distance of a unit circle centered at the origin
body(x, y) = sqrt(x^2 + y^2) - 1.0

# Mesh described as coordinate vectors along each axis
mesh = (
		range(-1.5, 1.5; length = 33) |> collect,
		range(-1.5, 1.5; length = 33) |> collect,
)

A, B, V, W, Cω, Cγ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids = true)

println("Number of cut cells: ", count(==( -1), cell_types))
```

The mesh input scales naturally to 1D, 2D, 3D, or 4D grids—no additional types are required. If you
only care about volume fractions and do not need interface centroids, pass `compute_centroids =
false` to skip that post-processing step.