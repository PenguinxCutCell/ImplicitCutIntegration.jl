module ImplicitCutIntegration

using ImplicitIntegration
using SparseArrays: spdiagm
using StaticArrays: SVector

export GeometricMoments

const DEFAULT_TOL = 1.0e-6

"""
    GeometricMoments(body::Function, mesh::NTuple{N,<:AbstractVector}; compute_centroids::Bool = true, tol::Real = DEFAULT_TOL) where {N}

Compute geometric moments (volumes, centroids, face capacities) on a tensor grid defined by
`mesh`, which stores the coordinates along each axis. Integration is performed with
`ImplicitIntegration.jl`, so `body` only needs to define the level-set function.

Returns `(A, B, V, W, C_ω, C_γ, Γ, cell_types)` where each component matches the capacity
notation from the cut-cell method literature.
"""
function GeometricMoments(body::Function, mesh::NTuple{N,<:AbstractVector}; compute_centroids::Bool = true, tol::Real = DEFAULT_TOL) where {N}
    @assert N > 0 "Mesh must contain at least one dimension"
    coords = mesh
    @assert all(length(coords[d]) >= 2 for d in 1:N) "Each mesh vector needs at least two nodes"

    dims = ntuple(d -> length(coords[d]) - 1, N)
    dims_tuple = Tuple(dims)
    dims_extended = ntuple(d -> dims[d] + 1, N)
    dims_extended_tuple = Tuple(dims_extended)

    Φ = level_set_wrapper(body, N)
    tol_float = Float64(tol)

    total_cells = prod(dims_tuple)
    V_dense = zeros(Float64, dims_tuple...)
    Γ_dense = zeros(Float64, dims_tuple...)
    cell_types_array = zeros(Int, dims_tuple...)
    centroid_coords = zeros(Float64, N)

    centroid_map = LinearIndices(dims_tuple)
    C_ω = Vector{SVector{N,Float64}}(undef, total_cells)
    zero_centroid = zero_svector(N)
    for idx in 1:total_cells
        C_ω[idx] = zero_centroid
    end

    A_dense = ntuple(_ -> zeros(Float64, dims_extended_tuple...), N)
    B_dense = ntuple(_ -> zeros(Float64, dims_extended_tuple...), N)
    W_dense = ntuple(_ -> zeros(Float64, dims_extended_tuple...), N)

    face_funcs = Dict{Tuple{Int,Float64}, Function}()

    # Pass 1: volumes, centroids, cell classification, interface measures
    for I in CartesianIndices(dims_tuple)
        linear_idx = centroid_map[I]
        a = cell_lower_bounds(coords, I)
        b = cell_upper_bounds(coords, I)
        cell_measure = prod(b[d] - a[d] for d in 1:N)

        vol = ImplicitIntegration.integrate(_ -> 1.0, Φ, a, b; tol = tol_float).val
        V_dense[I] = vol

        fill_cell_centroid!(centroid_coords, a, b)
        cell_type = classify_cell(vol, cell_measure, tol_float)

        if cell_type == -1
            update_cut_cell_centroid!(centroid_coords, Φ, a, b, vol, tol_float)
            Γ_dense[I] = ImplicitIntegration.integrate(_ -> 1.0, Φ, a, b; surface = true, tol = tol_float).val
        end

        cell_types_array[I] = cell_type
        C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)

        # Cache the face-aligned level set functions needed for face capacities
        for face_dim in 1:N
            face_coord = Float64(coords[face_dim][I[face_dim]])
            key = (face_dim, face_coord)
            if !haskey(face_funcs, key)
                face_funcs[key] = create_fixed_coordinate_function(body, face_dim, face_coord, N)
            end
        end
    end

    # Optional Pass 1b: interface centroids
    C_γ = compute_centroids ? compute_interface_centroids!(centroid_map, coords, cell_types_array, Γ_dense, Φ, tol_float) : SVector{N,Float64}[]

    # Pass 2: face (A) and center-line (B) capacities
    for I in CartesianIndices(dims_tuple)
        for face_dim in 1:N
            face_coord = Float64(coords[face_dim][I[face_dim]])
            Φ_face = face_funcs[(face_dim, face_coord)]

            if N == 1
                A_dense[face_dim][I] = Φ_face() <= 0.0 ? 1.0 : 0.0
            else
                a_reduced = lower_bounds(coords, I, face_dim)
                b_reduced = upper_bounds(coords, I, face_dim)
                A_dense[face_dim][I] = ImplicitIntegration.integrate(_ -> 1.0, Φ_face, a_reduced, b_reduced; tol = tol_float).val
            end
        end

        centroid = C_ω[centroid_map[I]]
        for dim in 1:N
            Φ_center = create_fixed_coordinate_function(body, dim, centroid[dim], N)
            if N == 1
                B_dense[dim][I] = Φ_center() <= 0.0 ? 1.0 : 0.0
            else
                a_reduced = lower_bounds(coords, I, dim)
                b_reduced = upper_bounds(coords, I, dim)
                B_dense[dim][I] = ImplicitIntegration.integrate(_ -> 1.0, Φ_center, a_reduced, b_reduced; tol = tol_float).val
            end
        end
    end

    # Pass 3: staggered volumes (W)
    for stagger_dim in 1:N
        for I in CartesianIndices(dims_extended_tuple)
            if !valid_stagger_index(I, dims, stagger_dim)
                continue
            end

            prev_idx = max(I[stagger_dim] - 1, 1)
            next_idx = min(I[stagger_dim], dims[stagger_dim])

            prev_I = CartesianIndex(ntuple(d -> d == stagger_dim ? prev_idx : I[d], N))
            next_I = CartesianIndex(ntuple(d -> d == stagger_dim ? next_idx : I[d], N))

            prev_centroid = C_ω[centroid_map[prev_I]]
            next_centroid = C_ω[centroid_map[next_I]]

            a = ntuple(d -> d == stagger_dim ? prev_centroid[d] : Float64(coords[d][I[d]]), N)
            b = ntuple(d -> d == stagger_dim ? next_centroid[d] : Float64(coords[d][I[d] + 1]), N)

            prev_type = cell_types_array[prev_I]
            next_type = cell_types_array[next_I]

            if prev_type != next_type || prev_type == -1 || next_type == -1
                W_dense[stagger_dim][I] = ImplicitIntegration.integrate(_ -> 1.0, Φ, a, b; tol = tol_float).val
            else
                if prev_type == 1
                    W_dense[stagger_dim][I] = prod(stagger_volume_extent(d, stagger_dim, prev_centroid, next_centroid, coords, I) for d in 1:N)
                else
                    W_dense[stagger_dim][I] = 0.0
                end
            end
        end
    end

    A = ntuple(i -> spdiagm(0 => vec(A_dense[i])), N)
    B = ntuple(i -> spdiagm(0 => vec(B_dense[i])), N)
    V = spdiagm(0 => vec(V_dense))
    W = ntuple(i -> spdiagm(0 => vec(W_dense[i])), N)
    Γ = spdiagm(0 => vec(Γ_dense))
    cell_types = vec(cell_types_array)

    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

# --- helpers ---------------------------------------------------------------

@inline zero_svector(N) = SVector{N,Float64}(ntuple(_ -> 0.0, N))

@inline function level_set_wrapper(body::Function, N::Int)
    return r -> body(ntuple(i -> r[i], N)...)
end

@inline function cell_lower_bounds(coords::NTuple{N,AbstractVector}, I::CartesianIndex{N}) where {N}
    ntuple(d -> Float64(coords[d][I[d]]), N)
end

@inline function cell_upper_bounds(coords::NTuple{N,AbstractVector}, I::CartesianIndex{N}) where {N}
    ntuple(d -> Float64(coords[d][I[d] + 1]), N)
end

@inline function lower_bounds(coords::NTuple{N,AbstractVector}, I::CartesianIndex{N}, skip_dim::Int) where {N}
    ntuple(k -> begin
        dim = k < skip_dim ? k : k + 1
        Float64(coords[dim][I[dim]])
    end, max(N - 1, 0))
end

@inline function upper_bounds(coords::NTuple{N,AbstractVector}, I::CartesianIndex{N}, skip_dim::Int) where {N}
    ntuple(k -> begin
        dim = k < skip_dim ? k : k + 1
        Float64(coords[dim][I[dim] + 1])
    end, max(N - 1, 0))
end

@inline function fill_cell_centroid!(centroid_coords::Vector{Float64}, a::NTuple{N,Float64}, b::NTuple{N,Float64}) where {N}
    for d in 1:N
        centroid_coords[d] = 0.5 * (a[d] + b[d])
    end
    return centroid_coords
end

@inline function classify_cell(vol::Float64, cell_measure::Float64, tol::Float64)
    full_tol = tol * max(1.0, cell_measure)
    if abs(vol) <= tol
        return 0
    elseif abs(vol - cell_measure) <= full_tol
        return 1
    else
        return -1
    end
end

function update_cut_cell_centroid!(centroid_coords, Φ, a, b, vol, tol)
    if abs(vol) <= tol
        return centroid_coords
    end
    N = length(centroid_coords)
    for d in 1:N
        coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; tol = tol).val
        value = coord_integral / vol
        centroid_coords[d] = isfinite(value) ? value : 0.5 * (a[d] + b[d])
    end
    return centroid_coords
end

function compute_interface_centroids!(centroid_map, coords, cell_types_array, Γ_dense, Φ, tol)
    dims_tuple = size(cell_types_array)
    N = length(coords)
    result = Vector{SVector{N,Float64}}(undef, length(cell_types_array))
    zero_centroid = zero_svector(N)
    for idx in eachindex(result)
        result[idx] = zero_centroid
    end

    centroid_coords = zeros(Float64, N)
    for I in CartesianIndices(dims_tuple)
        if cell_types_array[I] == -1 && Γ_dense[I] > tol
            linear_idx = centroid_map[I]
            a = cell_lower_bounds(coords, I)
            b = cell_upper_bounds(coords, I)
            interface_measure = Γ_dense[I]
            for d in 1:N
                coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; surface = true, tol = tol).val
                value = coord_integral / interface_measure
                centroid_coords[d] = isfinite(value) ? value : 0.5 * (a[d] + b[d])
            end
            result[linear_idx] = SVector{N,Float64}(centroid_coords)
        end
    end
    return result
end

@inline function valid_stagger_index(I::CartesianIndex{N}, dims::NTuple{N,Int}, stagger_dim::Int) where {N}
    for d in 1:N
        upper = d == stagger_dim ? dims[d] + 1 : dims[d]
        if I[d] < 1 || I[d] > upper
            return false
        end
    end
    return true
end

@inline function stagger_volume_extent(d, stagger_dim, prev_centroid, next_centroid, coords, I)
    if d == stagger_dim
        return next_centroid[d] - prev_centroid[d]
    else
        return Float64(coords[d][I[d] + 1] - coords[d][I[d]])
    end
end

function create_fixed_coordinate_function(body, fixed_dim, fixed_value, N)
    if N == 1
        return () -> body(fixed_value)
    else
        return y -> begin
            args = ntuple(i -> i == fixed_dim ? fixed_value : y[i - (i > fixed_dim ? 1 : 0)], N)
            body(args...)
        end
    end
end

end # module
