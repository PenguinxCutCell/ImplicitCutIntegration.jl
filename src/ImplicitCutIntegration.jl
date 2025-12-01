module ImplicitCutIntegration

using ImplicitIntegration
using SparseArrays: spdiagm
using StaticArrays: SVector

export GeometricMoments, GeometricMomentsThreaded

const DEFAULT_TOL = 1.0e-6
# Implicit Integration Direct implementation
"""
    GeometricMoments(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute geometric moments (volumes, centroids, face capacities) using direct integration.
This method uses ImplicitIntegration to compute various capacity quantities for a level set function.

# Arguments
- `body::Function`: The level set function defining the domain
- `mesh::AbstractMesh`: The mesh on which to compute geometric quantities
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components (A, B, V, W, C_ω, C_γ, Γ, cell_types)
"""
function GeometricMoments(body::Function, mesh::NTuple{N,AbstractVector}; compute_centroids::Bool = true, tol=DEFAULT_TOL) where {N}
    # Extract mesh dimensions
    dims = ntuple(i -> length(mesh[i])-1, N)
    coords = mesh    
    # Create dimension-appropriate level set function wrapper once
    Φ = if N == 1
        (r) -> body(r[1])
    elseif N == 2
        (r) -> body(r[1], r[2])
    elseif N == 3
        (r) -> body(r[1], r[2], r[3])
    elseif N == 4
        (r) -> body(r[1], r[2], r[3], r[4])
    else
        error("Unsupported dimension: $N")
    end
    
    # Get cell size for reference
    cell_sizes = ntuple(i -> (coords[i][2] - coords[i][1]), N)
    full_volume = prod(cell_sizes)
    
    # Pre-allocate result arrays with extended dimensions
    dims_extended = ntuple(i -> dims[i] + 1, N)
    total_cells_extended = prod(dims_extended)
    
    # Initialize all matrices with extended dimensions
    V_dense = zeros(dims_extended)
    cell_types_array = zeros(Int, dims_extended)
    C_ω = Vector{SVector{N,Float64}}(undef, total_cells_extended)
    for i in 1:total_cells_extended
        C_ω[i] = SVector{N,Float64}(zeros(N))
    end
    
    Γ_dense = zeros(dims_extended)
    
    # Initialize A, B, W with consistent extended dimensions
    A_dense = ntuple(_ -> zeros(dims_extended), N)
    B_dense = ntuple(_ -> zeros(dims_extended), N)
    W_dense = ntuple(_ -> zeros(dims_extended), N)
    
    # Pre-allocate arrays for repeated calculations
    centroid_coords = zeros(N)
    
    
    # Pre-create all fixed-coordinate level set functions
    # Only needed for A calculation
    face_funcs = Dict{Tuple{Int, Float64}, Function}()
    
    # First pass: Compute volumes, centroids, cell types, interface measures
    for I in CartesianIndices(dims)
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Get cell bounds (only compute once per cell)
        a = ntuple(d -> coords[d][I[d]], N)
        b = ntuple(d -> coords[d][I[d] + 1], N)
        
        # Compute volume fraction
        vol = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
        V_dense[I] = vol
        
        # Classify cell
        if vol < 1e-14
            cell_types_array[I] = 0  # Solid
            # Use geometric center for solid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        elseif abs(vol - full_volume) < 1e-14
            cell_types_array[I] = 1  # Fluid
            # Use geometric center for fluid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        else
            cell_types_array[I] = -1  # Cut
            
            # Only compute detailed centroid for cut cells
            for d in 1:N
                coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; tol=tol).val
                centroid_coords[d] = isnan(coord_integral/vol) ? 0.5*(a[d] + b[d]) : coord_integral/vol
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
            
            # Only compute interface measure for cut cells
            Γ_dense[I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; surface=true, tol=tol).val
        end
        
        # Cache face functions for each face coordinate (for A calculation)
        for face_dim in 1:N
            face_coord = coords[face_dim][I[face_dim]]
            key = (face_dim, face_coord)
            if !haskey(face_funcs, key)
                face_funcs[key] = create_fixed_coordinate_function(body, face_dim, face_coord, N)
            end
        end
    end
    
    # Compute interface centroids if requested (only for cut cells)
    C_γ = if compute_centroids
        interface_centroids = Vector{SVector{N,Float64}}(undef, total_cells_extended)
        
        # Initialize all to zero
        for i in 1:total_cells_extended
            interface_centroids[i] = SVector{N,Float64}(zeros(N))
        end
        
        # Only process cut cells with non-zero interface measure
        for I in CartesianIndices(dims)
            if cell_types_array[I] == -1 && Γ_dense[I] > 1e-14
                linear_idx = LinearIndices(dims_extended)[I]
                
                a = ntuple(d -> coords[d][I[d]], N)
                b = ntuple(d -> coords[d][I[d] + 1], N)
                
                interface_measure = Γ_dense[I]
                
                # Compute interface centroid for cut cells
                for d in 1:N
                    coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; surface=true, tol=tol).val
                    centroid_coords[d] = isnan(coord_integral/interface_measure) ? 
                                        0.5*(a[d] + b[d]) : coord_integral/interface_measure
                end
                interface_centroids[linear_idx] = SVector{N,Float64}(centroid_coords)
            end
        end
        
        interface_centroids
    else
        Vector{SVector{N,Float64}}(undef, 0)
    end
    
    # Second pass: Compute A, B
    for I in CartesianIndices(dims)
        # Only process cells that exist in the mesh (efficiency)
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Compute A (face capacities)
        for face_dim in 1:N
            face_coord = coords[face_dim][I[face_dim]]
            key = (face_dim, face_coord)
            Φ_face = face_funcs[key]
            
            # Get integration bounds for remaining dimensions
            a_reduced = [coords[d][I[d]] for d in 1:N if d != face_dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != face_dim]
            
            # Integrate to get face capacity
            if N == 1
                A_dense[face_dim][I] = Φ_face() ≤ 0.0 ? 1.0 : 0.0
            else
                A_dense[face_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_face, 
                                                   tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
        
        # Compute B (center line capacities)
        centroid = C_ω[linear_idx]
        
        for dim in 1:N
            # Create centroid-fixed function
            Φ_center = create_fixed_coordinate_function(body, dim, centroid[dim], N)
            
            # Get integration bounds
            a_reduced = [coords[d][I[d]] for d in 1:N if d != dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != dim]
            
            # Integrate to get center line capacity
            if N == 1
                B_dense[dim][I] = Φ_center() ≤ 0.0 ? 1.0 : 0.0
            else
                B_dense[dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_center, 
                                                 tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
    end
    
    # Third pass: Compute W (staggered volumes)
    # This is separate because it uses different indices
    for stagger_dim in 1:N
        for I in CartesianIndices(dims_extended)
            # Skip boundary cells to avoid out of bounds
            if all(1 <= I[d] <= (d == stagger_dim ? dims[d]+1 : dims[d]) for d in 1:N)
                # Find neighboring cells
                prev_idx = max(I[stagger_dim] - 1, 1)
                next_idx = min(I[stagger_dim], dims[stagger_dim])
                
                # Get cell indices
                prev_I = CartesianIndex(ntuple(d -> d == stagger_dim ? prev_idx : I[d], N))
                next_I = CartesianIndex(ntuple(d -> d == stagger_dim ? next_idx : I[d], N))
                
                # Get centroids
                prev_centroid = C_ω[LinearIndices(dims_extended)[prev_I]]
                next_centroid = C_ω[LinearIndices(dims_extended)[next_I]]
                
                # Build integration domain
                a = ntuple(d -> d == stagger_dim ? prev_centroid[d] : coords[d][I[d]], N)
                b = ntuple(d -> d == stagger_dim ? next_centroid[d] : coords[d][I[d] + 1], N)
                
                # Compute staggered volume - only if cells differ in type
                prev_type = cell_types_array[prev_I]
                next_type = cell_types_array[next_I] 
                
                if prev_type != next_type || prev_type == -1 || next_type == -1
                    W_dense[stagger_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
                else
                    # For consistent cell types (both fluid or both solid), we know the result
                    W_dense[stagger_dim][I] = (prev_type == 1) ? prod(d -> d == stagger_dim ? 
                                              (next_centroid[d] - prev_centroid[d]) : 
                                              (coords[d][I[d]+1] - coords[d][I[d]]), 1:N) : 0.0
                end
            end
        end
    end
    
    # Convert arrays to format required by Capacity struct (all at once)
    V = spdiagm(0 => reshape(V_dense, :))
    Γ = spdiagm(0 => reshape(Γ_dense, :))
    A = ntuple(i -> spdiagm(0 => reshape(A_dense[i], :)), N)
    B = ntuple(i -> spdiagm(0 => reshape(B_dense[i], :)), N)
    W = ntuple(i -> spdiagm(0 => reshape(W_dense[i], :)), N)
    cell_types = reshape(cell_types_array, :)
    
    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

"""
    create_fixed_coordinate_function(body, fixed_dim, fixed_value, N)

Create a level set function with one coordinate fixed at a specific value.
This generalizes both face functions and centroid-fixed functions.

# Arguments
- `body`: The original level set function
- `fixed_dim`: The dimension to fix (1 for x, 2 for y, etc.)
- `fixed_value`: The value to fix the coordinate at
- `N`: Total number of dimensions

# Returns
- A new function with the specified dimension fixed
"""
function create_fixed_coordinate_function(body, fixed_dim, fixed_value, N)
    # Special case for 1D - no input needed
    if N == 1
        return () -> body(fixed_value)
    else
        # For higher dimensions, create a function that inserts the fixed value
        # at the correct position and calls the body function
        return y -> body(ntuple(i -> i == fixed_dim ? fixed_value : y[i - (i > fixed_dim ? 1 : 0)], N)...)
    end
end 

"""
    GeometricMomentsThreaded(body::Function, mesh::NTuple{N,AbstractVector}; compute_centroids::Bool = true)

Multithreaded version of GeometricMoments. Compute geometric moments (volumes, centroids, 
face capacities) using direct integration with parallel processing.

This method uses ImplicitIntegration to compute various capacity quantities for a level set function.
Uses Julia's multithreading to parallelize computations across cells.

# Arguments
- `body::Function`: The level set function defining the domain
- `mesh::NTuple{N,AbstractVector}`: The mesh on which to compute geometric quantities
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components (A, B, V, W, C_ω, C_γ, Γ, cell_types)
"""
function GeometricMomentsThreaded(body::Function, mesh::NTuple{N,AbstractVector}; compute_centroids::Bool = true, tol=DEFAULT_TOL) where {N}
    # Extract mesh dimensions
    dims = ntuple(i -> length(mesh[i])-1, N)
    coords = mesh    
    # Create dimension-appropriate level set function wrapper once
    Φ = if N == 1
        (r) -> body(r[1])
    elseif N == 2
        (r) -> body(r[1], r[2])
    elseif N == 3
        (r) -> body(r[1], r[2], r[3])
    elseif N == 4
        (r) -> body(r[1], r[2], r[3], r[4])
    else
        error("Unsupported dimension: $N")
    end
    
    # Get cell size for reference
    cell_sizes = ntuple(i -> (coords[i][2] - coords[i][1]), N)
    full_volume = prod(cell_sizes)
    
    # Pre-allocate result arrays with extended dimensions
    dims_extended = ntuple(i -> dims[i] + 1, N)
    total_cells_extended = prod(dims_extended)
    
    # Initialize all matrices with extended dimensions
    V_dense = zeros(dims_extended)
    cell_types_array = zeros(Int, dims_extended)
    C_ω = Vector{SVector{N,Float64}}(undef, total_cells_extended)
    for i in 1:total_cells_extended
        C_ω[i] = SVector{N,Float64}(zeros(N))
    end
    
    Γ_dense = zeros(dims_extended)
    
    # Initialize A, B, W with consistent extended dimensions
    A_dense = ntuple(_ -> zeros(dims_extended), N)
    B_dense = ntuple(_ -> zeros(dims_extended), N)
    W_dense = ntuple(_ -> zeros(dims_extended), N)
    
    # Convert CartesianIndices to a vector for threaded iteration
    indices = collect(CartesianIndices(dims))
    
    # First pass: Compute volumes, centroids, cell types, interface measures (threaded)
    Threads.@threads for I in indices
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Get cell bounds (only compute once per cell)
        a = ntuple(d -> coords[d][I[d]], N)
        b = ntuple(d -> coords[d][I[d] + 1], N)
        
        # Compute volume fraction
        vol = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
        V_dense[I] = vol
        
        # Pre-allocate local centroid_coords for thread safety
        centroid_coords = zeros(N)
        
        # Classify cell
        if vol < 1e-14
            cell_types_array[I] = 0  # Solid
            # Use geometric center for solid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        elseif abs(vol - full_volume) < 1e-14
            cell_types_array[I] = 1  # Fluid
            # Use geometric center for fluid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        else
            cell_types_array[I] = -1  # Cut
            
            # Only compute detailed centroid for cut cells
            for d in 1:N
                coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; tol=tol).val
                centroid_coords[d] = isnan(coord_integral/vol) ? 0.5*(a[d] + b[d]) : coord_integral/vol
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
            
            # Only compute interface measure for cut cells
            Γ_dense[I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; surface=true, tol=tol).val
        end
    end
    
    # Compute interface centroids if requested (only for cut cells) - threaded
    C_γ = if compute_centroids
        interface_centroids = Vector{SVector{N,Float64}}(undef, total_cells_extended)
        
        # Initialize all to zero
        for i in 1:total_cells_extended
            interface_centroids[i] = SVector{N,Float64}(zeros(N))
        end
        
        # Only process cut cells with non-zero interface measure - threaded
        Threads.@threads for I in indices
            if cell_types_array[I] == -1 && Γ_dense[I] > 1e-14
                linear_idx = LinearIndices(dims_extended)[I]
                
                a = ntuple(d -> coords[d][I[d]], N)
                b = ntuple(d -> coords[d][I[d] + 1], N)
                
                interface_measure = Γ_dense[I]
                
                # Pre-allocate local centroid_coords for thread safety
                centroid_coords = zeros(N)
                
                # Compute interface centroid for cut cells
                for d in 1:N
                    coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; surface=true, tol=tol).val
                    centroid_coords[d] = isnan(coord_integral/interface_measure) ? 
                                        0.5*(a[d] + b[d]) : coord_integral/interface_measure
                end
                interface_centroids[linear_idx] = SVector{N,Float64}(centroid_coords)
            end
        end
        
        interface_centroids
    else
        Vector{SVector{N,Float64}}(undef, 0)
    end
    
    # Second pass: Compute A, B - threaded
    Threads.@threads for I in indices
        # Only process cells that exist in the mesh (efficiency)
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Compute A (face capacities)
        for face_dim in 1:N
            face_coord = coords[face_dim][I[face_dim]]
            Φ_face = create_fixed_coordinate_function(body, face_dim, face_coord, N)
            
            # Get integration bounds for remaining dimensions
            a_reduced = [coords[d][I[d]] for d in 1:N if d != face_dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != face_dim]
            
            # Integrate to get face capacity
            if N == 1
                A_dense[face_dim][I] = Φ_face() ≤ 0.0 ? 1.0 : 0.0
            else
                A_dense[face_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_face, 
                                                   tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
        
        # Compute B (center line capacities)
        centroid = C_ω[linear_idx]
        
        for dim in 1:N
            # Create centroid-fixed function
            Φ_center = create_fixed_coordinate_function(body, dim, centroid[dim], N)
            
            # Get integration bounds
            a_reduced = [coords[d][I[d]] for d in 1:N if d != dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != dim]
            
            # Integrate to get center line capacity
            if N == 1
                B_dense[dim][I] = Φ_center() ≤ 0.0 ? 1.0 : 0.0
            else
                B_dense[dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_center, 
                                                 tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
    end
    
    # Third pass: Compute W (staggered volumes) - threaded per stagger dimension
    for stagger_dim in 1:N
        stagger_indices = collect(CartesianIndices(dims_extended))
        Threads.@threads for I in stagger_indices
            # Skip boundary cells to avoid out of bounds
            if all(1 <= I[d] <= (d == stagger_dim ? dims[d]+1 : dims[d]) for d in 1:N)
                # Find neighboring cells
                prev_idx = max(I[stagger_dim] - 1, 1)
                next_idx = min(I[stagger_dim], dims[stagger_dim])
                
                # Get cell indices
                prev_I = CartesianIndex(ntuple(d -> d == stagger_dim ? prev_idx : I[d], N))
                next_I = CartesianIndex(ntuple(d -> d == stagger_dim ? next_idx : I[d], N))
                
                # Get centroids
                prev_centroid = C_ω[LinearIndices(dims_extended)[prev_I]]
                next_centroid = C_ω[LinearIndices(dims_extended)[next_I]]
                
                # Build integration domain
                a = ntuple(d -> d == stagger_dim ? prev_centroid[d] : coords[d][I[d]], N)
                b = ntuple(d -> d == stagger_dim ? next_centroid[d] : coords[d][I[d] + 1], N)
                
                # Compute staggered volume - only if cells differ in type
                prev_type = cell_types_array[prev_I]
                next_type = cell_types_array[next_I] 
                
                if prev_type != next_type || prev_type == -1 || next_type == -1
                    W_dense[stagger_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
                else
                    # For consistent cell types (both fluid or both solid), we know the result
                    W_dense[stagger_dim][I] = (prev_type == 1) ? prod(d -> d == stagger_dim ? 
                                              (next_centroid[d] - prev_centroid[d]) : 
                                              (coords[d][I[d]+1] - coords[d][I[d]]), 1:N) : 0.0
                end
            end
        end
    end
    
    # Convert arrays to format required by Capacity struct (all at once)
    V = spdiagm(0 => reshape(V_dense, :))
    Γ = spdiagm(0 => reshape(Γ_dense, :))
    A = ntuple(i -> spdiagm(0 => reshape(A_dense[i], :)), N)
    B = ntuple(i -> spdiagm(0 => reshape(B_dense[i], :)), N)
    W = ntuple(i -> spdiagm(0 => reshape(W_dense[i], :)), N)
    cell_types = reshape(cell_types_array, :)
    
    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

end # module ImplicitCutIntegration