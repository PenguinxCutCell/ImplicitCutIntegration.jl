using Test
using ImplicitCutIntegration
using LinearAlgebra: diag, norm
using Statistics: mean

const RADIUS = 0.45
const DOMAIN_EXTENT = 1.0
const RESOLUTION = Dict(1 => 8, 2 => 4)
const VOLUME_RTOL = Dict(1 => 1.0e-3, 2 => 2.0e-2)
const SURFACE_RTOL = Dict(1 => 1.0e-3, 2 => 3.0e-2)
const CENTROID_ATOL = Dict(1 => 0.01, 2 => 0.03)
const INTERFACE_FILTER_TOL = 1.0e-6

function hypersphere_volume(N, R)
    if N == 1
        return 2R
    elseif N == 2
        return pi * R^2
    elseif N == 3
        return (4.0 / 3.0) * pi * R^3
    elseif N == 4
        return (pi^2 / 2.0) * R^4
    else
        error("Unsupported dimension")
    end
end

function hypersphere_surface(N, R)
    if N == 1
        return 2.0
    elseif N == 2
        return 2pi * R
    elseif N == 3
        return 4pi * R^2
    elseif N == 4
        return 2pi^2 * R^3
    else
        error("Unsupported dimension")
    end
end

function build_mesh(N)
    cells = RESOLUTION[N]
    nodes = collect(range(-DOMAIN_EXTENT, DOMAIN_EXTENT; length = cells + 1))
    return ntuple(_ -> copy(nodes), N)
end

function centroid_norms(Cγ, Γ_diag)
    idxs = findall(x -> x > INTERFACE_FILTER_TOL, Γ_diag)
    return [norm(Cγ[i]) for i in idxs if i <= length(Cγ)]
end

@testset "Hypersphere moments" begin
    for N in 1:2
        mesh = build_mesh(N)
        body = (coords...) -> sum(y -> y^2, coords) - RADIUS^2
        A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids = true)

        V_diag = diag(V)
        Γ_diag = diag(Γ)
        V_sum = sum(V_diag)
        Γ_sum = sum(Γ_diag)

        @test isapprox(V_sum, hypersphere_volume(N, RADIUS); rtol = VOLUME_RTOL[N])
        @test isapprox(Γ_sum, hypersphere_surface(N, RADIUS); rtol = SURFACE_RTOL[N])

        norms = centroid_norms(C_γ, Γ_diag)
        @test !isempty(norms)
        @test isapprox(mean(norms), RADIUS; atol = CENTROID_ATOL[N], rtol = 0.2)
    end
end

@testset "Hypersphere moments (threaded)" begin
    for N in 1:2
        mesh = build_mesh(N)
        body = (coords...) -> sum(y -> y^2, coords) - RADIUS^2
        A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMomentsThreaded(body, mesh; compute_centroids = true)

        V_diag = diag(V)
        Γ_diag = diag(Γ)
        V_sum = sum(V_diag)
        Γ_sum = sum(Γ_diag)

        @test isapprox(V_sum, hypersphere_volume(N, RADIUS); rtol = VOLUME_RTOL[N])
        @test isapprox(Γ_sum, hypersphere_surface(N, RADIUS); rtol = SURFACE_RTOL[N])

        norms = centroid_norms(C_γ, Γ_diag)
        @test !isempty(norms)
        @test isapprox(mean(norms), RADIUS; atol = CENTROID_ATOL[N], rtol = 0.2)
    end
end

@testset "Single-threaded vs Threaded comparison" begin
    for N in 1:2
        mesh = build_mesh(N)
        body = (coords...) -> sum(y -> y^2, coords) - RADIUS^2
        
        # Run both versions
        A1, B1, V1, W1, C_ω1, C_γ1, Γ1, cell_types1 = GeometricMoments(body, mesh; compute_centroids = true)
        A2, B2, V2, W2, C_ω2, C_γ2, Γ2, cell_types2 = GeometricMomentsThreaded(body, mesh; compute_centroids = true)
        
        # Compare volume matrices
        @test diag(V1) ≈ diag(V2)
        
        # Compare interface measures
        @test diag(Γ1) ≈ diag(Γ2)
        
        # Compare cell types
        @test cell_types1 == cell_types2
        
        # Compare face capacities (A)
        for i in 1:N
            @test diag(A1[i]) ≈ diag(A2[i])
        end
        
        # Compare center line capacities (B)
        for i in 1:N
            @test diag(B1[i]) ≈ diag(B2[i])
        end
        
        # Compare staggered volumes (W)
        for i in 1:N
            @test diag(W1[i]) ≈ diag(W2[i])
        end
        
        # Compare volume centroids
        @test all(c1 ≈ c2 for (c1, c2) in zip(C_ω1, C_ω2))
        
        # Compare interface centroids
        @test all(c1 ≈ c2 for (c1, c2) in zip(C_γ1, C_γ2))
    end
end
