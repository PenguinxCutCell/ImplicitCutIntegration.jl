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
