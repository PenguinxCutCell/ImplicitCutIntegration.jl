using Test
using ImplicitCutIntegration
using LinearAlgebra: diag, norm
using Statistics: mean

# mesh and geometry parameters
const RADIUS = 0.45
const DOMAIN_EXTENT = 1.0

mesh = ntuple(_ -> collect(range(-DOMAIN_EXTENT, DOMAIN_EXTENT; length = 5)), 2)
body = (x, y) -> x^2 + y^2 - RADIUS^2

@time A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids = true)
