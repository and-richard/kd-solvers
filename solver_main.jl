using LinearAlgebra, StaticArrays, SparseArrays
using Parameters
using CairoMakie, LaTeXStrings, Printf
using Statistics
using Polyester
using SpecialFunctions
using OrdinaryDiffEq, OrdinaryDiffEqSSPRK
using DataInterpolations
using Integrals
using JLD2

# ============================================================================== #
# TABLE OF CONTENTS
# - PARAMETERS, DATA STRUCTURES, AND RECONSTRUCTION  : tag_params_structs_recon
# - SPATIAL DISCRETIZATION AND PDE SOLVER            : tag_grid_and_solver
# - DATA MANAGEMENT AND SERIALIZATION                : tag_data_and_serialization
# - VISUALIZATION AND ANALYSIS PIPELINE              : tag_visualization
# - SIMULATION CONFIGURATION AND EXECUTION           : tag_config_and_execution
# ============================================================================== #


# ============================================================================== #
# PARAMETERS, DATA STRUCTURES, AND RECONSTRUCTION
# tag_params_structs_recon
# ============================================================================== #

# Physical unit conversion factors:
# - OMEGA_CONVERSION: converts laser wavelength in nm to angular frequency ω in atomic units
# - SHELF_CONVERSION: converts pulse flat-top shelf duration in fs to dimensionless length L
const OMEGA_CONVERSION = 45.56335252912
const SHELF_CONVERSION = 1883.651567309

# --- Data Structures ---

# Fundamental constants and physical properties of the target particle
struct PhysicalUnits
    m::Float64
    q::Float64
    c::Float64
    ħ::Float64

    function PhysicalUnits(m::Real, q::Real, c::Real, ħ::Real)
        m > 0  || throw(ArgumentError("Particle mass m must be strictly positive (got $m)"))
        c > 0  || throw(ArgumentError("Speed of light c must be strictly positive (got $c)"))
        ħ > 0  || throw(ArgumentError("Reduced Planck constant ħ must be strictly positive (got $ħ)"))
        q != 0 || throw(ArgumentError("Particle charge q cannot be zero"))
        return new(Float64(m), Float64(q), Float64(c), Float64(ħ))
    end
end
PhysicalUnits(; m, q, c, ħ) = PhysicalUnits(m, q, c, ħ)
PhysicalUnits(nt::NamedTuple) = PhysicalUnits(; nt...)

# Optical pulse envelope parameters and duration
struct PulseParameters
    A0::Float64
    λ_nm::Float64
    σ::Float64
    shelf_duration_fs::Float64

    function PulseParameters(A0::Real, λ_nm::Real, σ::Real, shelf_duration_fs::Real = 0.0)
        A0 > 0                 || throw(ArgumentError("Field amplitude A0 must be strictly positive (got $A0)"))
        λ_nm > 0               || throw(ArgumentError("Laser wavelength λ_nm must be strictly positive (got $λ_nm)"))
        σ > 0                  || throw(ArgumentError("Pulse rise parameter σ must be strictly positive (got $σ)"))
        shelf_duration_fs >= 0 || throw(ArgumentError("Shelf duration must be non-negative (got $shelf_duration_fs)"))
        return new(Float64(A0), Float64(λ_nm), Float64(σ), Float64(shelf_duration_fs))
    end
end
PulseParameters(; A0, λ_nm, σ, shelf_duration_fs = 0.0) = PulseParameters(A0, λ_nm, σ, shelf_duration_fs)
PulseParameters(nt::NamedTuple) = PulseParameters(; nt...)

# Spatial and temporal simulation domain boundaries
struct Domain
    T_sim::NTuple{2, Float64}     # Total integration interval: (T_min, T_max)
    T_save::NTuple{2, Float64}    # Diagnostic recording window: (T_i, T_f)
    dZ_steps::NTuple{2, Float64}  # Spatial resolution bounds: (dZ_min, dZ_max)
    N_time_steps::Int             # Number of recorded time slices
end

# --- Kinematic and Parameter Conversion Helpers ---

@inline compute_omega(λ_nm::Real) = OMEGA_CONVERSION / Float64(λ_nm)
@inline compute_omega(p::PulseParameters) = compute_omega(p.λ_nm)

@inline compute_shelf_L(shelf_duration_fs::Real, λ_nm::Real) = SHELF_CONVERSION * Float64(shelf_duration_fs) / Float64(λ_nm)
@inline compute_shelf_L(p::PulseParameters) = compute_shelf_L(p.shelf_duration_fs, p.λ_nm)

# Minkowski inner product under metric signature (+, -, -, -)
@inline function four_dot(a::SVector{4, Float64}, b::SVector{4, Float64})
    return a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4]
end

# Modulated field amplitude for a flat-shelf pulse with half-Gaussian rising and falling edges
@inline function field_amplitude(φ, A0, inv_m2σ2, L)
    s = max(0.0, abs(φ) - 0.5 * L)
    return A0 * cos(φ) * exp(s * s * inv_m2σ2)
end

# Volkov phase spatial derivative w.r.t. laser phase φ_i
@inline function dSi(qa, m_inv_β, pϵ)
    return muladd(2.0, pϵ, qa) * qa * m_inv_β
end

# Resolves simulation finish time T_max for a given shelf L and pulse parameter σ.
# Supports functional expressions (L, σ) -> Real, explicit vectors/tuples of times,
# or fallback to symmetric termination about τ = 0.
function resolve_finish_time(T_max_spec, shelf_idx::Int, L::Float64, σ::Float64, T_min::Float64)
    T_symmetric = abs(T_min)
    if isnothing(T_max_spec)
        return T_symmetric
    elseif isa(T_max_spec, Function)
        return max(T_min + 1.0, Float64(T_max_spec(L, σ)))
    elseif isa(T_max_spec, Union{AbstractVector, Tuple})
        if shelf_idx <= length(T_max_spec)
            entry = T_max_spec[shelf_idx]
            if isnothing(entry) || entry === missing || (entry isa AbstractString && isempty(strip(entry)))
                return T_symmetric
            else
                return max(T_min + 1.0, Float64(entry))
            end
        else
            return T_symmetric
        end
    elseif isa(T_max_spec, Real)
        return max(T_min + 1.0, Float64(T_max_spec))
    else
        return T_symmetric
    end
end

# Evaluates 2nd-order non-uniform derivative for shock-bordering plateau slope diagnostics
function calc_slope_2nd_order(z0, z1, z2, y0, y1, y2)
    term0 = y0 * (2.0 * z0 - z1 - z2) / ((z0 - z1) * (z0 - z2))
    term1 = y1 * (z0 - z2) / ((z1 - z0) * (z1 - z2))
    term2 = y2 * (z0 - z1) / ((z2 - z0) * (z2 - z1))
    return term0 + term1 + term2
end

# Formats floating-point values into LaTeX scientific notation strings
function format_latex_sci(value::Float64)
    value == 0.0 && return "0.0"
    str_val = @sprintf("%.1e", value)
    parts = split(str_val, "e")
    mantissa = parts[1]
    exponent = parse(Int, parts[2])
    return exponent == 0 ? mantissa : "$mantissa \\times 10^{$exponent}"
end

# --- High-Order Numerical Reconstruction and Spline Quadrature ---

# Fifth-order WENO-Z reconstruction on a uniform five-point stencil
# Evaluates left-biased interface value from candidate stencils (S0, S1, S2)
# using Jiang-Shu smoothness indicators and Borges weights
function WENO5_Z(fm2, fm1, fc0, fp1, fp2)
    S0 =  2.0 * fm2 - 7.0 * fm1 + 11.0 * fc0
    S1 =  5.0 * fc0 + 2.0 * fp1 -  1.0 * fm1  
    S2 =  2.0 * fc0 + 5.0 * fp1 -  1.0 * fp2

    coeff = 13.0 / 12.0
    β0 = coeff * (fm2 - 2.0*fm1 + fc0)^2 + 0.25 * (fm2 - 4.0*fm1 + 3.0*fc0)^2
    β1 = coeff * (fm1 - 2.0*fc0 + fp1)^2 + 0.25 * (fm1 - fp1)^2
    β2 = coeff * (fc0 - 2.0*fp1 + fp2)^2 + 0.25 * (3.0*fc0 - 4.0*fp1 + fp2)^2

    τ5 = abs(β0 - β2)
    α0 = 0.1 * (1.0 + (τ5 / (β0 + 1e-40))^2)
    α1 = 0.6 * (1.0 + (τ5 / (β1 + 1e-40))^2)
    α2 = 0.3 * (1.0 + (τ5 / (β2 + 1e-40))^2)

    return (α0*S0 + α1*S1 + α2*S2) / (6.0 * (α0 + α1 + α2))
end

# Cumulative integration using piecewise Akima cubic Hermite spline interpolation
function cumint_akima!(dest::AbstractVecOrMat, in_vals, x_grid; col::Integer = 1)
    target = dest isa AbstractMatrix ? @view(dest[:, col]) : dest
    interp = AkimaInterpolation(in_vals, x_grid)
    
    idx_start = firstindex(target)
    target[idx_start] = zero(eltype(target))
    
    @inbounds for i in (idx_start + 1):lastindex(target)
        target[i] = target[i - 1] + DataInterpolations.integral(interp, x_grid[i - 1], x_grid[i])
    end

    return dest
end


# ============================================================================== #
# SPATIAL DISCRETIZATION AND PDE SOLVER
# tag_grid_and_solver
#
# Functions in this section:
# - compute_spatial_grid
# - nonuniform_grid_params
# - pde_system!
# - run_solver
# ============================================================================== #

# --- Non-Uniform Grid Generator ---

# Generates a non-uniform 1D spatial grid focused in the central standing-wave interaction
# zone by solving the inverse cumulative distribution function relation C(ζ) = ξ C(ζ_max)
function compute_spatial_grid(grid_params)
    @unpack Z_min, Z_max, dZ_min, dZ_max, L_drift, σ, L, shelf_pad = grid_params

    # Target point densities for the fine central flat-shelf region (rho_max) 
    # and coarse marginal regions (rho_min)
    rho_max = 1.0 / dZ_min
    rho_min = 1.0 / dZ_max
    delta_rho = rho_max - rho_min
    
    sqrt_2 = sqrt(2.0)
    sqrt_pi_over_2 = sqrt(pi) / sqrt_2
    inv_σ = 1.0 / σ

    # Effective interaction region boundaries accounting for guiding-center drift
    Z_base = 0.5 * (L + sqrt_2 * σ) + abs(shelf_pad)
    Z_active_L = -Z_base + min(L_drift, 0.0)
    Z_active_R =  Z_base + max(L_drift, 0.0)
    
    tail_gauss_integral(z) = delta_rho * 0.5 * sqrt_pi_over_2 * σ * erf(z * inv_σ) + rho_min * z
    
    function C_raw(Z)
        if Z > Z_active_R
            return rho_max * Z_active_R + tail_gauss_integral(Z - Z_active_R)
        elseif Z < Z_active_L
            return rho_max * Z_active_L - tail_gauss_integral(-Z + Z_active_L)
        else
            return rho_max * Z
        end
    end
    
    # Cumulative node allocation function across the domain
    C_min = C_raw(Z_min)
    C_max = C_raw(Z_max) - C_min
    C_cdf(Z) = C_raw(Z) - C_min

    # Node count covering round(C_max) spatial cells
    N = round(Int, C_max) + 1
    C_shelf_L = C_cdf(Z_active_L)
    C_shelf_R = C_cdf(Z_active_R)
    grid_pad  = 5.0 * dZ_max

    # Solves C(ζ) = target algebraically on the shelf or via bisection on the falloff tails
    @inline function invert_cdf(ξ, lo_falloff, hi_falloff)
        target = ξ * C_max

        # Algebraic solution within the linear uniform shelf
        if C_shelf_L <= target <= C_shelf_R
            return Z_active_L + (target - C_shelf_L) / rho_max
        end

        # Numerical bisection within the Gaussian falloff tails
        lo = ifelse(target < C_shelf_L, lo_falloff, Z_active_R)
        hi = ifelse(target < C_shelf_L, Z_active_L, hi_falloff)

        for _ in 1:60
            mid = 0.5 * (lo + hi)
            is_less = C_cdf(mid) < target
            lo = ifelse(is_less, mid, lo)
            hi = ifelse(is_less, hi, mid)
        end

        return 0.5 * (lo + hi)
    end

    # Coordinates of uniform ξ grid cell centers
    Z_grid = zeros(Float64, N)
    Z_grid[1] = Z_min
    Z_grid[end] = Z_max
    @inbounds @batch for i in 2:(N - 1)
        ξ_i = (i - 1) / (N - 1)
        Z_grid[i] = invert_cdf(ξ_i, Z_min, Z_max)
    end

    # Coordinates of uniform ξ grid cell interfaces
    Z_inter = zeros(Float64, N + 1)
    @inbounds @batch for j in 1:(N + 1)
        ξ_j = (j - 1.5) / (N - 1)
        Z_inter[j] = invert_cdf(ξ_j, Z_min - grid_pad, Z_max + grid_pad)
    end

    # Extrapolated ghost-node coordinates for boundary stencils
    Z_ext = zeros(Float64, N + 6)
    Z_ext[4:N+3] .= Z_grid
    
    dZ_left  = Z_grid[2] - Z_grid[1]
    dZ_right = Z_grid[end] - Z_grid[end-1]
    
    Z_ext[3] = Z_grid[1] - dZ_left
    Z_ext[2] = Z_grid[1] - 2.0 * dZ_left
    Z_ext[1] = Z_grid[1] - 3.0 * dZ_left
    
    Z_ext[N+4] = Z_grid[end] + dZ_right
    Z_ext[N+5] = Z_grid[end] + 2.0 * dZ_right
    Z_ext[N+6] = Z_grid[end] + 3.0 * dZ_right

    # Precomputed inverse non-uniform grid metrics
    inv_dZ_nodes = 1.0 ./ diff(Z_ext[2:N+5])
    inv_dZ_cells = 1.0 ./ diff(Z_inter)     

    return N, Z_grid, Z_inter, inv_dZ_nodes, inv_dZ_cells
end

# --- Parameter and Cache Bundler ---

# Assembles pre-allocated caches for state extension, interface numerical fluxes,
# extrema tracking, non-uniform spatial grid metrics, and background kinematics.
function nonuniform_grid_params(constants, grid_params)
    @unpack q, A0, α, β1, β2, pϵ1, pϵ2 = constants
    @unpack σ, L = grid_params

    N, _, Z_inter, inv_dZ_nodes, inv_dZ_cells = compute_spatial_grid(grid_params)

    caches = (;
        H_flux = zeros(Float64, N + 1),
        ΘZ_ext = zeros(Float64, N + 6),
        extrema_track = [-Inf, Inf]
    )

    grid = (; N, Z_inter, inv_dZ_nodes, inv_dZ_cells)

    phys = (;
        q, A0, inv_m2σ2 = -0.5 / (σ * σ), L, α,
        β_p = β1 + β2, β_m = β2 - β1,
        m_inv_β1 = -1.0 / β1, m_inv_β2 = -1.0 / β2,
        pϵ1, pϵ2
    )

    return (; caches, grid, phys)
end

# --- Core PDE Operator ---

# Evaluates the semi-discrete right-hand side du/dt for the semiclassical Hamilton-Jacobi
# phase gradient equation ∂_τ Θ_ζ + ∂_ζ H(Θ_ζ) = 0 via fifth-order WENO-Z and LLF flux.
function pde_system!(du, u, params, T)
    (; caches, grid, phys) = params
    (; H_flux, ΘZ_ext) = caches
    (; N, Z_inter, inv_dZ_cells) = grid
    (; q, A0, inv_m2σ2, L, α, β_p, β_m, m_inv_β1, m_inv_β2, pϵ1, pϵ2) = phys

    # --- State Unpacking and Ghost Padding ---
    @inbounds @batch for i in 1:N
        ΘZ_ext[i+3] = u[i]
    end
    
    # Constant zeroth-order extrapolation across boundary ghost nodes
    ΘZ_1 = u[1]; ΘZ_N = u[N]
    ΘZ_ext[1]   = ΘZ_1; ΘZ_ext[2]   = ΘZ_1; ΘZ_ext[3]   = ΘZ_1
    ΘZ_ext[N+4] = ΘZ_N; ΘZ_ext[N+5] = ΘZ_N; ΘZ_ext[N+6] = ΘZ_N

    # --- Interface Flux Evaluation and LLF Stabilization ---
    @inbounds @batch for i in eachindex(H_flux)
        Z = Z_inter[i]

        # Fifth-order WENO-Z reconstruction to cell interface from left and right
        ΘZ_L = WENO5_Z(ΘZ_ext[i],   ΘZ_ext[i+1], ΘZ_ext[i+2], ΘZ_ext[i+3], ΘZ_ext[i+4])
        ΘZ_R = WENO5_Z(ΘZ_ext[i+5], ΘZ_ext[i+4], ΘZ_ext[i+3], ΘZ_ext[i+2], ΘZ_ext[i+1])

        # Local field amplitudes and Volkov phase derivatives
        qa1 = q * field_amplitude(T - Z, A0, inv_m2σ2, L)
        qa2 = q * field_amplitude(T + Z, A0, inv_m2σ2, L)
        dS1 = dSi(qa1, m_inv_β1, pϵ1)
        dS2 = dSi(qa2, m_inv_β2, pϵ2)

        # Light-cone potential components and background interaction term
        W_term = 4.0 * (2.0 * qa1 * qa2 - α * dS1 * dS2)
        B_m = muladd(α, dS2 - dS1, β_m)
        B_p = muladd(-α, dS1 + dS2, β_p)
        B_p2 = B_p * B_p

        # Semiclassical Hamilton-Jacobi flux candidate polynomials
        αΘZ_L = α * ΘZ_L
        αΘZ_R = α * ΘZ_R

        C_L = muladd(muladd(2.0, B_m, αΘZ_L), ΘZ_L, W_term)
        C_R = muladd(muladd(2.0, B_m, αΘZ_R), ΘZ_R, W_term)

        sqrtΔ_L = sqrt(max(muladd(α, C_L, B_p2), 0.0))
        sqrtΔ_R = sqrt(max(muladd(α, C_R, B_p2), 0.0))

        H_L = C_L / (B_p + sqrtΔ_L)
        H_R = C_R / (B_p + sqrtΔ_R)

        # Local characteristic wave speed for numerical dissipation
        abs_dH_L = abs(αΘZ_L + B_m) / (sqrtΔ_L + 1e-16)
        abs_dH_R = abs(αΘZ_R + B_m) / (sqrtΔ_R + 1e-16)
        dissip = min(max(abs_dH_L, abs_dH_R), 1.0)

        # Local Lax-Friedrichs (LLF) numerical flux
        H_flux[i] = 0.5 * (H_L + H_R - dissip * (ΘZ_R - ΘZ_L))
    end

    # --- Conservative Flux Divergence ---
    @inbounds @batch for i in 1:N
        du[i] = (H_flux[i] - H_flux[i+1]) * inv_dZ_cells[i]
    end
end

# --- Time-Integration Driver ---

# Configures kinematic invariants, prepares non-uniform grid structures, sets up continuous
# extrema tracking callbacks, and integrates the PDE using SSPRK43.
function run_solver(P::SVector{4, Float64}, domain::Domain, units::PhysicalUnits, 
                    pulse_params::PulseParameters, polarizations::NTuple{2, SVector{4, Float64}}, 
                    propagations::NTuple{2, SVector{4, Float64}})
    @unpack m, q, c, ħ = units
    @unpack T_sim, T_save, dZ_steps, N_time_steps = domain
    T_min, T_max = T_sim
    T_i, T_f = T_save
    dZ_min, dZ_max = dZ_steps

    (; A0, σ) = pulse_params
    ω = compute_omega(pulse_params)
    L = compute_shelf_L(pulse_params)
    ϵ1, ϵ2 = polarizations
    n1, n2 = propagations

    # Constants and relativistic invariants
    k = ω / c; k1 = k * n1; k2 = k * n2
    ħpk1 = ħ * four_dot(P, k1); ħpk2 = ħ * four_dot(P, k2)
    pϵ1 = four_dot(P, ϵ1); pϵ2 = four_dot(P, ϵ2)

    α = 2.0 * ħ * ħ * four_dot(k1, k2)
    β1 = 2.0 * ħpk1; β2 = 2.0 * ħpk2

    # Longitudinal drift velocity and active spatial domain limits
    v_drift = (β2 - β1) / (β1 + β2)
    L_drift = v_drift * (T_max - T_min)
    
    Z_base = 0.5 * L + 5.0 * σ / sqrt(2.0) + 1.0
    Z_min = -Z_base + min(L_drift, 0.0)
    Z_max =  Z_base + max(L_drift, 0.0)

    constants = (; q, A0, α, β1, β2, pϵ1, pϵ2)
    grid_params = (; Z_min, Z_max, dZ_min, dZ_max, L_drift, σ, L, shelf_pad = 1.0)

    pde_params = nonuniform_grid_params(constants, grid_params)
    u0 = zeros(pde_params.grid.N)

    # Continuous global extrema tracker across all intermediate integrator sub-stages
    condition(u, t, integrator) = true
    function affect!(integrator)
        current_min, current_max = extrema(integrator.u)
        track = integrator.p.caches.extrema_track
        track[1] = max(track[1], current_max)
        track[2] = min(track[2], current_min)
    end
    tracking_cb = DiscreteCallback(condition, affect!, save_positions = (false, false))

    T_range = range(T_i, T_f, length = N_time_steps + 1)
    prob = ODEProblem(pde_system!, u0, (T_min, T_max), pde_params)
    sol = solve(prob, SSPRK43(), abstol = 1e-12, reltol = 1e-4, 
                callback = tracking_cb, saveat = T_range, save_everystep = false, save_start = false)

    # Clean stray pre-recorded endpoint if solver prepends initial time
    u_saved = (length(sol.u) > length(T_range) && sol.t[1] < T_range[1] - 1e-10) ? sol.u[2:end] : sol.u

    ΘZ_array = reduce(hcat, u_saved)
    final_max, final_min = pde_params.caches.extrema_track
    P_KD = final_max - final_min
    
    return grid_params, T_range, ΘZ_array, P_KD
end


# ============================================================================== #
# DATA MANAGEMENT AND SERIALIZATION
# tag_data_and_serialization
#
# Functions in this section:
# - get_workspace_paths
# - resolve_run_filename
# - resolve_run_filenames
# - resolve_data_source
# - run_and_save
# - interactive_loader
# ============================================================================== #

# --- Directory Scaffold ---

# Constructs the fixed internal directories for datasets, visual figures, and animations
function get_workspace_paths(base_dir::String, workspace_name::Union{AbstractString, Symbol} = "Workspace_Main")
    workspace = joinpath(base_dir, string(workspace_name))
    data_dir  = joinpath(workspace, "Data")
    plots_dir = joinpath(workspace, "Plots")
    anims_dir = joinpath(workspace, "Animations")
    
    for d in (workspace, data_dir, plots_dir, anims_dir)
        isdir(d) || mkpath(d)
    end
    return (; workspace, data_dir, plots_dir, anims_dir)
end

# --- File Name Handler ---

# Formulates a collision-safe identifier and file path, auto-incrementing if requested
function resolve_run_filename(dir_path::String, name_spec = nothing; reserved::Set{String} = Set{String}())
    isdir(dir_path) || mkpath(dir_path)
    existing_files = readdir(dir_path)
    
    existing_indices = Int[]
    for f in existing_files
        m = match(r"^Run_(\d+)\.jld2$", f)
        m !== nothing && push!(existing_indices, parse(Int, m.captures[1]))
    end
    for r in reserved
        m = match(r"^Run_(\d+)$", r)
        m !== nothing && push!(existing_indices, parse(Int, m.captures[1]))
    end
    next_auto_idx = isempty(existing_indices) ? 1 : maximum(existing_indices) + 1

    is_blank(x) = isnothing(x) || x === missing || x === :_ || x === "_" || (x isa AbstractString && isempty(strip(x)))
    if is_blank(name_spec)
        run_name = "Run_$(next_auto_idx)"
    else
        raw_str = string(name_spec)
        run_name = startswith(raw_str, "Run_") ? raw_str : "Run_$(raw_str)"
    end

    base_candidate = run_name
    counter = 1
    while ("$run_name.jld2" in existing_files) || (run_name in reserved)
        counter += 1
        run_name = "$(base_candidate)_$(counter)"
    end

    push!(reserved, run_name)
    file_path = joinpath(dir_path, "$run_name.jld2")
    return run_name, file_path
end

# --- Batch File Name Resolver ---

# Resolves a collection of run identifiers across a shared reservation registry
function resolve_run_filenames(dir_path::String, specs)
    reserved = Set{String}()
    if specs isa Union{Tuple, AbstractVector}
        return [resolve_run_filename(dir_path, s; reserved = reserved) for s in specs]
    else
        return resolve_run_filename(dir_path, specs; reserved = reserved)
    end
end

# --- Dual-Input Source Resolver ---

# Extracts file path and workspace geometry from either a loaded data_pack Dict or raw string path
function resolve_data_source(source::Union{Dict, AbstractString})
    if source isa Dict
        haskey(source, :file_path) || throw(ArgumentError("data_pack must contain the :file_path key."))
        return String(source[:file_path]), source[:workspace]
    else
        file_path = normpath(abspath(String(source)))
        isfile(file_path) || throw(ArgumentError("Specified dataset file does not exist: $file_path"))
        
        parent1 = dirname(file_path)
        parent2 = dirname(parent1)
        if basename(parent1) == "Data"
            ws = get_workspace_paths(dirname(parent2), basename(parent2))
        else
            ws = get_workspace_paths(parent1, "Workspace_Main")
        end
        return file_path, ws
    end
end

# --- Batch Simulation Driver and Data Serializer ---

# Executes grid parameter exploration across combinations of initial momenta and shelf durations,
# serializing the resulting fields and observables hierarchically into a compressed JLD2 archive.
function run_and_save(cfg::NamedTuple)
    @unpack units, pulse_params, initial_momenta, shelf_durations_fs, spatial_resolutions, 
            N_time_steps, work_directory = cfg

    ws_name = get(cfg, :workspace_name, "Workspace_Main")
    ws = get_workspace_paths(work_directory, ws_name)
    req_name = get(cfg, :run_name, nothing)
    run_name, file_path = resolve_run_filename(ws.data_dir, req_name)
    println("Initializing batch run: $run_name (Workspace: $ws_name)")

    (; m, q, c, ħ) = units
    (; A0, λ_nm, σ) = pulse_params
    ω = compute_omega(pulse_params)

    ϵ1 = SVector(0.0, 1.0, 0.0,  0.0); ϵ2 = SVector(0.0, 1.0, 0.0,  0.0)
    n1 = SVector(1.0, 0.0, 0.0,  1.0); n2 = SVector(1.0, 0.0, 0.0, -1.0)

    T_max_spec = get(cfg, :T_max, nothing)
    raw_Ti_spec = get(cfg, :T_i, nothing)
    raw_Tf_spec = get(cfg, :T_f, nothing)

    num_momenta = length(initial_momenta)
    num_shelves = length(shelf_durations_fs)

    jldopen(file_path, "w"; compress = true) do file
        file["metadata/units"]               = (m = units.m, q = units.q, c = units.c, ħ = units.ħ)
        file["metadata/pulse_params"]        = (A0 = A0, λ_nm = λ_nm, σ = σ, ω = ω)
        file["metadata/polarizations"]       = (ϵ1, ϵ2)
        file["metadata/propagations"]        = (n1, n2)
        file["metadata/initial_momenta"]     = initial_momenta
        file["metadata/shelf_lengths_fs"]    = shelf_durations_fs
        file["metadata/spatial_separations"] = spatial_resolutions
        file["metadata/N_time_steps"]        = N_time_steps
        file["metadata/work_directory"]      = work_directory
        file["metadata/workspace_name"]      = string(ws_name)

        for (i, p) in enumerate(initial_momenta)
            P = SVector{4, Float64}(sqrt(dot(p, p) + m*m * c*c), p[1], p[2], p[3])
            
            for (j, L_fs) in enumerate(shelf_durations_fs)
                L = compute_shelf_L(L_fs, λ_nm)
                active_pulse = PulseParameters(A0 = A0, λ_nm = λ_nm, σ = σ, shelf_duration_fs = L_fs)

                T_min = -(0.5 * L + 5.0 * σ + 1.0)
                T_max_sim = resolve_finish_time(T_max_spec, j, L, σ, T_min)

                T_i_val = isnothing(raw_Ti_spec) ? T_min : raw_Ti_spec
                T_f_val = isnothing(raw_Tf_spec) ? T_max_sim : raw_Tf_spec
                T_i_clamped = clamp(min(T_i_val, T_f_val), T_min, T_max_sim)
                T_f_clamped = clamp(max(T_i_val, T_f_val), T_min, T_max_sim)

                domain = Domain((T_min, T_max_sim), (T_i_clamped, T_f_clamped), spatial_resolutions, N_time_steps)

                println("Batch [$i/$num_momenta, $j/$num_shelves]: p = $p | L = $(round(L; digits=1)) ($(L_fs) fs)")
                println("  Domain: τ ∈ [$T_min, $T_max_sim] | Window: τ ∈ [$T_i_clamped, $T_f_clamped]")
                
                @time grid_params, T_range, ΘZ_array, P_KD = run_solver(P, domain, units, active_pulse, (ϵ1, ϵ2), (n1, n2))

                grp = "p_$i/L_$j"
                file["$grp/metadata/T_sim"]        = (T_min, T_max_sim)
                file["$grp/metadata/T_interval"]   = (T_range[1], T_range[end])
                file["$grp/metadata/grid_params"]  = grid_params
                file["$grp/metadata/L_dimension"]  = L
                file["$grp/metadata/L_shelf_fs"]   = L_fs
                
                file["$grp/Theta_zeta"] = Float32.(ΘZ_array)
                file["$grp/P_KD"]       = P_KD
            end
        end
    end
    println("Batch execution complete. Saved: $file_path")
end

# --- Interactive REPL Dataset Loader ---

# Lists available run archives inside the workspace and interactively loads selected field groups into memory
function interactive_loader(base_dir::String, workspace_name::Union{AbstractString, Symbol} = "Workspace_Main")
    ws = get_workspace_paths(base_dir, workspace_name)
    files = filter(f -> startswith(f, "Run") && endswith(f, ".jld2"), readdir(ws.data_dir))
    
    if isempty(files)
        return println("No Run files found in $(ws.data_dir).")
    end

    println("\n=== Available Runs in [$(ws.workspace)] ===")
    for (idx, f) in enumerate(files)
        println("  [$idx] $f")
    end
    print("Select a run index (or 0 to cancel): ")
    run_choice = parse(Int, readline())
    (run_choice == 0 || run_choice > length(files)) && return println("Operation cancelled.")
    
    target_file = joinpath(ws.data_dir, files[run_choice])
    file = jldopen(target_file, "r")

    momenta = file["metadata/initial_momenta"]
    shelves = file["metadata/shelf_lengths_fs"]
    N_time_steps = file["metadata/N_time_steps"]

    println("\n=== Available Initial Momenta ===")
    for (idx, p) in enumerate(momenta)
        println("  [$idx] p = $p")
    end
    print("Select a momentum index (or 0 to cancel): ")
    p_choice = parse(Int, readline())
    if p_choice == 0 || p_choice > length(momenta)
        close(file); return println("Operation cancelled.")
    end

    println("\n=== Available Shelf Lengths ===")
    for (idx, L) in enumerate(shelves)
        println("  [$idx] L = $L fs")
    end
    print("Select a shelf length index (or 0 to cancel): ")
    L_choice = parse(Int, readline())
    if L_choice == 0 || L_choice > length(shelves)
        close(file); return println("Operation cancelled.")
    end

    grp = "p_$p_choice/L_$L_choice"
    println("\nLoading data from $grp...")

    T_bounds = file["$grp/metadata/T_interval"]
    grid_params = file["$grp/metadata/grid_params"]
    _, Z_grid, _, _, _ = compute_spatial_grid(grid_params)

    data_pack = Dict{Symbol, Any}(
        :file_path          => target_file,
        :units              => file["metadata/units"],
        :pulse_params       => file["metadata/pulse_params"],
        :polarizations      => file["metadata/polarizations"],
        :propagations       => file["metadata/propagations"],
        :selected_momentum  => momenta[p_choice],
        :selected_shelf_fs  => shelves[L_choice],
        :T_interval         => T_bounds,
        :N_time_steps       => N_time_steps,
        :Z_grid             => Z_grid,
        :Theta_zeta         => file["$grp/Theta_zeta"],
        :P_KD               => file["$grp/P_KD"],
        :workspace          => ws
    )

    close(file)
    println("Load successful. Data returned to REPL.")
    return data_pack
end


# ============================================================================== #
# VISUALIZATION AND ANALYSIS PIPELINE
# tag_visualization
#
# Functions in this section:
# - decimate_heatmap_extrema
# - animate_results
# - generate_density_plot
# - plot_zoomed_snapshot
# - plot_thermalization_summary
# - plot_PKD
# - plot_PKD_aggregated
# - generate_integral_table
# ============================================================================== #

# --- Extrema-Preserving Decimation Engine ---

# Reduces 2D spatial-temporal matrix dimensions for responsive rendering while strictly
# preserving the positive or negative peak extrema of caustic shocks and steep gradients
function decimate_heatmap_extrema(Z_grid::AbstractVector{Float64}, T_grid::AbstractVector{Float64}, 
                                  matrix::AbstractMatrix, target_Z::Int = 1500, target_T::Int = 1000)
    Nz, Nt = size(matrix)
    step_z = max(1, ceil(Int, Nz / target_Z))
    step_t = max(1, ceil(Int, Nt / target_T))
    
    # Bypass downsampling if the matrix already falls within target dimensions
    if step_z == 1 && step_t == 1
        return Z_grid, T_grid, matrix
    end
    
    new_Nz = length(1:step_z:Nz)
    new_Nt = length(1:step_t:Nt)
    
    new_mat = Matrix{Float32}(undef, new_Nz, new_Nt)
    new_Z   = zeros(Float64, new_Nz)
    new_T   = zeros(Float64, new_Nt)
    
    # Precompute spatial coordinate midpoints for all blocks
    @inbounds for (i_new, i) in enumerate(1:step_z:Nz)
        i_end = min(i + step_z - 1, Nz)
        new_Z[i_new] = 0.5 * (Z_grid[i] + Z_grid[i_end])
    end

    # Multi-threaded temporal decimation pass
    @inbounds @batch for j_new in 1:new_Nt
        j = 1 + (j_new - 1) * step_t
        j_end = min(j + step_t - 1, Nt)
        new_T[j_new] = 0.5 * (T_grid[j] + T_grid[j_end])
        
        for (i_new, i) in enumerate(1:step_z:Nz)
            i_end = min(i + step_z - 1, Nz)
            block = view(matrix, i:i_end, j:j_end)
            
            # Identify dominant signed extremum across the block
            local_max = maximum(block)
            local_min = minimum(block)
            new_mat[i_new, j_new] = abs(local_max) > abs(local_min) ? local_max : local_min
        end
    end
    
    println("Extrema decimation complete: ($Nz × $Nt) -> ($new_Nz × $new_Nt) nodes.")
    return new_Z, new_T, new_mat
end

# --- Evolution Animation Generator ---

# Produces an animated video of phase gradient Θ_ζ or integrated phase Θ across the spatial grid
function animate_results(data_pack::Dict, variable::Symbol;
                         zeta_lims::Tuple{Float64, Float64} = (-10.0, 10.0),
                         time_window::Tuple{Float64, Float64} = (-Inf, Inf),
                         fps::Int = 30,
                         save_filename::String = "semiclassical_evolution.mp4")

    # --- Data Extraction and Coordinate Reconciliation ---
    ws = data_pack[:workspace]
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.anims_dir, save_filename)

    x_grid = data_pack[:Z_grid]
    T_interval = data_pack[:T_interval]
    raw_data_mat = data_pack[:Theta_zeta]
    px, py, pz = round.(data_pack[:selected_momentum]; digits = 3)

    num_spatial = size(raw_data_mat, 1)
    num_total_frames = size(raw_data_mat, 2)

    T_range = range(T_interval[1], T_interval[2], length = num_total_frames)
    valid_indices = findall(t -> time_window[1] <= t <= time_window[2], T_range)

    if isempty(valid_indices)
        return println("Warning: No frames found within time window $(time_window).")
    end

    num_frames = length(valid_indices)
    is_gradient = (variable === :Theta_zeta)
    y_data_mat = Matrix{Float64}(undef, num_spatial, num_frames)

    # --- Parallel Field Integration ---
    if is_gradient
        y_data_mat .= view(raw_data_mat, :, valid_indices)
    else
        println("Integrating phase profiles in parallel...")
        @batch for idx in eachindex(valid_indices)
            matrix_idx = valid_indices[idx]
            cumint_akima!(y_data_mat, view(raw_data_mat, :, matrix_idx), x_grid; col = idx)
        end
    end

    # --- Canvas Layout and Styling Constants ---
    fig_size   = (800, 600)
    line_color = :dodgerblue
    line_width = 2.5
    title_size = 22
    axis_size  = 20

    data_min, data_max = extrema(y_data_mat)
    rel_pad = max(0.01 * abs(data_max - data_min), 1e-6)
    lims_min, lims_max = zeta_lims
    lim_pad = max(0.01 * abs(lims_max - lims_min), 1e-6)

    ylabel_sym = is_gradient ? L"\Theta_\zeta(\tau, \zeta)" : L"\Theta(\tau, \zeta)"
    title_str = L"%$(ylabel_sym) \text{ for } \mathbf{p} = (%$px, %$py, %$pz)"

    fig = Figure(size = fig_size)
    ax = Axis(fig[1, 1], title = title_str, titlesize = title_size, 
        xlabel = L"\zeta = kz", ylabel = "", xlabelsize = axis_size, ylabelsize = axis_size,
        limits = ((lims_min - lim_pad, lims_max + lim_pad), (data_min - rel_pad, data_max + rel_pad)),
        yticks = range(data_min, data_max, length = 7),
        ytickformat = values -> [@sprintf("%.2e", v) for v in values])

    # --- Animation Rendering Pipeline ---
    current_y = Observable(y_data_mat[:, 1])
    lines!(ax, x_grid, current_y, color = line_color, linewidth = line_width)

    time_index = Observable(1)
    text_label = @lift("T: $(@sprintf("%.3f", T_range[valid_indices[$time_index]]))")
    text!(ax, 0.05, 0.9, text = text_label, space = :relative, fontsize = 18)

    println("Rendering $(num_frames) frames at $fps FPS...")
    record(fig, save_path, 1:num_frames; framerate = fps) do step_idx
        time_index[] = step_idx
        current_y[] = view(y_data_mat, :, step_idx)
    end

    println("Saved animation: $save_path")
    return nothing
end

# --- Publication Schematics and Analysis Tools ---

# Generates a visual schematic showing the non-uniform node density distribution and interaction regions
function generate_density_plot(data_source::Union{Dict, AbstractString} = ""; save_filename::String = "Node_Density_Schematic.pdf")
    # --- Source Resolution and Workspace Scaffold ---
    ws = if data_source isa Dict
        data_source[:workspace]
    elseif !isempty(data_source) && isdir(String(data_source))
        get_workspace_paths(String(data_source), "Workspace_Main")
    elseif !isempty(data_source) && isfile(String(data_source))
        _, ws_res = resolve_data_source(data_source)
        ws_res
    else
        get_workspace_paths(pwd(), "Workspace_Main")
    end
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)

    # --- Proportional Geometry and Domain Partitioning ---
    L = 67.0
    σ_int = 14.0
    l_drift = 13.0
    l_pad = 13.0
    r_zeta = 4

    rho_max = 1.0 / 1e-3
    rho_min = 1.0 / 1e-2

    z_bL = -0.5 * L
    z_bR =  0.5 * L
    z_aL = z_bL - σ_int
    z_aR = z_bR + σ_int
    z_adR = z_aR + l_drift
    z_actL = z_aL - l_pad
    z_actR = z_adR + l_pad
    z_min = z_actL - r_zeta * σ_int
    z_max = z_actR + r_zeta * σ_int

    z_lims = (z_min - 1.0 * σ_int, z_max + 1.0 * σ_int)
    Z_plot = range(z_lims[1], z_lims[2], length = 1000)

    # --- Analytical Conceptual Density Curve ---
    function rho_conceptual(z)
        if z < z_actL
            return (rho_max - rho_min) * exp(-0.5 * ((z - z_actL) / σ_int)^2) + rho_min
        elseif z > z_actR
            return (rho_max - rho_min) * exp(-0.5 * ((z - z_actR) / σ_int)^2) + rho_min
        else
            return rho_max
        end
    end

    rho_plot = rho_conceptual.(Z_plot)

    # --- Canvas Layout and Typography Constants ---
    fig_size    = (1000, 750)
    fig_padding = (10, 10, 5, 5)

    main_labelsize      = 38    # Primary axis titles (ζ = kz, ρ(ζ))
    main_ticklabelsize  = 38    # Discrete tick notation along ζ
    main_leg_labelsize  = 40    # Upper-right analytical boundary legend
    guideline_labelsize = 36    # Asymptotic density bounds (ρ_min, ρ_max)
    region_labelsize    = 36    # Vertical sector descriptors (Active ext., Drift ext., Padding)
    span_labelsize      = 38    # Boundary interval offsets (±σ_int, +l_drift, ±l_pad)
    shelf_labelsize     = 34    # Central shelf label and ±0.5L coordinate marks

    fig = Figure(size = fig_size, fontsize = main_labelsize, figure_padding = fig_padding)

    # Vacuum and boundary tick coordinates
    val_left_outer = [z_actL - i * σ_int for i in r_zeta:-1:1]
    lbl_left_outer = [i == r_zeta ? L"\zeta_{\text{\textbf{min}}}" : "" for i in r_zeta:-1:1]
    val_left_inner = [z_actL, z_aL, z_bL]
    lbl_left_inner = [L"\zeta_{\text{\textbf{active}}}^{-}", "", ""]
    val_center = [0.0]
    lbl_center = [L"0"]
    val_right_inner = [z_bR, z_aR, z_adR, z_actR]
    lbl_right_inner = ["", "", "", L"\zeta_{\text{\textbf{active}}}^{+}"]
    val_right_outer = [z_actR + i * σ_int for i in 1:r_zeta]
    lbl_right_outer = [i == r_zeta ? L"\zeta_{\text{\textbf{max}}}" : "" for i in 1:r_zeta]

    custom_xticks = (
        vcat(val_left_outer, val_left_inner, val_center, val_right_inner, val_right_outer),
        vcat(lbl_left_outer, lbl_left_inner, lbl_center, lbl_right_inner, lbl_right_outer)
    )

    ax = Axis(fig[1, 1],
        xlabel = L"$\zeta = kz$",
        ylabel = L"Node Density $\rho(\zeta)$",
        xticks = custom_xticks,
        xlabelsize = main_labelsize,
        ylabelsize = main_labelsize,
        xticklabelsize = main_ticklabelsize,
        xtickwidth = 2.5,
        xticksize = 10,
        ytickwidth = 2.5,
        yticksize = 10,
        yticks = [rho_min, 0.5 * rho_max, rho_max],
        yticksvisible = true,
        yticklabelsvisible = false,
        xgridvisible = false,
        ygridvisible = false,
        bottomspinevisible = true,
        leftspinevisible = true,
        topspinevisible = false,
        rightspinevisible = false
    )

    y_bottom = 0.0
    ylims!(ax, y_bottom, rho_max * 1.25)
    xlims!(ax, z_lims[1], z_lims[2])

    # --- Background Colored Sectors ---
    band!(ax, [z_bL, z_bR], [y_bottom, y_bottom], [rho_max, rho_max], color = (:dodgerblue, 0.15))
    band!(ax, [z_aL, z_bL], [y_bottom, y_bottom], [rho_max, rho_max], color = (:green, 0.15))
    band!(ax, [z_bR, z_aR], [y_bottom, y_bottom], [rho_max, rho_max], color = (:green, 0.15))
    band!(ax, [z_aR, z_adR], [y_bottom, y_bottom], [rho_max, rho_max], color = (:firebrick, 0.15))
    band!(ax, [z_actL, z_aL], [y_bottom, y_bottom], [rho_max, rho_max], color = (:mediumorchid, 0.25))
    band!(ax, [z_adR, z_actR], [y_bottom, y_bottom], [rho_max, rho_max], color = (:mediumorchid, 0.25))

    # --- Asymptotic Guidelines and Density Limit Labels ---
    rho_labels_y = (rho_min, rho_max)
    hlines!(ax, [rho_labels_y[1], rho_labels_y[2]], color = (:black, 0.4), linestyle = (:dot, :dense), linewidth = 4.0)
    text!(ax, z_min - 0.9 * σ_int, rho_labels_y[1] - 0.90 * rho_min,
          text = L"\rho_{\text{\textbf{min}}} = 1/\Delta \zeta_{\text{\textbf{max}}}", 
          align = (:left, :bottom), offset = (0, 6), fontsize = guideline_labelsize)
    text!(ax, z_min - 0.9 * σ_int, rho_labels_y[2] - 0.05 * rho_min,
          text = L"\rho_{\text{\textbf{max}}} = 1/\Delta \zeta_{\text{\textbf{min}}}", 
          align = (:left, :bottom), offset = (0, 6), fontsize = guideline_labelsize)

    # --- Region Band Name Labels ---
    y_text = (rho_max + rho_min) * 0.5
    text!(ax, 0.0, y_text, text = "Initial shelf", align = (:center, :center), fontsize = shelf_labelsize)
    text!(ax, (z_aL + z_bL) / 2, y_text, text = "Active ext.", align = (:center, :center), rotation = pi / 2, fontsize = region_labelsize)
    text!(ax, (z_bR + z_aR) / 2, y_text, text = "Active ext.", align = (:center, :center), rotation = pi / 2, fontsize = region_labelsize)
    text!(ax, (z_aR + z_adR) / 2, y_text, text = "Drift ext.", align = (:center, :center), rotation = pi / 2, fontsize = region_labelsize)
    text!(ax, (z_actL + z_aL) / 2, y_text, text = "Padding", align = (:center, :center), rotation = pi / 2, fontsize = region_labelsize)
    text!(ax, (z_adR + z_actR) / 2, y_text, text = "Padding", align = (:center, :center), rotation = pi / 2, fontsize = region_labelsize)

    # --- Bottom Coordinate Span Markers ---
    y_promoted = 0.05 * rho_min
    text!(ax, (z_aL + z_bL) / 2, y_promoted, text = L"-\sigma_{\text{\textbf{int}}}", align = (:left, :center), rotation = pi / 2, fontsize = span_labelsize)
    text!(ax, (z_bL + 0.0) / 2, y_promoted, text = L"-0.5L", align = (:center, :bottom), fontsize = shelf_labelsize)
    text!(ax, (0.0 + z_bR) / 2, y_promoted, text = L"+0.5L", align = (:center, :bottom), fontsize = shelf_labelsize)
    text!(ax, (z_bR + z_aR) / 2, y_promoted, text = L"+\sigma_{\text{\textbf{int}}}", align = (:left, :center), rotation = pi / 2, fontsize = span_labelsize)
    text!(ax, (z_aR + z_adR) / 2, y_promoted, text = L"+l_{\text{\textbf{drift}}}", align = (:left, :center), rotation = pi / 2, fontsize = span_labelsize)
    text!(ax, (z_actL + z_aL) / 2, y_promoted, text = L"-l_{\text{\textbf{pad}}}", align = (:left, :center), rotation = pi / 2, fontsize = span_labelsize)
    text!(ax, (z_adR + z_actR) / 2, y_promoted, text = L"+l_{\text{\textbf{pad}}}", align = (:left, :center), rotation = pi / 2, fontsize = span_labelsize)

    # --- Density Curve and Laser Interfaces ---
    lines!(ax, Z_plot, rho_plot, color = :black, linewidth = 4.5)
    for pos in [z_actL, z_aL, z_bL, z_bR, z_aR, z_adR, z_actR]
        lines!(ax, [pos, pos], [y_bottom, rho_max], color = :black, linestyle = (:dash, :dense), linewidth = 3.0)
    end

    # --- Boundary Definition Legend ---
    lines!(ax, [0], [0], color = :transparent, label = L"\zeta_{\text{active}}^{-} = -\zeta_{\text{base}} + \min(l_{\text{drift}},\,0) - l_{\text{pad}}")
    lines!(ax, [0], [0], color = :transparent, label = L"\zeta_{\text{active}}^{+} = +\zeta_{\text{base}} + \max(l_{\text{drift}},\,0) + l_{\text{pad}}")
    lines!(ax, [0], [0], color = :transparent, label = L"\zeta_{\text{max},\,\text{min}} = \zeta_{\text{active}}^{\pm} \pm r_{\zeta}\,\sigma_{\text{int}}")

    axislegend(ax, position = :rt, framevisible = true, labelsize = main_leg_labelsize, rowgap = 5, padding = (0, 10, 5, 5))

    # --- Save and Feedback ---
    save(save_path, fig, px_per_unit = 4)
    println("Saved density schematic: $save_path")
    return fig
end

# Generates a zoomed profile across a shock front with 2nd-order plateau slope annotations
function plot_zoomed_snapshot(data_pack::Dict, target_tau::Float64;
                              is_gradient::Bool = true, 
                              zoom_lims::NTuple{2, Float64} = (pi/2 - 0.5, pi/2 + 0.5),
                              save_filename::String = "zoomed_snapshot.pdf")

    # --- Data Extraction and Coordinate Reconciliation ---
    ws = data_pack[:workspace]
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)

    Z_range = data_pack[:Z_grid]
    T_interval = data_pack[:T_interval]
    raw_data = data_pack[:Theta_zeta]
    p = data_pack[:selected_momentum]
    L_fs = data_pack[:selected_shelf_fs]
    λ_nm = data_pack[:pulse_params].λ_nm

    px, py, pz = round.(p; digits = 2)
    L_dim = compute_shelf_L(L_fs, λ_nm)

    num_time_slices = size(raw_data, 2)
    T_range = range(T_interval[1], T_interval[2], length = num_time_slices)
    time_idx = argmin(abs.(T_range .- target_tau))
    actual_tau = T_range[time_idx]

    # --- Frame Extraction and Unit Normalization ---
    slice_data = raw_data[:, time_idx]
    if is_gradient
        y_data = slice_data ./ 1e4
        ylabel_str = L"\Theta_\zeta(\tau, \zeta) \text{ [$10^4$ photon recoils]}"
    else
        y_integrated = zeros(length(slice_data))
        cumint_akima!(y_integrated, slice_data, Z_range)
        y_data = y_integrated ./ 1e4
        ylabel_str = L"\Theta(\tau, \zeta) \text{ [$10^4$ rad]}"
    end

    # --- Window Cropping and Extrema Detection ---
    z_min, z_max = zoom_lims
    valid_indices = findall(z -> z_min <= z <= z_max, Z_range)
    isempty(valid_indices) && error("Zoom limits $zoom_lims fall outside domain [$(Z_range[1]), $(Z_range[end])].")

    y_zoom = y_data[valid_indices]
    z_zoom = Z_range[valid_indices]
    ymin, ymax = extrema(y_zoom)
    ypad = max(0.05 * (ymax - ymin), 0.1)

    # --- Shock Boundary Detection and Tangent Fitting ---
    dz = diff(z_zoom); dy = diff(y_zoom)
    slopes = dy ./ dz
    max_slope, idx_steep = findmax(abs.(slopes))
    thresh = 0.01 * max_slope

    idx_L = idx_steep
    while idx_L > 1 && abs(slopes[idx_L - 1]) > thresh; idx_L -= 1; end
    idx_L = max(1, idx_L - 1)

    idx_R = idx_steep + 1
    while idx_R <= length(slopes) && abs(slopes[idx_R]) > thresh; idx_R += 1; end
    idx_R = min(length(y_zoom), idx_R + 1)

    if idx_L > 2
        m_left = calc_slope_2nd_order(
            z_zoom[idx_L], z_zoom[idx_L - 1], z_zoom[idx_L - 2],
            y_zoom[idx_L], y_zoom[idx_L - 1], y_zoom[idx_L - 2]
        )
    elseif idx_L > 1
        m_left = slopes[idx_L - 1]
    else
        m_left = 0.0
    end

    if idx_R < length(y_zoom) - 1
        m_right = calc_slope_2nd_order(
            z_zoom[idx_R], z_zoom[idx_R + 1], z_zoom[idx_R + 2],
            y_zoom[idx_R], y_zoom[idx_R + 1], y_zoom[idx_R + 2]
        )
    elseif idx_R <= length(slopes)
        m_right = slopes[idx_R]
    else
        m_right = 0.0
    end

    b_left  = y_zoom[idx_L] - m_left * z_zoom[idx_L]
    b_right = y_zoom[idx_R] - m_right * z_zoom[idx_R]

    # --- Canvas Layout and Style Constants ---
    fig_size          = (550, 400)
    fig_padding       = (10, 10, 5, 5)
    main_line_width   = 3.5
    dash_line_width   = 2.5
    point_marker_size = 12
    anno_fontsize     = 18

    fig = Figure(size = fig_size, figure_padding = fig_padding)
    legend_str = L"\mathbf{p}=(%$px, %$py, %$pz), \, L=%$(round(L_dim; digits=1)) (%$(round(L_fs; digits=1))\text{ fs})"
    
    ax = Axis(fig[1, 1], xlabel = L"\zeta = kz", ylabel = ylabel_str,
              xticks = LinearTicks(7), xlabelsize = 24, ylabelsize = 24,
              limits = ((z_min, z_max), (ymin - ypad, ymax + ypad)))

    # --- Visual Rendering, Tangent Plotting, and Annotations ---
    lines!(ax, Z_range, y_data, color = :dodgerblue, linewidth = main_line_width, label = legend_str)
    axislegend(ax, position = :lc, labelsize = 18, framevisible = true)

    ablines!(ax, [b_left, b_right], [m_left, m_right], color = :red, linestyle = :dash, linewidth = dash_line_width)
    scatter!(ax, [z_zoom[idx_L], z_zoom[idx_R]], [y_zoom[idx_L], y_zoom[idx_R]], color = :red, markersize = point_marker_size)

    m_left_str = format_latex_sci(m_left)
    m_right_str = format_latex_sci(m_right)
    lbl_left  = is_gradient ? L"\Theta_{\zeta\zeta} \approx %$m_left_str" : L"\Theta_{\zeta} \approx %$m_left_str"
    lbl_right = is_gradient ? L"\Theta_{\zeta\zeta} \approx %$m_right_str" : L"\Theta_{\zeta} \approx %$m_right_str"

    z_margin = 0.03 * (z_max - z_min)
    y_margin = 0.03 * (ymax - ymin)

    text!(ax, z_max - z_margin, (m_left * (z_max - z_margin) + b_left) - y_margin, 
          text = lbl_left, color = :red, align = (:right, :top), fontsize = anno_fontsize)
    text!(ax, z_min + z_margin, (m_right * (z_min + z_margin) + b_right) + y_margin, 
          text = lbl_right, color = :red, align = (:left, :bottom), fontsize = anno_fontsize)

    # --- File Export and Feedback ---
    save(save_path, fig, px_per_unit = 3)
    println("Saved zoomed snapshot (τ = $(round(actual_tau; digits=3))): $save_path")
    return fig
end

# Renders a side-by-side thermalization figure comprising a space-time heatmap and snapshot curves
function plot_thermalization_summary(data_pack::Dict, target_taus::Vector{Float64};
                                     save_filename::String = "thermalization_summary.pdf")

    # --- Data Extraction and Kinematic Invariants ---
    ws = data_pack[:workspace]
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)

    Z_range = data_pack[:Z_grid]
    T_interval = data_pack[:T_interval]
    raw_data = data_pack[:Theta_zeta] ./ 1e4
    p = data_pack[:selected_momentum]
    L_fs = data_pack[:selected_shelf_fs]
    λ_nm = data_pack[:pulse_params].λ_nm

    num_time_slices = size(raw_data, 2)
    T_range = range(T_interval[1], T_interval[2], length = num_time_slices)

    # Colorbar range extracted from global extrema
    c_max = maximum(abs, raw_data)
    y_ticks_vals = range(-c_max, c_max, length = 7)

    # Guiding-center drift velocity and trajectory bounds
    (; m, q, c, ħ) = data_pack[:units]
    (; A0, σ) = data_pack[:pulse_params]
    ω = compute_omega(λ_nm)
    k_val = ω / c
    P_vec = SVector{4, Float64}(sqrt(dot(p, p) + m*m * c*c), p[1], p[2], p[3])
    k1 = SVector(k_val, 0.0, 0.0,  k_val)
    k2 = SVector(k_val, 0.0, 0.0, -k_val)
    β1 = 2.0 * ħ * four_dot(P_vec, k1)
    β2 = 2.0 * ħ * four_dot(P_vec, k2)
    v_drift = (β2 - β1) / (β1 + β2)

    L_dim = compute_shelf_L(L_fs, λ_nm)
    px, py, pz = round.(p; digits = 3)
    L_val_fs = round(L_fs; digits = 1)
    L_val_dim = round(L_dim; digits = 1)

    # --- Extrema Preservation and Heatmap Decimation ---
    heat_Z, heat_T, heat_mat = decimate_heatmap_extrema(Z_range, T_range, raw_data, 1500, 1000)

    # --- Canvas Layout and Style Constants ---
    fig_size       = (1200, 500)
    fig_padding    = (0, 5, 2, 0)
    pub_colors     = [:limegreen, :darkorange, :purple, :red, :blue]
    axis_titlesize = 26
    label_fontsize = 26
    tick_fontsize  = 22
    anno_fontsize  = 24

    fig = Figure(size = fig_size, figure_padding = fig_padding)

    # --- Coordinate Axes Setup ---
    ax_heat = Axis(fig[1, 1], xlabel = L"\zeta = kz", ylabel = L"\tau = \omega t",
        title = "(a)", titlesize = axis_titlesize, titlefont = :regular,
        xlabelsize = label_fontsize, ylabelsize = label_fontsize, 
        xticklabelsize = tick_fontsize, yticklabelsize = tick_fontsize)

    Colorbar(fig[1, 2], colormap = Reverse(:RdBu), limits = (-c_max, c_max),
             ticks = y_ticks_vals, ticklabelsvisible = false, ticksize = 10, width = 15)

    ax_snap = Axis(fig[1, 3], xlabel = L"\zeta = kz", ylabel = L"\Theta_\zeta(\tau, \zeta) \text{ [$10^4$ photon recoils]}",
        title = "(b)", titlesize = axis_titlesize, titlefont = :regular,
        xlabelsize = label_fontsize, ylabelsize = label_fontsize, 
        xticklabelsize = tick_fontsize, yticklabelsize = tick_fontsize,
        yticks = (y_ticks_vals, [@sprintf("%.1f", v) for v in y_ticks_vals]),
        xgridvisible = true, ygridvisible = true)

    # --- Space-Time Heatmap and Drift Trajectory Rendering ---
    heatmap!(ax_heat, heat_Z, heat_T, heat_mat, colormap = Reverse(:RdBu),
             colorrange = (-c_max, c_max), rasterize = true)

    lines!(ax_heat, [0.0, 0.0], [0.0, T_interval[2]], color = (:black, 0.8),
           linewidth = 3.5, linestyle = (:dash, :dense), label = L"\text{No drift}")
    lines!(ax_heat, [0.0, v_drift * T_interval[2]], [0.0, T_interval[2]], color = :black,
           linewidth = 4.0, label = L"\text{Drift}")

    # --- Snapshot Extraction and 5-Point Envelope Detection ---
    sorted_targets = sort(target_taus)
    for (k, target_t) in enumerate(sorted_targets)
        idx = argmin(abs.(T_range .- target_t))
        actual_t = T_range[idx]
        y_slice = raw_data[:, idx]
        c_col = pub_colors[mod1(k, length(pub_colors))]
        time_str = @sprintf("%.2f", actual_t)

        hlines!(ax_heat, [actual_t], color = c_col, linewidth = 4.5, 
                linestyle = (:dot, :dense), label = L"\tau_%$k = %$time_str")
        lines!(ax_snap, Z_range, y_slice, color = (c_col, 0.4), linewidth = 2.0)

        # Robust 5-point local extrema search with backward-equality bounds for flat plateaus
        max_idx = [1]; min_idx = [1]
        for i in 3:length(y_slice)-2
            is_max = y_slice[i] >= y_slice[i-1] && y_slice[i] >= y_slice[i-2] && 
                     y_slice[i] >  y_slice[i+1] && y_slice[i] >  y_slice[i+2]
            is_min = y_slice[i] <= y_slice[i-1] && y_slice[i] <= y_slice[i-2] && 
                     y_slice[i] <  y_slice[i+1] && y_slice[i] <  y_slice[i+2]
            
            if is_max
                push!(max_idx, i)
            elseif is_min
                push!(min_idx, i)
            end
        end
        push!(max_idx, length(y_slice)); push!(min_idx, length(y_slice))

        lines!(ax_snap, Z_range[max_idx], y_slice[max_idx], color = c_col, linewidth = 2.5)
        lines!(ax_snap, Z_range[min_idx], y_slice[min_idx], color = c_col, linewidth = 2.5)
    end

    # --- Parameter Annotations, Legends, and File Export ---
    text!(ax_snap, 0.98, 0.98, text = L"\mathbf{p} = (%$px, %$py, %$pz)", space = :relative, 
          align = (:right, :top), fontsize = anno_fontsize, color = :black)
    text!(ax_snap, 0.98, 0.91, text = L"L = %$L_val_dim \, (%$L_val_fs\text{ fs})", space = :relative, 
          align = (:right, :top), fontsize = anno_fontsize, color = :black)

    ylims!(ax_snap, -c_max, c_max)
    xlims!(ax_snap, Z_range[1] + 5.0, Z_range[end] - 5.0)
    xlims!(ax_heat, Z_range[1] + 5.0, Z_range[end] - 5.0)
    axislegend(ax_heat, position = :rb, framevisible = true, labelsize = 22)

    save(save_path, fig, px_per_unit = 3)
    println("Saved thermalization summary: $save_path")
    return fig
end

# Generates publication curves of Kapitza-Dirac momentum transfer ΔP_KD vs shelf duration L
function plot_PKD(data_source::Union{Dict, AbstractString}; save_filename::String = "PKD_vs_L.pdf")
    # --- Data Extraction and Source Resolution ---
    data_filepath, ws = resolve_data_source(data_source)
    
    file = jldopen(data_filepath, "r")
    momenta = file["metadata/initial_momenta"]
    shelves_fs = file["metadata/shelf_lengths_fs"]
    pulse_params = file["metadata/pulse_params"]
    units = file["metadata/units"]

    num_p = length(momenta)
    num_L = length(shelves_fs)
    shelves_dim = [compute_shelf_L(L_fs, pulse_params.λ_nm) for L_fs in shelves_fs]

    P_KD_matrix = zeros(Float64, num_p, num_L)
    for i in 1:num_p, j in 1:num_L
        P_KD_matrix[i, j] = file["p_$i/L_$j/P_KD"] / 1e4
    end
    close(file)

    # --- Relativistic Kinematics and Theoretical Intrinsic Bound ---
    k_val = pulse_params.ω / units.c
    P_NR_2 = 2.0 * pulse_params.A0 * sqrt(2.0) / (units.ħ * k_val)
    a0 = abs(units.q * pulse_params.A0 / (units.m * units.c))
    ΔP_int_scaled = (P_NR_2 * sqrt(1.0 + a0^2 / 2.0)) / 1e4

    # --- Canvas Layout and Styling Constants ---
    fig_size     = (600, 450)
    fig_padding  = (2.5, 7.5, 2.0, 2.0)
    marker_cycle = [:rect, :circle, :diamond, :cross, :utriangle, :star5]
    colors       = Makie.wong_colors()

    fig = Figure(size = fig_size, figure_padding = fig_padding)

    # --- Coordinate Axes and Tick Formatting ---
    ax_main = Axis(fig[1, 1], xlabel = L"\text{Shelf length } L \text{ (dimensionless)}",
        ylabel = L"\Delta P_{\text{KD}}(L; \mathbf{p}) \text{ [$10^4$ photon recoils]}",
        xlabelsize = 24, ylabelsize = 26, xticklabelsize = 17, yticklabelsize = 20,
        xticks = (shelves_dim, [@sprintf("%.1f", v) for v in shelves_dim]),
        xticklabelrotation = pi / 4, xgridvisible = true, ygridvisible = true)

    ax_top = Axis(fig[1, 1], xlabel = L"\text{Shelf duration (fs)}", xaxisposition = :top,
        xlabelsize = 24, xticklabelsize = 17, xticks = (shelves_dim, [@sprintf("%.1f", v) for v in shelves_fs]),
        xticklabelrotation = pi / 4, yticksvisible = false, yticklabelsvisible = false,
        xgridvisible = false, ygridvisible = false, topspinevisible = false, rightspinevisible = false)

    # --- Curve Rendering, Theoretical Reference Line, and Export ---
    for i in 1:num_p
        px, py, pz = round.(momenta[i]; digits = 3)
        lbl = L"\mathbf{p}_{%$(i-1)} = (%$px, %$py, %$pz)"
        c_col = colors[mod1(i, length(colors))]
        m_shape = marker_cycle[mod1(i, length(marker_cycle))]

        scatterlines!(ax_main, shelves_dim, P_KD_matrix[i, :], label = lbl,
                      color = c_col, linewidth = 5.0, marker = m_shape, markersize = 20)
    end

    hlines!(ax_main, [ΔP_int_scaled], color = :red, linewidth = 4.5, label = L"\Delta P_{\text{int}} \text{ (theory)}")
    linkxaxes!(ax_main, ax_top)
    axislegend(ax_main, position = :rb, framevisible = true, labelsize = 22)

    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)
    save(save_path, fig, px_per_unit = 3)
    println("Saved PKD curve: $save_path")
    return fig
end

# Aggregates and plots multiple dataset files together onto a unified set of coordinate axes
function plot_PKD_aggregated(baseline_source::Union{Dict, AbstractString}, comparison_filepaths::Vector{String};
                             save_filename::String = "PKD_aggregated.pdf")

    # --- Source Resolution and Batch Aggregation ---
    baseline_fp, ws = resolve_data_source(baseline_source)

    function extract_file(fp)
        jldopen(fp, "r") do file
            moms = file["metadata/initial_momenta"]
            shelves_fs = file["metadata/shelf_lengths_fs"]
            p_params = file["metadata/pulse_params"]
            u_params = file["metadata/units"]

            shelves_dim = [compute_shelf_L(fs, p_params.λ_nm) for fs in shelves_fs]
            lines = []
            for i in 1:length(moms)
                pkd_vals = [file["p_$i/L_$j/P_KD"] / 1e4 for j in 1:length(shelves_fs)]
                push!(lines, (; p = moms[i], shelves_fs, shelves_dim, P_KD = pkd_vals))
            end
            return lines, p_params, u_params
        end
    end

    base_lines, pulse_params, units = extract_file(baseline_fp)
    all_lines = copy(base_lines)
    for fp in comparison_filepaths
        cmp_lines, _, _ = extract_file(fp)
        append!(all_lines, cmp_lines)
    end

    # --- Unified Dimensionless Coordinate Construction ---
    all_shelves_fs_raw = vcat([line.shelves_fs for line in all_lines]...)
    unique_fs_dict = Dict{Float64, Float64}()
    for fs in all_shelves_fs_raw
        unique_fs_dict[round(fs; digits = 4)] = fs
    end
    unified_shelves_fs = sort(collect(values(unique_fs_dict)))
    unified_dim = [compute_shelf_L(fs, pulse_params.λ_nm) for fs in unified_shelves_fs]

    # --- Relativistic Kinematics and Theoretical Intrinsic Bound ---
    k_val = pulse_params.ω / units.c
    P_NR_2 = 2.0 * pulse_params.A0 * sqrt(2.0) / (units.ħ * k_val)
    a0 = abs(units.q * pulse_params.A0 / (units.m * units.c))
    ΔP_int_scaled = (P_NR_2 * sqrt(1.0 + a0^2 / 2.0)) / 1e4

    # --- Canvas Layout and Styling Constants ---
    fig_size     = (600, 450)
    fig_padding  = (5, 10, 2.5, 2.5)
    marker_cycle = [:rect, :circle, :diamond, :cross, :utriangle, :star5]
    colors       = Makie.wong_colors()

    fig = Figure(size = fig_size, figure_padding = fig_padding)
    ax_main = Axis(fig[1, 1], xlabel = L"\text{Shelf length } L \text{ (dimensionless)}",
        ylabel = L"P_{\text{KD}}(L; \mathbf{p}) \text{ [$10^4$ photon recoils]}",
        xlabelsize = 25, ylabelsize = 26, xticklabelsize = 20, yticklabelsize = 20,
        xticks = (unified_dim, [@sprintf("%.1f", v) for v in unified_dim]),
        xticklabelrotation = pi / 4, xgridvisible = true, ygridvisible = true)

    ax_top = Axis(fig[1, 1], xlabel = L"\text{Shelf duration (fs)}", xaxisposition = :top,
        xlabelsize = 25, xticklabelsize = 20, xticks = (unified_dim, [@sprintf("%.1f", v) for v in unified_shelves_fs]),
        xticklabelrotation = pi / 4, yticksvisible = false, yticklabelsvisible = false,
        xgridvisible = false, ygridvisible = false, topspinevisible = false, rightspinevisible = false)

    # --- Curve Rendering, Reference Line, and Export ---
    hlines!(ax_main, [ΔP_int_scaled], color = :red, linewidth = 4.5, label = L"\Delta P_{\text{int}} \text{ (theory)}")

    for (i, line) in enumerate(all_lines)
        px, py, pz = round.(line.p; digits = 3)
        lbl = L"\mathbf{p}_{%$(i-1)} = (%$px, %$py, %$pz)"
        c_col = colors[mod1(i, length(colors))]
        m_shape = marker_cycle[mod1(i, length(marker_cycle))]

        scatterlines!(ax_main, line.shelves_dim, line.P_KD, label = lbl,
                      color = c_col, linewidth = 5.0, marker = m_shape, markersize = 21)
    end

    linkxaxes!(ax_main, ax_top)
    axislegend(ax_main, position = :rb, framevisible = true, labelsize = 22)

    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)
    save(save_path, fig, px_per_unit = 3)
    println("Saved aggregated PKD curve: $save_path")
    return fig
end

# Calculates and prints the spatial root-mean-square norm across time slices to verify conservation
function generate_integral_table(data_source::Union{Dict, AbstractString})
    # --- Data Extraction and Source Resolution ---
    data_filepath, _ = resolve_data_source(data_source)

    file = jldopen(data_filepath, "r")
    momenta = file["metadata/initial_momenta"]
    shelves_fs = file["metadata/shelf_lengths_fs"]

    num_p = length(momenta)
    num_L = length(shelves_fs)
    rms_I_matrix = zeros(Float64, num_p, num_L)

    # --- Parallel Sampled Quadrature Across Time Slices ---
    println("Evaluating spatial integrals across $(num_p * num_L) parameter entries in $(basename(data_filepath))...")
    @time begin
        for i in 1:num_p
            for j in 1:num_L
                grp = "p_$i/L_$j"
                grid_params = file["$grp/metadata/grid_params"]
                _, Z_grid, _, _, _ = compute_spatial_grid(grid_params)
                Theta_zeta = file["$grp/Theta_zeta"]
                num_t = size(Theta_zeta, 2)

                integrals_over_time = zeros(Float64, num_t)
                @inbounds @batch for t in 1:num_t
                    prob = SampledIntegralProblem(view(Theta_zeta, :, t), Z_grid)
                    integrals_over_time[t] = solve(prob, TrapezoidalRule()).u
                end
                rms_I_matrix[i, j] = sqrt(mean(abs2, integrals_over_time))
            end
        end
    end
    close(file)

    # --- Diagnostic Table Output Formatting ---
    println("\n" * "="^80)
    println("ROOT MEAN SQUARE (RMS) [10^-4] OF TOTAL INTEGRAL OVER TIME")
    println("="^80)
    print(rpad("Momentum (px, py, pz)", 28))
    for L in shelves_fs
        print(lpad(@sprintf("%.1f fs", L), 15))
    end
    println("\n" * "-"^80)
    for i in 1:num_p
        px, py, pz = round.(momenta[i]; digits = 3)
        print(rpad("($px, $py, $pz)", 28))
        for j in 1:num_L
            print(lpad(@sprintf("%.2f", rms_I_matrix[i, j] * 1e4), 15))
        end
        println()
    end
    println("="^80 * "\n")
end


# ============================================================================== #
# SIMULATION CONFIGURATION AND EXECUTION
# tag_config_and_execution
# ============================================================================== #

# --- Configuration ---
work_directory = raw"C:\Users\PC\Desktop\PRA_submission_scripts"
workspace_name = :Workspace_Main

run_config = (
    # Identifier (String, Symbol, or nothing / :_ for auto-incremented Run_i name)
    run_name            = :_,
    
    # Particle properties and optical pulse parameters
    units               = PhysicalUnits(m = 1.0, q = -1.0, c = 137.036, ħ = 1.0),
    pulse_params        = PulseParameters(A0 = 13.0, λ_nm = 800.0, σ = 10.0),
    
    # Batch parameter permutations
    initial_momenta     = [
        SVector(0.0, 0.0, 0.0),
        SVector(0.128, 0.343, 3.996)
    ],
    shelf_durations_fs  = [10.0, 15.0, 25.0],
    
    # Spatial grid resolution bounds: (dZ_min in refined shelf, dZ_max in asymptotic margins)
    spatial_resolutions = (7.5e-4, 1e-2),
    N_time_steps        = 300,
    
    # Simulation finish time specification:
    # 1. Functional expression: (L, σ) -> 0.5 * L + 3.0 * σ + 1.0
    # 2. Vector of times: [20.0, nothing, 45.0] (empty entries/shorter list default to symmetric |T_min|)
    # 3. nothing: all shelves terminate symmetrically at |T_min|
    T_max               = (L, σ) -> 0.5 * L + 3.0 * σ + 1.0,
    
    # Diagnostic recording sub-window (set to nothing to record full duration [T_min, T_max])
    T_i                 = nothing,
    T_f                 = nothing,
    
    # Root workspace directory and workspace folder
    work_directory      = work_directory,
    workspace_name      = workspace_name
)

# Diagnostic interaction timestamps for thermalization snapshot slices
target_thermalization_taus = [7.64, 23.45, 45.88]

# Snapshot interaction timestamp for zoomed shock profile
target_shock_tau = -1.72


# --- Execution ---
# - Run batch simulation and serialize into <workspace>/Data/
# run_and_save(run_config)


# --- Workflow ---
# - Load data for a specific momentum/shelf pair from a file into REPL:
# loaded_data = interactive_loader(work_directory, workspace_name)

# - Render evolution animation:
# -- Minimal call; defaults: zeta_lims = (-10, 10), time_window = (-Inf, Inf), fps = 30):
# animate_results(loaded_data, :Theta_zeta)
#
# -- Explicit keyword call:
# animate_results(loaded_data, :Theta;
#                 zeta_lims     = (-15.0, 15.0),
#                 time_window   = (-10.0, 40.0),
#                 fps           = 30,
#                 save_filename = "phase_evolution.mp4")


# - Generate non-uniform grid density schematic:
# -- Minimal call (saves to <workspace>/Plots/Node_Density_Schematic.pdf):
# generate_density_plot(loaded_data)
#
# -- Custom output file name call:
# generate_density_plot(loaded_data, "Custom_Grid_Schematic.pdf")


# - Generate zoomed snapshot across shock front:
# -- Minimal call; defaults: is_gradient = true, zoom_lims = (π/2 - 0.5, π/2 + 0.5)):
# plot_zoomed_snapshot(loaded_data, target_shock_tau)
#
# -- Explicit keyword call:
# plot_zoomed_snapshot(loaded_data, target_shock_tau;
#                      is_gradient   = true,
#                      zoom_lims     = (pi/2 - 0.5, pi/2 + 0.5),
#                      save_filename = "zoomed_shock_gradient.pdf")


# - Generate thermalization summary (space-time heatmap + snapshot slices):
# -- Minimal call (saves to <workspace>/Plots/thermalization_summary.pdf):
# plot_thermalization_summary(loaded_data, target_thermalization_taus)
#
# -- Explicit keyword call:
# plot_thermalization_summary(loaded_data, target_thermalization_taus;
#                             save_filename = "thermalization_run1.pdf")


# - Generate momentum spread/transfer plot (ΔP_KD vs L):
# -- Minimal call (saves to <workspace>/Plots/PKD_vs_L.pdf):
# plot_PKD(loaded_data)
#
# -- Custom output file name call:
# plot_PKD(loaded_data; save_filename = "PKD_vs_L_run1.pdf")
#
# -- Direct file path call (without interactive loader):
# run_file = joinpath(work_directory, string(workspace_name), "Data", "Run_name.jld2")
# plot_PKD(run_file; save_filename = "PKD_vs_L_direct.pdf")


# - Print spatial integral table to REPL:
# -- Minimal call:
# generate_integral_table(loaded_data)
#
# -- Direct file path call:
# run_file = joinpath(work_directory, string(workspace_name), "Data", "Run_1.jld2")
# generate_integral_table(run_file)


# - Aggregate multiple runs into a unified P_KD comparison figure:
# baseline_run = joinpath(work_directory, string(workspace_name), "Data", "Run_1.jld2")
# additional_runs = [
#     joinpath(work_directory, string(workspace_name), "Data", "Run_2.jld2"),
#     joinpath(work_directory, string(workspace_name), "Data", "Run_3.jld2")
# ]
# plot_PKD_aggregated(baseline_run, additional_runs; save_filename = "PKD_comparison_aggregated.pdf")