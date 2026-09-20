using LinearAlgebra, StaticArrays, SparseArrays
using Parameters
using CairoMakie, LaTeXStrings, Printf
using SpecialFunctions
using OrdinaryDiffEq, OrdinaryDiffEqSSPRK
using Polyester
using JLD2
using DataInterpolations

# ============================================================================== #
# TABLE OF CONTENTS
# - PARAMETERS, DATA STRUCTURES, AND RECONSTRUCTION  : tag_params_structs_recon
# - SPATIAL DISCRETIZATION AND PDE SOLVER            : tag_grid_and_solver
# - DATA MANAGEMENT AND SERIALIZATION                : tag_data_and_serialization
# - VISUALIZATION AND PLOTTING PIPELINE              : tag_visualization
# - SIMULATION CONFIGURATION AND EXECUTION           : tag_config_and_execution
# ============================================================================== #


# ============================================================================== #
# PARAMETERS, DATA STRUCTURES, AND RECONSTRUCTION
# tag_params_structs_recon
# ============================================================================== #

# Physical unit conversion factors:
# - OMEGA_CONVERSION: converts wavelength in nm to angular frequency ω in atomic units
# - SHELF_CONVERSION: converts pulse flat-top shelf duration in fs to shelf length L
const OMEGA_CONVERSION = 45.56335252912
const SHELF_CONVERSION = 1883.651567309

# --- Data Structures ---

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

struct PulseParameters
    A0::Float64
    λ_nm::Float64
    σ::Float64
    shelf_duration_fs::Float64

    function PulseParameters(A0::Real, λ_nm::Real, σ::Real, shelf_duration_fs::Real)
        A0 > 0                 || throw(ArgumentError("Field amplitude A0 must be strictly positive (got $A0)"))
        λ_nm > 0               || throw(ArgumentError("Laser wavelength λ_nm must be strictly positive (got $λ_nm)"))
        σ > 0                  || throw(ArgumentError("Pulse rise parameter σ must be strictly positive (got $σ)"))
        shelf_duration_fs >= 0 || throw(ArgumentError("Shelf duration must be non-negative (got $shelf_duration_fs)"))
        return new(Float64(A0), Float64(λ_nm), Float64(σ), Float64(shelf_duration_fs))
    end
end
PulseParameters(; A0, λ_nm, σ, shelf_duration_fs) = PulseParameters(A0, λ_nm, σ, shelf_duration_fs)
PulseParameters(nt::NamedTuple) = PulseParameters(; nt...)

# Spatial and temporal simulation domain boundaries
struct Domain
    T_sim::NTuple{2, Float64}     # Total integration interval: (T_min, T_max)
    T_save::NTuple{2, Float64}    # Diagnostic recording window: (T_i, T_f)
    dZ_steps::NTuple{2, Float64}  # Spatial resolution bounds: (dZ_min, dZ_max)
    N_time_steps::Int             # Number of recorded time slices
end

# --- Helper Functions ---

@inline compute_omega(λ_nm::Real) = OMEGA_CONVERSION / Float64(λ_nm)
@inline compute_omega(p::PulseParameters) = compute_omega(p.λ_nm)

@inline compute_shelf_L(shelf_duration_fs::Real, λ_nm::Real) = SHELF_CONVERSION * Float64(shelf_duration_fs) / Float64(λ_nm)
@inline compute_shelf_L(p::PulseParameters) = compute_shelf_L(p.shelf_duration_fs, p.λ_nm)

# Minkowski inner product under signature (+, -, -, -)
@inline function four_dot(a::SVector{4, Float64}, b::SVector{4, Float64})
    return a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4]
end

# Modulated field amplitude by flat-shelf pulse with half-Gaussian rising and falling edges
@inline function field_amplitude(φ, A0, inv_m2σ2, L)
    s = max(0.0, abs(φ) - 0.5 * L)
    return A0 * cos(φ) * exp(s * s * inv_m2σ2)
end

# Volkov phase derivative w.r.t. φ_i, i = 1,2
@inline function dSi(qa, m_inv_β, pϵ)
    return muladd(2.0, pϵ, qa) * qa * m_inv_β
end

# --- High-Order Numerical Reconstruction and Integration ---

# Fifth-order WENO-Z reconstruction on a uniform five-point stencil
# Evaluates left-biased interface value from candidate stencils (S0, S1, S2)
# using Jiang-Shu smoothness indicators and Borges weights
function WENO5_Z(fm2, fm1, fc0, fp1, fp2)
    # Stencil candidate polynomials
    S0 =  2.0 * fm2 - 7.0 * fm1 + 11.0 * fc0
    S1 =  5.0 * fc0 + 2.0 * fp1 -  1.0 * fm1  
    S2 =  2.0 * fc0 + 5.0 * fp1 -  1.0 * fp2

    # Jiang-Shu smoothness indicators
    coeff = 13.0 / 12.0
    β0 = coeff * (fm2 - 2.0*fm1 + fc0)^2 + 0.25 * (fm2 - 4.0*fm1 + 3.0*fc0)^2
    β1 = coeff * (fm1 - 2.0*fc0 + fp1)^2 + 0.25 * (fm1 - fp1)^2
    β2 = coeff * (fc0 - 2.0*fp1 + fp2)^2 + 0.25 * (3.0*fc0 - 4.0*fp1 + fp2)^2

    # WENO-Z unnormalized weights
    τ5 = abs(β0 - β2)
    α0 = 0.1 * (1.0 + (τ5 / (β0 + 1e-40))^2)
    α1 = 0.6 * (1.0 + (τ5 / (β1 + 1e-40))^2)
    α2 = 0.3 * (1.0 + (τ5 / (β2 + 1e-40))^2)

    return (α0*S0 + α1*S1 + α2*S2) / (6.0 * (α0 + α1 + α2))
end

# Cumulative integration of Akima spline interpolation
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

    # Effective interaction region boundaries accounting for drift
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

        # Algebraic solution within the linear shelf
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

    # Precomputed inverse non-uniform grid steps
    inv_dZ_nodes = 1.0 ./ diff(Z_ext[2:N+5])
    inv_dZ_cells = 1.0 ./ diff(Z_inter)     

    return N, Z_grid, Z_inter, inv_dZ_nodes, inv_dZ_cells
end

# --- Parameter and Cache Bundler ---

# Assembles pre-allocated caches for the extended data arrays and numerical flux values,
# non-uniform spatial grid elements and physical constant into named tuples passed to the ODE integrator
function nonuniform_grid_params(constants, grid_params)
    @unpack q, A0, α, β1, β2, pϵ1, pϵ2 = constants
    @unpack σ, L = grid_params

    N, _, Z_inter, inv_dZ_nodes, inv_dZ_cells = compute_spatial_grid(grid_params)

    caches = (;
        H_flux = zeros(ComplexF64, N + 1),
        ΛZ_ext = zeros(Float64, N + 6),
        ΘZ_ext = zeros(Float64, N + 6)
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

# Used to solve the hyperbolic conservation law for the complex u_ζ = Λ_ζ + i Θ_ζ
function pde_system!(du, u, params, T)
    (; caches, grid, phys) = params
    (; H_flux, ΛZ_ext, ΘZ_ext) = caches
    (; N, Z_inter, inv_dZ_nodes, inv_dZ_cells) = grid
    (; q, A0, inv_m2σ2, L, α, β_p, β_m, m_inv_β1, m_inv_β2, pϵ1, pϵ2) = phys

    # --- State Unpacking and Ghost Padding ---
    # Unpack state vector u into padded arrays to preserve cache locality across workers
    @inbounds @batch for i in 1:N
        ΛZ_ext[i+3] = u[i]
        ΘZ_ext[i+3] = u[N+i]
    end
    
    # Constant zeroth-order extrapolation across ghost boundaries
    ΛZ_1 = u[1];   ΛZ_N = u[N]
    ΛZ_ext[1]   = ΛZ_1; ΛZ_ext[2]   = ΛZ_1; ΛZ_ext[3]   = ΛZ_1
    ΛZ_ext[N+4] = ΛZ_N; ΛZ_ext[N+5] = ΛZ_N; ΛZ_ext[N+6] = ΛZ_N

    ΘZ_1 = u[N+1]; ΘZ_N = u[2N]
    ΘZ_ext[1]   = ΘZ_1; ΘZ_ext[2]   = ΘZ_1; ΘZ_ext[3]   = ΘZ_1
    ΘZ_ext[N+4] = ΘZ_N; ΘZ_ext[N+5] = ΘZ_N; ΘZ_ext[N+6] = ΘZ_N

    # --- Interface Flux Evaluation and LLF Calculation ---
    @inbounds @batch for i in eachindex(H_flux)
        Z = Z_inter[i]

        # Fifth-order WENO-Z reconstruction to cell interface from left and right
        ΛZ_L = WENO5_Z(ΛZ_ext[i],   ΛZ_ext[i+1], ΛZ_ext[i+2], ΛZ_ext[i+3], ΛZ_ext[i+4])
        ΛZ_R = WENO5_Z(ΛZ_ext[i+5], ΛZ_ext[i+4], ΛZ_ext[i+3], ΛZ_ext[i+2], ΛZ_ext[i+1])

        ΘZ_L = WENO5_Z(ΘZ_ext[i],   ΘZ_ext[i+1], ΘZ_ext[i+2], ΘZ_ext[i+3], ΘZ_ext[i+4])
        ΘZ_R = WENO5_Z(ΘZ_ext[i+5], ΘZ_ext[i+4], ΘZ_ext[i+3], ΘZ_ext[i+2], ΘZ_ext[i+1])

        v_L = complex(ΛZ_L, ΘZ_L)
        v_R = complex(ΛZ_R, ΘZ_R)
    
        # Second order spatial derivative approximation at the grid:
        # --- Finite difference across the cell
        Λ_zz = (ΛZ_ext[i+3] - ΛZ_ext[i+2]) * inv_dZ_nodes[i+1]
        Θ_zz = (ΘZ_ext[i+3] - ΘZ_ext[i+2]) * inv_dZ_nodes[i+1]
        v_zz = complex(Λ_zz, Θ_zz)
        
        # Local field amplitudes and Volkov phase derivatives
        qa1 = q * field_amplitude(T - Z, A0, inv_m2σ2, L)
        qa2 = q * field_amplitude(T + Z, A0, inv_m2σ2, L)
        dS1 = dSi(qa1, m_inv_β1, pϵ1)
        dS2 = dSi(qa2, m_inv_β2, pϵ2)

        # Interaction potential terms and light-cone projections
        W_term = 4.0 * (2.0 * qa1 * qa2 - α * dS1 * dS2)
        B_m = muladd(-α, dS1 - dS2, β_m)
        B_p = muladd(-α, dS1 + dS2, β_p)
        B_p2 = B_p * B_p

        im_2B_m = complex(0.0, 2.0 * B_m)
    
        # Complex Hamilton-Jacobi flux candidates
        C_L = α * (v_L*v_L + v_zz) + im_2B_m * v_L - W_term
        C_R = α * (v_R*v_R + v_zz) + im_2B_m * v_R - W_term

        S_L = sign(B_p) * sqrt(B_p2 - α * C_L)
        S_R = sign(B_p) * sqrt(B_p2 - α * C_R)

        H_L = -im * C_L / (B_p + S_L)
        H_R = -im * C_R / (B_p + S_R)

        # Local characteristic wave speed for Lax-Friedrichs numerical dissipation
        dHdv_L = (-im * α * v_L + B_m) / S_L
        dHdv_R = (-im * α * v_R + B_m) / S_R
        c_max = min(max(abs(dHdv_L), abs(dHdv_R)), 1.0)

        # Local Lax-Friedrichs (LLF) numerical flux
        H_flux[i] = 0.5 * (H_L + H_R - c_max * (v_R - v_L))
    end

    # --- Conservative Flux Divergence ---
    # Evaluates spatial derivative of numerical flux across each control volume
    @inbounds @batch for i in 1:N
        m_dH_dZ = (H_flux[i] - H_flux[i+1]) * inv_dZ_cells[i]
        du[i], du[N+i] = reim(m_dH_dZ)
    end
end 

# --- Time-Integration Driver ---

# Configures initial kinematic states, sets up non-uniform coordinates, and integrates
# the PDE system using a third-order strong-stability-preserving Runge-Kutta solver.
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

    # Constants
    k = ω / c; k1 = k * n1; k2 = k * n2
    ħpk1 = ħ * four_dot(P, k1); ħpk2 = ħ * four_dot(P, k2)
    pϵ1 = four_dot(P, ϵ1); pϵ2 = four_dot(P, ϵ2)

    α = 2.0 * ħ * ħ * four_dot(k1, k2)
    β1 = 2.0 * ħpk1; β2 = 2.0 * ħpk2

    # Guiding-center drift compensation
    v_drift = (β2 - β1) / (β1 + β2)
    L_drift = v_drift * (T_max - T_min)
    
    Z_base = 0.5 * L + 5.0 * σ / sqrt(2.0) + 1.0
    Z_min = -Z_base + min(L_drift, 0.0)
    Z_max =  Z_base + max(L_drift, 0.0)

    constants = (; q, A0, α, β1, β2, pϵ1, pϵ2)
    grid_params = (; Z_min, Z_max, dZ_min, dZ_max, L_drift, σ, L, shelf_pad = 1.0)

    pde_params = nonuniform_grid_params(constants, grid_params)
    u0 = zeros(2 * pde_params.grid.N)
    
    T_range = range(T_i, T_f, length = N_time_steps + 1)
    prob = ODEProblem(pde_system!, u0, (T_min, T_max), pde_params)
    sol = solve(prob, SSPRK43(), abstol = 1e-12, reltol = 1e-4, 
                saveat = T_range, save_everystep = false, save_start = false)

    # Prune stray pre-recorded endpoint if solver prepends T_min
    u_saved = (length(sol.u) > length(T_range) && sol.t[1] < T_range[1] - 1e-10) ? sol.u[2:end] : sol.u

    full_matrix = reduce(hcat, u_saved)
    ΛZ_array = full_matrix[1:pde_params.grid.N, :]
    ΘZ_array = full_matrix[pde_params.grid.N+1:end, :]
    
    return grid_params, T_range, ΛZ_array, ΘZ_array
end


# ============================================================================== #
# DATA MANAGEMENT AND SERIALIZATION
# tag_data_and_serialization
#
# Functions in this section:
# - get_workspace_paths
# - resolve_run_filename
# - resolve_run_filenames
# - run_and_save
# - interactive_loader
# ============================================================================== #

# --- Directory Scaffold ---

# Constructs the fixed internal directories for datasets, visual figures, and animations
function get_workspace_paths(base_dir::String, workspace_name::Union{AbstractString, Symbol} = "Workspace_Auxiliary")
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
    
    # Collect existing numerical indices from disk or batch reservation sets
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

    # Format base run identifier
    is_blank(x) = isnothing(x) || x === missing || x === :_ || x === "_" || (x isa AbstractString && isempty(strip(x)))
    if is_blank(name_spec)
        run_name = "Run_$(next_auto_idx)"
    else
        raw_str = string(name_spec)
        run_name = startswith(raw_str, "Run_") ? raw_str : "Run_$(raw_str)"
    end

    # Increment suffix to avoid overwriting existing data files
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

# --- Simulation Driver and Data Serializer ---

# Dispatches the numerical solver and serializes coordinate grids, state fields,
# and physical metadata into a compressed HDF5/JLD2 container.
function run_and_save(cfg::NamedTuple)
    @unpack units, pulse_params, initial_momentum, spatial_resolutions, N_time_steps, 
            T_max, T_i, T_f, work_directory = cfg

    ws_name = get(cfg, :workspace_name, "Workspace_Auxiliary")
    ws = get_workspace_paths(work_directory, ws_name)
    req_name = get(cfg, :run_name, nothing)
    run_name, file_path = resolve_run_filename(ws.data_dir, req_name)
    println("Initializing run: $run_name (Workspace: $ws_name)")

    (; m, q, c, ħ) = units
    (; A0, σ) = pulse_params
    ω = compute_omega(pulse_params)
    L = compute_shelf_L(pulse_params)

    # Total integration boundaries and recording sub-window limits
    T_min = -(0.5 * L + 5.0 * σ + 1.0)
    T_max_sim = isnothing(T_max) ? abs(T_min) : T_max

    raw_Ti = isnothing(T_i) ? T_min : T_i
    raw_Tf = isnothing(T_f) ? T_max_sim : T_f
    T_i_clamped = clamp(min(raw_Ti, raw_Tf), T_min, T_max_sim)
    T_f_clamped = clamp(max(raw_Ti, raw_Tf), T_min, T_max_sim)

    # Initial laser geometry vectors and electron on-shell four-momentum
    ϵ1 = SVector(0.0, 1.0, 0.0,  0.0); ϵ2 = SVector(0.0, 1.0, 0.0,  0.0)
    n1 = SVector(1.0, 0.0, 0.0,  1.0); n2 = SVector(1.0, 0.0, 0.0, -1.0)
    p  = initial_momentum
    P  = SVector{4, Float64}(sqrt(dot(p, p) + m*m * c*c), p[1], p[2], p[3])

    domain = Domain((T_min, T_max_sim), (T_i_clamped, T_f_clamped), spatial_resolutions, N_time_steps)

    println("Simulating: τ ∈ [$T_min, $T_max_sim] | Recording: τ ∈ [$T_i_clamped, $T_f_clamped] ($(N_time_steps + 1) steps)")
    @time grid_params, T_range, ΛZ_array, ΘZ_array = run_solver(P, domain, units, pulse_params, (ϵ1, ϵ2), (n1, n2))

    # Serialize dataset and simulation metadata to disk
    jldopen(file_path, "w"; compress = true) do file
        file["metadata/units"]               = (m = units.m, q = units.q, c = units.c, ħ = units.ħ)
        file["metadata/pulse_params"]        = (A0 = pulse_params.A0, λ_nm = pulse_params.λ_nm, σ = pulse_params.σ, 
                                                shelf_duration_fs = pulse_params.shelf_duration_fs, ω = ω, L = L)
        file["metadata/polarizations"]       = (ϵ1, ϵ2)
        file["metadata/propagations"]        = (n1, n2)
        file["metadata/initial_momentum"]    = initial_momentum
        file["metadata/L_shelf_fs"]          = pulse_params.shelf_duration_fs
        file["metadata/spatial_separations"] = spatial_resolutions
        file["metadata/N_time_steps"]        = N_time_steps
        file["metadata/T_sim"]               = (T_min, T_max_sim)
        file["metadata/T_interval"]          = (T_range[1], T_range[end])
        file["metadata/grid_params"]         = grid_params
        file["metadata/work_directory"]      = work_directory
        file["metadata/workspace_name"]      = string(ws_name)
        
        file["Lambda_zeta"] = Float32.(ΛZ_array)
        file["Theta_zeta"]  = Float32.(ΘZ_array)
    end
    println("Run complete. Saved: $file_path")
end

# --- Interactive REPL Dataset Loader ---

# Lists all run archives inside a designated workspace and loads fields into memory.
function interactive_loader(base_dir::String, workspace_name::Union{AbstractString, Symbol} = "Workspace_Auxiliary")
    ws = get_workspace_paths(base_dir, workspace_name)
    files = filter(f -> startswith(f, "Run_") && endswith(f, ".jld2"), readdir(ws.data_dir))
    
    if isempty(files)
        return println("No Auxiliary Run files found in $(ws.data_dir).")
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
    println("\nLoading data from $(files[run_choice])...")
    
    T_bounds = file["metadata/T_interval"]
    grid_params = file["metadata/grid_params"]
    _, Z_grid, _, _, _ = compute_spatial_grid(grid_params)

    data_pack = Dict{Symbol, Any}(
        :units              => file["metadata/units"],
        :pulse_params       => file["metadata/pulse_params"],
        :selected_momentum  => file["metadata/initial_momentum"],
        :selected_shelf_fs  => file["metadata/L_shelf_fs"],
        :T_interval         => T_bounds,
        :N_time_steps       => file["metadata/N_time_steps"],
        :Z_grid             => Z_grid,  
        :Lambda_zeta        => file["Lambda_zeta"],
        :Theta_zeta         => file["Theta_zeta"],
        :workspace          => ws
    )
    
    close(file)
    println("Load successful. Data returned to REPL.")
    return data_pack
end


# ============================================================================== #
# VISUALIZATION AND PLOTTING PIPELINE
# tag_visualization
#
# Functions in this section:
# - animate_results
# - plot_Lambda_snapshots
# ============================================================================== #

# --- Evolution Animation Generator ---

# Produces a video of the spatial field evolution over a chosen time interval.
function animate_results(data_pack::Dict, variable::Symbol;
                         zeta_lims::Tuple{Float64, Float64} = (-10.0, 10.0),
                         time_window::Tuple{Float64, Float64} = (-Inf, Inf),
                         fps::Int = 30,
                         save_filename::String = "auxiliary_evolution.mp4")

    ws = data_pack[:workspace]
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.anims_dir, save_filename)

    # --- Temporal Indexing and Validation ---
    x_grid = data_pack[:Z_grid]
    T_interval = data_pack[:T_interval]
    N_steps = data_pack[:N_time_steps]
    px, py, pz = round.(data_pack[:selected_momentum]; digits = 3)
    
    T_range = range(T_interval[1], T_interval[2], length = N_steps + 1)
    valid_indices = findall(t -> time_window[1] <= t <= time_window[2], T_range)
    
    if isempty(valid_indices)
        return println("Warning: No frames found within time window $(time_window).")
    end

    # --- Parallel Field Integration ---
    is_gradient = (variable === :Theta_zeta || variable === :Lambda_zeta)
    raw_data_mat = (variable === :Theta_zeta || variable === :Theta) ? data_pack[:Theta_zeta] : data_pack[:Lambda_zeta]
    y_data_mat = Matrix{Float64}(undef, size(raw_data_mat, 1), length(valid_indices))

    if is_gradient
        y_data_mat .= view(raw_data_mat, :, valid_indices)
    else
        println("Integrating field snapshots in parallel...")

        @inbounds @batch for idx in eachindex(valid_indices)
            matrix_idx = valid_indices[idx]
            cumint_akima!(y_data_mat, view(raw_data_mat, :, matrix_idx), x_grid; col = idx)
        end
    end

    # --- Canvas Setup and Auto-Scaling ---
    data_min, data_max = minimum(y_data_mat), maximum(y_data_mat)
    rel_pad = 0.01 * abs(data_max - data_min)
    lims_min, lims_max = zeta_lims
    lim_pad = 0.01 * abs(lims_max - lims_min)

    var_map = Dict(:Theta_zeta => L"\Theta_\zeta", :Lambda_zeta => L"\Lambda_\zeta", :Theta => L"\Theta", :Lambda => L"\Lambda")
    title_str = L"%$(var_map[variable])(\tau, \zeta) \text{ for } \mathbf{p} = (%$px, %$py, %$pz)"

    fig = Figure(size = (800, 600), dpi = 400)
    ax = Axis(fig[1, 1], title = title_str, xlabel = L"\zeta", ylabel = "",
        limits = ((lims_min - lim_pad, lims_max + lim_pad), (data_min - rel_pad, data_max + rel_pad)),
        yticks = range(data_min, data_max, length = 7),
        ytickformat = values -> [@sprintf("%.2e", v) for v in values])

    # --- Animation Rendering Pipeline ---
    time_index = Observable(1)
    pos_y = @lift(view(y_data_mat, :, $time_index))
    lines!(ax, x_grid, pos_y, color = :firebrick)

    text_label = @lift("T: $(@sprintf("%.3f", T_range[valid_indices[$time_index]]))")
    text!(ax, 0.05, 0.9, text = text_label, space = :relative, fontsize = 16)

    println("Rendering $(length(valid_indices)) frames at $fps FPS...")
    record(fig, save_path, 1:length(valid_indices); framerate = fps) do step_idx
        time_index[] = step_idx
    end

    println("Saved animation: $save_path")

    return nothing
end

# --- Snapshot Visualizer ---

# Renders multi-snapshot profiles of the integrated log-amplitude field Λ(τ, ζ)
# and generates a zoomed inset highlighting the left central caustic spike.
function plot_Lambda_snapshots(data_pack::Dict, target_times::Vector{Float64};
                               zeta_lims::Tuple{Float64, Float64} = (-10.5, 10.5), 
                               n_peak_pairs::Int = 1,
                               save_filename::String = "Lambda_snapshots.pdf")

    # --- Initialization and Target Times Selection ---
    ws = data_pack[:workspace]
    save_path = isabspath(save_filename) ? save_filename : joinpath(ws.plots_dir, save_filename)

    x_grid     = data_pack[:Z_grid]
    T_interval = data_pack[:T_interval]
    N_steps    = data_pack[:N_time_steps]
    L_fs       = data_pack[:selected_shelf_fs]
    λ_nm       = data_pack[:pulse_params].λ_nm
    L          = compute_shelf_L(L_fs, λ_nm)

    T_range   = range(T_interval[1], T_interval[2], length = N_steps + 1)
    T_targets = sort(target_times, rev = true) 
    
    valid_indices = Int[]
    actual_times  = Float64[]
    for t in T_targets
        if t < T_interval[1] || t > T_interval[2]
            @warn "Requested time $t outside simulation interval $(T_interval)."
        end
        idx = argmin(abs.(T_range .- t))
        push!(valid_indices, idx)
        push!(actual_times, T_range[idx])
    end

    # --- Parallel Field Integration and Recorded Gradients ---
    raw_data_mat     = data_pack[:Lambda_zeta]
    integrated_snaps = [similar(raw_data_mat[:, 1]) for _ in 1:length(valid_indices)]
    
    @batch for k in eachindex(valid_indices)
        step_idx = valid_indices[k]
        cumint_akima!(integrated_snaps[k], view(raw_data_mat, :, step_idx), x_grid)
    end

    # Maximum gradient amplitudes across selected snapshots and the entire simulation run
    peak_grad_snaps  = maximum(abs, @view raw_data_mat[:, valid_indices])
    peak_grad_global = maximum(abs, raw_data_mat)

    # --- Peak Detection and Inset Geometry ---
    main_pad           = 0.01   # Main plot Y-axis baseline margin factor
    main_headroom_mult = 5.0    # Main plot Y-axis headroom multiplier (headroom_pad = headroom_mult * pad)

    inset_node_radius  = 16.0   # Inset half-width in local grid intervals
    inset_pad_rel      = 0.05   # Inset Y-axis padding factor

    # Locates the central pair of peaks bounding the interaction center.
    # Candidate local maxima are first ranked by amplitude to filter out flanking shoulders.
    # The top 2*n candidates are then sorted spatially, with the center defined by their
    # amplitude-weighted position.
    function locate_central_peaks(x_coords, y_vals, n_pairs::Int)
        peak_indices = Int[]
        for i in 2:(length(y_vals) - 1)
            if y_vals[i] > y_vals[i - 1] && y_vals[i] > y_vals[i + 1]
                push!(peak_indices, i)
            end
        end

        if length(peak_indices) < 2
            mid = div(length(y_vals), 2)
            return max(1, mid - 1), min(length(y_vals), mid + 1), x_coords[mid]
        end

        # Rank detected peaks descending by absolute amplitude
        sort!(peak_indices, by = i -> abs(y_vals[i]), rev = true)

        # Select the top 2*n dominant peaks
        k_count = min(length(peak_indices), 2 * n_pairs)
        top_indices = peak_indices[1:k_count]

        # Compute center coordinate as the amplitude-weighted sum of positions
        weights = [abs(y_vals[i]) for i in top_indices]
        sum_weights = sum(weights)
        center_coord = sum(x_coords[i] * w for (i, w) in zip(top_indices, weights)) / sum_weights

        # Order selected peaks by spatial coordinate
        sort!(top_indices, by = i -> x_coords[i])

        # Identify the innermost pair flanking the weighted center
        left_cands  = filter(i -> x_coords[i] <= center_coord, top_indices)
        right_cands = filter(i -> x_coords[i] > center_coord, top_indices)

        idx_left  = isempty(left_cands) ? top_indices[1] : left_cands[end]
        idx_right = isempty(right_cands) ? top_indices[end] : right_cands[1]

        return idx_left, idx_right, center_coord
    end

    latest_snap  = integrated_snaps[1]
    y_lo, y_hi   = extrema(latest_snap)
    y_pad        = main_pad * (y_hi - y_lo)

    # Detect the central peak pair for the reference snapshot using the weighted center
    peak_idx_L, _, _ = locate_central_peaks(x_grid, latest_snap, n_peak_pairs)
    peak_x = x_grid[peak_idx_L]
    peak_y = latest_snap[peak_idx_L]
    
    # Local node spacing inside the uniform refinement shelf
    dx_local     = 0.5 * (x_grid[peak_idx_L + 1] - x_grid[peak_idx_L - 1])
    window_width = inset_node_radius * dx_local
    in_xmin, in_xmax = peak_x - window_width, peak_x + window_width

    # Vertical framing of the inset viewport
    window_mask   = abs.(x_grid .- peak_x) .<= window_width
    min_y_window  = minimum(@view latest_snap[window_mask])
    window_height = peak_y - min_y_window

    y_window_pad  = inset_pad_rel * window_height
    in_ymin       = min_y_window - y_window_pad
    in_ymax       = peak_y + y_window_pad

    # --- Canvas Layout and Coordinate Axes ---
    fig_size       = (800, 600)
    fig_padding    = (10, 10, 5, 5)
    line_colors    = [:dodgerblue, :red, :limegreen, :darkorange, :purple, :black]

    main_labelsize      = 30
    main_ticklabelsize  = 25

    inset_ticklabelsize = 17
    inset_layer         = 150

    ylabel_str = L"\Lambda(\tau, \zeta); \text{$\mathbf{p}$}=\mathbf{0}, L=%$(round(L; digits=1)) (%$(round(L_fs; digits=1))\text{ fs})"
    fig = Figure(size = fig_size, figure_padding = fig_padding)

    # Primary coordinate axis
    ax_main = Axis(fig[1, 1], xlabel = L"\zeta = kz", xlabelsize = main_labelsize, ylabel = ylabel_str, ylabelsize = main_labelsize,          
        xticklabelsize = main_ticklabelsize, yticklabelsize = main_ticklabelsize, 
        limits = (zeta_lims, (y_lo - y_pad, y_hi + main_headroom_mult * y_pad)), 
        xgridvisible = true, ygridvisible = true, ytickformat = values -> [@sprintf("%.1f", v) for v in values])

    # Inset axis
    ax_inset = Axis(fig[1, 1], width = Relative(0.35), height = Relative(0.4), halign = 0.08, valign = 0.95,  
        backgroundcolor = :white, xticks = LinearTicks(2), xtickformat = values -> [@sprintf("%.3f", v) for v in values],
        xgridvisible = true, ygridvisible = true, xticklabelsize = inset_ticklabelsize, yticklabelsize = inset_ticklabelsize,
        limits = ((in_xmin, in_xmax), (in_ymin, in_ymax)))
    
    translate!(ax_inset.blockscene, 0, 0, inset_layer)
    translate!(ax_inset.scene, 0, 0, inset_layer)

    # --- Snapshots and Bridge Levels ---
    main_linewidth        = 4.5
    bridge_linewidth      = 4.0
    bridge_x_offset       = 0.9     # Horizontal overhang past the right peak
    bridge_label_offset   = 1.05    # Offset past overhang for text annotation
    bridge_label_fontsize = 25
    bridge_linestyle      = (:dot, :dense)

    println("Extracting and rendering $(length(valid_indices)) snapshots of Lambda with inset...")

    for (k, y_snap) in enumerate(integrated_snaps)
        step_idx       = valid_indices[k]
        frame_max_grad = maximum(abs, @view raw_data_mat[:, step_idx])
        line_col       = line_colors[mod1(k, length(line_colors))]
        time_str       = @sprintf("%.3f", actual_times[k])
        
        y_snap_lo, y_snap_hi = extrema(y_snap)
        @printf("τ = %s | Λ extrema: (%.2f, %.2f) | Frame max|Λ_ζ|: %.3f\n", time_str, y_snap_lo, y_snap_hi, frame_max_grad)

        # Main profile curve
        lines!(ax_main, x_grid, y_snap, color = line_col, linewidth = main_linewidth, 
               label = L"\tau = %$time_str", joinstyle = :miter)
               
        # Identify central peak pair relative to the amplitude-weighted center coordinate
        idx_L, idx_R, _ = locate_central_peaks(x_grid, y_snap, n_peak_pairs)
        x_peak_L = x_grid[idx_L]
        x_peak_R = x_grid[idx_R]
        y_bridge = max(y_snap[idx_L], y_snap[idx_R])
        
        # Dotted horizontal bridge marker linking symmetric peaks
        lines!(ax_main, [x_peak_L, x_peak_R + bridge_x_offset], [y_bridge, y_bridge], 
               color = line_col, linestyle = bridge_linestyle, linewidth = bridge_linewidth)
        
        # Peak amplitude numerical label
        maxY_str = @sprintf("%.2f", y_bridge)
        text!(ax_main, x_peak_R + bridge_label_offset, y_bridge, text = L"\mathbf{%$(maxY_str)}", 
              color = line_col, align = (:left, :center), fontsize = bridge_label_fontsize)
    end

    # --- Zoomed Inset ---
    inset_linewidth    = 4.0
    base_marker_size   = 13
    base_stroke_width  = 2.0
    peak_radius_factor = 3.5     # Captures apex ±3 flanking nodes (7 nodes total)
    peak_marker_size   = 16
    peak_stroke_width  = 3.0
    box_pad_x          = 0.12    # Horizontal bounding box margin
    box_linewidth      = 2.0

    latest_color = line_colors[1]
    lines!(ax_inset, x_grid, latest_snap, color = latest_color, linewidth = inset_linewidth, joinstyle = :miter)

    # Base and flanking grid nodes captured in the inset window
    scatter!(ax_inset, x_grid[window_mask], latest_snap[window_mask], 
             color = (:white, 0.2), strokecolor = :black, strokewidth = base_stroke_width, markersize = base_marker_size)

    # Highlighted nodes comprising the sharp peak spike
    peak_mask = abs.(x_grid .- peak_x) .<= peak_radius_factor * dx_local
    scatter!(ax_inset, x_grid[peak_mask], latest_snap[peak_mask], 
             color = (:white, 0.5), strokecolor = :black, strokewidth = peak_stroke_width, 
             markersize = peak_marker_size, label = "Grid\nnodes")

    # Bounding box on main axis indicating inset region
    box_x = [in_xmin - box_pad_x, in_xmax + box_pad_x, in_xmax + box_pad_x, in_xmin - box_pad_x, in_xmin - box_pad_x]
    box_y = [in_ymin, in_ymin, in_ymax, in_ymax, in_ymin]
    lines!(ax_main, box_x, box_y, color = :black, linestyle = :dash, linewidth = box_linewidth)

    # --- Legends, Console Printouts and File Export ---
    legend_layer        = 50
    inset_legend_layer  = inset_layer + legend_layer
    main_leg_labelsize  = 29
    inset_leg_labelsize = 22
    render_density      = 3

    main_leg  = axislegend(ax_main, position = :rt, framevisible = true, labelsize = main_leg_labelsize)
    inset_leg = axislegend(ax_inset, position = :rt, framevisible = true, labelsize = inset_leg_labelsize) 
    translate!(main_leg.blockscene, 0, 0, legend_layer)
    translate!(inset_leg.blockscene, 0, 0, inset_legend_layer)

    println("-------------------------------------------------------------")
    @printf("Peak max|Λ_ζ| across plotted snapshots : %.3f\n", peak_grad_snaps)
    @printf("Peak max|Λ_ζ| across entire dataset    : %.3f\n", peak_grad_global)
    println("-------------------------------------------------------------")

    save(save_path, fig, px_per_unit = render_density)
    println("Saved snapshot plot: $save_path")
end


# ============================================================================== #
# SIMULATION CONFIGURATION AND EXECUTION
# tag_config_and_execution
# ============================================================================== #

# --- Configuration ---
work_directory = raw"C:\Users\PC\Desktop\PRA_submission_scripts"
workspace_name = :Workspace_Auxiliary

run_config = (
    # Run identifier (String, Symbol, or nothing / :_ for auto-incremented Run_i name)
    run_name            = :_,
    
    # Laser pulse parameters and particle setup
    units               = PhysicalUnits(m = 1.0, q = -1.0, c = 137.036, ħ = 1.0),
    pulse_params        = PulseParameters(A0 = 13.0, λ_nm = 800.0, σ = 10.0, shelf_duration_fs = 15.0),
    initial_momentum    = SVector(0.0, 0.0, 0.0),
    
    # Spatial grid resolution bounds (dZ_min in refined shelf, dZ_max in outer regions)
    spatial_resolutions = (1.5e-4, 1e-2),
    N_time_steps        = 200,
    
    # Temporal integration interval and recording window
    T_max               = -15.0,   # Symmetrical |T_min| if nothing
    T_i                 = -15.50,  # Recording window start
    T_f                 = -15.10,  # Recording window end
    
    # Root workspace directory and separate workspace folder
    work_directory      = work_directory,
    workspace_name      = workspace_name
)

target_timestamps = [-15.22, -15.19, -15.16, -15.13]

# --- Execution ---
# - Run simulation and save data into <work_directory>/<workspace_name>/Data/ :
# run_and_save(run_config)


# --- Workflow ---
# - Load dataset into REPL:
# loaded_data = interactive_loader(work_directory, workspace_name)

# - Render forensic evolution animation (saves directly to <workspace>/Animations/):
# -- Minimal call; defaults: zeta_lims = (-10, 10), time_window = (-Inf, Inf), fps = 30:
# animate_results(loaded_data, :Lambda)
#
# -- Explicit keyword call:
# animate_results(loaded_data, :Lambda;
#                 zeta_lims     = (-10.0, 10.0),
#                 time_window   = (-15.5, -15.1),
#                 fps           = 30,
#                 save_filename = "auxiliary_coarse.mp4")

# - Generate snapshots figure and report gradient values (saves to <workspace>/Plots/) ---
# -- Minimal call; defaults: zeta_lims = (-10.5, 10.5), n_peak_pairs  = 1, save_filename = "Lambda_snapshots.pdf":
# plot_Lambda_snapshots(loaded_data, target_timestamps)
#
# -- Explicit keyword call:
# plot_Lambda_snapshots(loaded_data, target_timestamps;
#                       zeta_lims     = (-10.5, 10.5),
#                       n_peak_pairs  = 1,
#                       save_filename = "Lambda_snapshots.pdf")