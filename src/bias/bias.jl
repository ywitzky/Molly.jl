# Bias potentials

export
    LinearBias,
    bias_gradient,
    SquareBias,
    FlatBottomSquareBias,
    PeriodicFlatBottomBias,
    BiasPotential


@doc raw"""
    LinearBias(k, cv_target)

A linear bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = k |\boldsymbol{s} - \boldsymbol{s}_t|
```
where $s$ and $s_t$ are the system and target CV values respectively.

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct LinearBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(lb::LinearBias, cv_sim; kwargs...)
    return lb.k * abs(cv_sim - lb.cv_target)
end

"""
    bias_gradient(bias::BiasType, cv_sim::Real)

Calculate the gradient of a bias potential with respect to the value of a
collective variable.

# Arguments
- `b::BiasType`: A struct that defines the type of bias to be used.
- `cv_sim::Real`: The value of a measured collective variable given
    the coordinates of a simulation.
"""
function bias_gradient(lb::LinearBias, cv_sim)
    d = cv_sim - lb.cv_target
    iszero(d) && return zero(lb.k)
    return lb.k * d / abs(d)
end

@doc raw"""
    SquareBias(k, cv_target)

A harmonic bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (\boldsymbol{s} - \boldsymbol{s}_t)^2
```
where $s$ and $s_t$ are the system and target CV values respectively.

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct SquareBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(sb::SquareBias, cv_sim; kwargs...)
    return (sb.k / 2) * (cv_sim - sb.cv_target)^2
end

function bias_gradient(sb::SquareBias, cv_sim)
    return sb.k * (cv_sim - sb.cv_target)
end


 function validate_flat_bottom_width(r_fb, label::AbstractString)
    if !isfinite(ustrip(r_fb)) || r_fb < zero(r_fb)
        throw(ArgumentError("$(label) flat-bottom width must be finite and non-negative, got $(r_fb)."))
    end
    return r_fb
end

@doc raw"""
    FlatBottomSquareBias(k, r_fb, cv_target)

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_t| - r_{fb})^2 H
```
where $s$ and $s_t$ are the system and target CV values respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| \geq r_{fb} \\
\end{array} \right.
```

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct FlatBottomSquareBias{K, R, C}
    k::K
    r_fb::R
    cv_target::C

    function FlatBottomSquareBias(k::K, r_fb::R, cv_target::C) where {K, R, C}
        validate_flat_bottom_width(r_fb, "FlatBottomSquareBias")
        return new{K, R, C}(k, r_fb, cv_target)
    end
end

function potential_energy(fb::FlatBottomSquareBias, cv_sim; kwargs...)
    d_abs = abs(cv_sim - fb.cv_target)
    H = (d_abs < fb.r_fb ? 0 : 1)
    return (fb.k / 2) * (d_abs - fb.r_fb)^2 * H
end

function bias_gradient(fb::FlatBottomSquareBias, cv_sim)
    d = cv_sim - fb.cv_target
    d_abs = abs(d)
    d_abs <= fb.r_fb && return zero(fb.k * fb.r_fb)
    return fb.k * (d_abs - fb.r_fb) * d / d_abs
end

@doc raw"""
    PeriodicFlatBottomBias(k, r_fb, cv_target)

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

This variant handles periodicity in the CV wrapping around the (-π, π) range.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_t| - r_{fb})^2 H
```
where $s$ and $s_t$ are the system and target CV values respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| \geq r_{fb} \\
\end{array} \right.
```

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct PeriodicFlatBottomBias{K, R, T}
    k::K
    r_fb::R
    cv_target::T

    function PeriodicFlatBottomBias(k::K, r_fb::R, cv_target::T) where {K, R, T}
        validate_flat_bottom_width(r_fb, "PeriodicFlatBottomBias")
        return new{K, R, T}(k, r_fb, cv_target)
    end
end

 function periodic_flat_bottom_displacement(cv_sim, cv_target)
    d = cv_sim - cv_target
    FT = typeof(float(ustrip(d)))
    twopi = FT(2π) * oneunit(d)
    half_period = twopi / FT(2)
    return mod(d + half_period, twopi) - half_period
end

function potential_energy(pb::PeriodicFlatBottomBias, cv_sim; kwargs...)
    FT = typeof(float(ustrip(cv_sim - pb.cv_target)))
    d_wrapped = periodic_flat_bottom_displacement(cv_sim, pb.cv_target)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_fb
        return zero(pb.k * pb.r_fb^2)
    else
        disp = dist - pb.r_fb
        return FT(0.5) * pb.k * disp^2
    end
end

function bias_gradient(pb::PeriodicFlatBottomBias, cv_sim)
    d_wrapped = periodic_flat_bottom_displacement(cv_sim, pb.cv_target)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_fb
        return zero(pb.k * pb.r_fb)
    else
        disp = dist - pb.r_fb
        return pb.k * disp * sign(d_wrapped)
    end
end

"""
    BiasPotential(cv_type, bias_type)

A potential to bias a simulation along a collective variable (CV), implemented
as an AtomsCalculators.jl calculator.

The `cv_type` could for example be [`CalcDist`](@ref) and the `bias_type`
could be [`LinearBias`](@ref).

Forces resulting from the bias potential are evaluated in two steps, specfically by
(1) calculating the gradient of the bias potential with respect to the value of the CV, and
(2) calculating the gradient of the CV with respect to the atomic coordinates.

Gradients can be calculated with either automatic differentiation or explicitly defined
gradient functions.
Enzyme should be imported in the first case.

Virial contributions must be explicitly defined.

CV computation runs fully on the GPU when the `System` is GPU-resident, with no host transfer of
coordinates, atoms or forces, including `cv_type.correction = :pbc` (the default for the built-in
CV types), which unwraps bonded molecules across the periodic boundary using a GPU-native
spanning-forest traversal.
"""
mutable struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
    uses_persistent_buffers::Bool   # fixed at construction, see uses_builtin_cv_gradient!
    grad::Any                       # lazily-allocated N_atoms-sized CV gradient buffer
    d_buf::Any                      # lazily-allocated 1-element CV-value buffer
    fs_svec::Any                    # lazily-allocated N_atoms-sized bias-force buffer
    dist_scratch::Any               # CalcMinDist/CalcMaxDist only: fused-kernel O(group) scratch
    extremal_cache::Any             # CalcMinDist/CalcMaxDist only: cached extremal pair for virial reuse
    d_bias_buf::Any                 # lazily-allocated 1-element device buffer: bias_gradient's output,
                                     # used only on the cuda_graph_capturing=true path (see forces! below)
end

function BiasPotential(cv_type::C, bias_type::B) where {C, B}
    return BiasPotential{C, B}(cv_type, bias_type, uses_builtin_cv_gradient!(cv_type),
                               nothing, nothing, nothing, nothing, nothing, nothing)
end

bias_all_finite(values::AbstractArray) = all(bias_all_finite, values)
bias_all_finite(value) = isfinite(ustrip(value))

 function bias_max_abs_ustrip(values::AbstractArray)
    isempty(values) && return 0.0
    return mapreduce(bias_max_abs_ustrip, max, values)
end

bias_max_abs_ustrip(value) = abs(ustrip(value))

 function check_bias_finite(value, label::AbstractString, bias::BiasPotential;
                            cv_sim=nothing, max_abs_component=nothing)
    bias_all_finite(value) && return value
    msg = "BiasPotential with CV $(typeof(bias.cv_type)) and bias " *
          "$(typeof(bias.bias_type)) produced non-finite $(label)"
    if !isnothing(cv_sim)
        msg *= ", cv_sim=$(cv_sim)"
    end
    if !isnothing(max_abs_component)
        msg *= ", max_abs_component=$(max_abs_component)"
    end
    error(msg * ".")
end

# `buffers`/`step_n`, when supplied (from AtomsCalculators.forces!'s existing kwarg plumbing,
# force.jl's general_inters loop), route through the shared, once-per-step unwrap cache
# (ensure_unwrapped_coords!, src/force.jl) instead of recomputing unwrap_molecules independently
# for every attached BiasPotential -- falls back to today's per-call behaviour otherwise.
function bias_coords(sys, cv_type, buffers=nothing, step_n=nothing)
    cv_type.correction != :pbc && return sys.coords
    if buffers !== nothing && step_n !== nothing && hasproperty(buffers, :unwrapped_coords)
        return ensure_unwrapped_coords!(buffers, sys, step_n)
    end
    return unwrap_molecules(sys)
end

bias_needs_unwrap(b::BiasPotential) = b.cv_type.correction == :pbc

# Lazily allocate BiasPotential's persistent CV-value/gradient scratch (bias.grad/bias.d_buf) the
# first time it's needed. Only reached when bias.uses_persistent_buffers is true (a CV type with
# a real cv_gradient!/calculate_cv! -- custom/AD-only CV types never take this path, see
# uses_builtin_cv_gradient! in cv.jl).
function ensure_bias_buffers!(bias::BiasPotential, coords)
    if bias.grad === nothing
        bias.grad, bias.d_buf = zero_cv_gradient_buffers(bias.cv_type, coords)
    end
    return nothing
end

# Uploads a host index Vector to a persistent device array once -- shared by every ensure_*_scratch!
# below so none of them repeat `@view coords[cv.atom_inds_1]`'s per-call re-upload (see
# MinMaxScratch's docstring, cv.jl, for how that was confirmed live).
function upload_idx(coords, inds::Vector{Int})
    idx_dev = similar(coords, Int, length(inds))
    copyto!(idx_dev, inds)
    return idx_dev
end

# Lazily allocate CalcMinDist/CalcMaxDist-specific scratch: `extremal_cache` (small, cheap on any
# backend) lets calculate_virial! reuse cv_gradient!'s search result instead of recomputing it;
# `dist_scratch` (a `MinMaxScratch`) is only allocated when `coords` is actually GPU-resident,
# since the CPU path never uses it. `idx1_dev`/`idx2_dev` upload `cv.atom_inds_1`/`atom_inds_2` to
# the device exactly once here -- see MinMaxScratch's docstring (cv.jl) for why that matters (it's
# what lets the two-kernel calculate_cv!/cv_gradient! path avoid `@view coords[cv.atom_inds_1]`,
# which re-uploads those same indices on every call otherwise).
#
# Also handles CalcCMDist (a `CMDistScratch`, see cv.jl), CalcRg (a `RgScratch`), and CalcRMSD (a
# `RmsdScratch`) the same way: each CV type's own dispatch of calculate_cv!/cv_gradient! only takes
# its fused-kernel fast path once its matching scratch struct has been populated here, falling back
# to the generic broadcast path otherwise (CPU, or a caller that never supplied `scratch=`).
function ensure_bias_dist_scratch!(bias::BiasPotential, coords, atoms)
    cv = bias.cv_type
    if cv isa CalcDist{<:Union{CalcMinDist, CalcMaxDist}} && bias.extremal_cache === nothing
        bias.extremal_cache = ExtremalPairCache(false, 0, 0, nothing, nothing)
        if is_gpu_resident(coords)
            idx1_dev, idx2_dev = upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2)
            na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
            # T caps mindist_tile_kernel!'s worker count (and mindist_finalize_*_kernel!'s serial
            # scan length) at MINDIST_TILE_CAP regardless of how large na*nb gets -- see
            # MinMaxScratch's docstring (cv.jl) for why this replaced the old O(na)-sized buffers.
            T = min(na * nb, MINDIST_TILE_CAP)
            bias.dist_scratch = MinMaxScratch(
                idx1_dev,
                idx2_dev,
                similar(coords, eltype(eltype(coords)), T),
                similar(coords, Int, T),
                similar(coords, Int, T),
                similar(coords, T),
                similar(coords, Int, 1),
                similar(coords, Int, 1),
                similar(coords, 1),
            )
        end
    elseif cv isa CalcDist{CalcCMDist} && bias.dist_scratch === nothing && is_gpu_resident(coords)
        # T1/T2 (capped at 1024) size the partial-reduction buffers cmdist_reduce_kernel! writes
        # into -- see CMDistScratch's docstring (cv.jl) for why this is what fixes CMDist's
        # scaling. Sample types (not values -- `only(Array(...[1:1]))` is a one-off, construction-
        # time-only host sync) drive the buffer eltypes: a mass-weighted position has different
        # units than either a bare position or a bare mass, so `similar(coords, ...)` alone can't
        # produce them.
        na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
        T1, T2 = min(na, 1024), min(nb, 1024)
        coord_sample = only(Array(view(coords, 1:1)))
        mass_sample = mass(only(Array(view(atoms, 1:1))))
        wpos_sample = coord_sample * mass_sample
        dir_sample = coord_sample / norm(coord_sample)
        bias.dist_scratch = CMDistScratch(
            upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2),
            similar(coords, typeof(mass_sample), T1), similar(coords, typeof(wpos_sample), T1),
            similar(coords, typeof(mass_sample), T2), similar(coords, typeof(wpos_sample), T2),
            similar(coords, typeof(dir_sample), 1),
            similar(coords, typeof(mass_sample), 1), similar(coords, typeof(mass_sample), 1),
        )
    elseif cv isa CalcRg && bias.dist_scratch === nothing && is_gpu_resident(coords)
        # T (capped at RG_TILE_CAP) sizes the partial-reduction buffers rg_com_reduce_kernel!/
        # rg_isum_reduce_*_kernel! write into -- see RgScratch's docstring (cv.jl) for why this is
        # what fixes Rg's scaling. Same construction-time-only sample-type probing as CMDistScratch
        # above (a mass-weighted position and a mass-weighted squared-deviation each have their
        # own units, so `similar(coords, ...)` alone can't produce them).
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        n = length(inds)
        T = min(n, RG_TILE_CAP)
        coord_sample = only(Array(view(coords, 1:1)))
        mass_sample = mass(only(Array(view(atoms, 1:1))))
        wpos_sample = coord_sample * mass_sample
        isum_sample = sum_abs2(coord_sample) * mass_sample
        bias.dist_scratch = RgScratch(
            upload_idx(coords, inds),
            similar(coords, typeof(mass_sample), T), similar(coords, typeof(wpos_sample), T),
            similar(coords, typeof(isum_sample), T),
            similar(coords, typeof(coord_sample), 1), similar(coords, typeof(mass_sample), 1),
        )
    elseif cv isa CalcRMSD && bias.dist_scratch === nothing && is_gpu_resident(coords)
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        ref_inds = iszero(length(cv.ref_atom_inds)) ? collect(1:length(cv.ref_coords)) : cv.ref_atom_inds
        ref_coords_used = cv.ref_coords[ref_inds]
        # cv.ref_coords never changes after construction, so its Kabsch-centered form (one host
        # sync) is computed exactly once, here, instead of on every cv_gradient!/calculate_cv! call
        # -- see RmsdScratch's docstring (cv.jl). T (capped at RMSD_TILE_CAP) sizes
        # rmsd_isum_reduce_kernel!'s partial-sum buffer -- `sum_abs2` of a length-valued SVector
        # gives a length^2-valued sample (`rot` in the actual reduction is unitless, so this
        # matches the real per-atom summand's type without needing rot/trans_1/trans_2 in hand).
        n = length(inds)
        T = min(n, RMSD_TILE_CAP)
        coord_sample = only(Array(view(coords, 1:1)))
        bias.dist_scratch = RmsdScratch(upload_idx(coords, inds), similar(coords, length(inds)),
                                        ref_coords_used, kabsch_centered(ref_coords_used),
                                        similar(coords, typeof(sum_abs2(coord_sample)), T))
    end
    return nothing
end

# Computes bias_gradient (a plain, branchy scalar function -- LinearBias/SquareBias/
# FlatBottomSquareBias/PeriodicFlatBottomBias's bodies above are all already GPU-kernel-safe
# scalar Julia, same as e.g. single_dist_cv_gradient_kernel!'s `if d > zero(d) ... end`) entirely
# device-side, writing into a persistent 1-element buffer instead of reading `cv_sim` back to the
# host first. Used only on the `cuda_graph_capturing=true` forces! path below: that path needs
# zero host syncs, since a host sync (or the host branch/error inside check_bias_finite) cannot
# appear inside a captured CUDA graph region.
@kernel inbounds=true function bias_gradient_kernel!(d_bias_buf, @Const(d_buf), bias_type)
    idx = @index(Global, Linear)
    idx == 1 && (d_bias_buf[1] = bias_gradient(bias_type, d_buf[1]))
end

# Fuses `fs_svec .= d_bias_buf .* grad` and `fs .-= fs_svec` (previously 2 separate N_atoms-sized
# broadcast kernels, `AtomsCalculators.forces!`'s cuda_graph_capturing branch below) into one
# N_atoms-sized kernel launch. `grad` is zero at every atom index the CV doesn't touch (cv_gradient!
# only ever writes its own group's indices, never clears the rest), so most threads here do a
# harmless zero-write; the point isn't skipping that work, it's halving the *launch* count this
# costs per bias per step -- with `n_bias` BiasPotentials each contributing this pair, that's
# `n_bias` fewer graph nodes/launches per step.
@kernel inbounds=true function bias_apply_force_kernel!(fs, fs_svec, @Const(grad), @Const(d_bias_buf))
    i = @index(Global, Linear)
    v = d_bias_buf[1] * grad[i]
    fs_svec[i] = v
    fs[i] -= v
end

function ensure_bias_gradient_buffer!(bias::BiasPotential)
    if bias.d_bias_buf === nothing
        # bias_gradient's output has units of energy/CV-unit (e.g. kJ/mol/nm for a distance CV),
        # NOT bias.d_buf's own units (the CV's native units, e.g. nm) -- similar(bias.d_buf, 1)
        # would allocate the wrong (mismatched) unit type. Derive the correct type by running
        # bias_gradient once on a representative host value (cheap, one-off, not a real CV
        # evaluation) instead of guessing/hardcoding a units formula per bias type.
        sample = bias_gradient(bias.bias_type, oneunit(eltype(bias.d_buf)))
        bias.d_bias_buf = similar(bias.d_buf, typeof(sample), 1)
    end
    return nothing
end

# Periodic, out-of-graph finiteness check for a BiasPotential whose forces! is being run with
# cuda_graph_capturing=true (which skips check_bias_finite inside forces! itself, since it
# host-syncs/branches -- illegal during graph capture). Call this every `finite_check_every`
# steps from simulate!'s step loop instead, outside any captured region. A no-op until the
# relevant buffers have actually been allocated (i.e. before forces!/potential_energy's first call).
function check_bias_finite_periodic(bias::BiasPotential)
    bias.d_buf === nothing && return nothing
    cv_sim = only(from_device(bias.d_buf))
    check_bias_finite(cv_sim, "collective variable", bias)
    bias.grad === nothing || check_bias_finite(bias.grad, "CV gradient", bias; cv_sim=cv_sim)
    if bias.fs_svec !== nothing
        check_bias_finite(bias.fs_svec, "bias force", bias; cv_sim=cv_sim,
                          max_abs_component=bias_max_abs_ustrip(bias.fs_svec))
    end
    return nothing
end

function AtomsCalculators.potential_energy(sys, bias::BiasPotential; kwargs...)
    coords = bias_coords(sys, bias.cv_type)

    if bias.uses_persistent_buffers
        ensure_bias_buffers!(bias, coords)
        ensure_bias_dist_scratch!(bias, coords, sys.atoms)
        calculate_cv_buffered!(bias.cv_type, coords, sys.atoms, sys.boundary, bias.d_buf, sys.velocities;
                               scratch=bias.dist_scratch, kwargs...)
        cv_sim = only(from_device(bias.d_buf))
    else
        cv_sim = calculate_cv(bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities; kwargs...)
    end
    check_bias_finite(cv_sim, "collective variable", bias)

    pe = potential_energy(bias.bias_type, cv_sim; kwargs...)
    return check_bias_finite(pe, "potential energy", bias; cv_sim=cv_sim)
end

function AtomsCalculators.forces!(
    fs, sys, bias::BiasPotential;
    needs_vir::Bool = false,
    buffers = nothing, # Dummy to be able to have explicit kwarg. In reality a buffer will always be passed
    step_n = nothing,
    cuda_graph_capturing::Bool = false,
    kwargs...
)
    coords = bias_coords(sys, bias.cv_type, buffers, step_n)

    # cuda_graph_capturing=true: zero-host-sync path for use inside a captured CUDA graph (see
    # simulators.jl's use_cuda_graph option). Requires bias.uses_persistent_buffers (checked at
    # simulate! entry, not here) and skips every check_bias_finite call, since each one host-syncs
    # or host-branches -- illegal during graph capture. Use check_bias_finite_periodic (above),
    # called from simulate!'s step loop outside the captured region, instead.
    if cuda_graph_capturing
        ensure_bias_buffers!(bias, coords)
        ensure_bias_dist_scratch!(bias, coords, sys.atoms)
        ensure_bias_gradient_buffer!(bias)
        cv_gradient!(bias.grad, bias.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                    extremal_cache=bias.extremal_cache, scratch=bias.dist_scratch)
        backend = get_backend(bias.d_buf)
        kernel! = bias_gradient_kernel!(backend, 1)
        kernel!(bias.d_bias_buf, bias.d_buf, bias.bias_type; ndrange=1)
        # bias_apply_force_kernel! below writes every index of fs_svec itself (ndrange=n_atoms
        # covers the whole array every call), so this only needs the right shape/eltype, not zeros.
        # `similar(fs)` (not `bias.grad`) because fs_svec's eltype is d_bias_buf .* grad's
        # (force-with-units), same as fs's own eltype -- grad itself is unitless.
        bias.fs_svec === nothing && (bias.fs_svec = similar(fs))
        # needs_vir is excluded from the captured region entirely (see simulators.jl), so no
        # calculate_virial! call here.
        n_atoms = length(bias.grad)
        apply_kernel! = bias_apply_force_kernel!(backend, min(n_atoms, 256))
        apply_kernel!(fs, bias.fs_svec, bias.grad, bias.d_bias_buf; ndrange=n_atoms)
        return fs
    end

    # Gradient of CV with respect to coordinates
    if bias.uses_persistent_buffers
        ensure_bias_buffers!(bias, coords)
        ensure_bias_dist_scratch!(bias, coords, sys.atoms)
        # Also warm up d_bias_buf here (not just on the cuda_graph_capturing path above) so that
        # if a later step DOES call the cuda_graph_capturing branch first-of-its-kind, it isn't
        # the one allocating it -- see the cuda_graph_capturing branch's comment above and
        # simulate!'s explicit warm-up call (src/simulators.jl) for the same reasoning.
        is_gpu_resident(coords) && ensure_bias_gradient_buffer!(bias)
        cv_gradient!(bias.grad, bias.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                    extremal_cache=bias.extremal_cache, scratch=bias.dist_scratch)
        d_coords, cv_sim = bias.grad, only(from_device(bias.d_buf))
    else
        d_coords, cv_sim = cv_gradient(
            bias.cv_type,
            coords,
            sys.atoms,
            sys.boundary,
            sys.velocities,
        )
    end
    check_bias_finite(cv_sim, "collective variable", bias)
    check_bias_finite(d_coords, "CV gradient", bias; cv_sim=cv_sim)

    # Gradient of bias function with respect to CV
    d_bias = bias_gradient(bias.bias_type, cv_sim)
    check_bias_finite(d_bias, "bias gradient", bias; cv_sim=cv_sim)

    if bias.uses_persistent_buffers
        if bias.fs_svec === nothing
            bias.fs_svec = d_bias .* d_coords
        else
            bias.fs_svec .= d_bias .* d_coords
        end
        fs_svec = bias.fs_svec
    else
        fs_svec = d_bias .* d_coords
    end
    check_bias_finite(
        fs_svec,
        "bias force",
        bias;
        cv_sim=cv_sim,
        max_abs_component = bias_max_abs_ustrip(fs_svec),
    )

    if needs_vir && bias.cv_type.has_virial
        calculate_virial!(buffers.virial, bias.cv_type, coords, -fs_svec, sys.atoms, sys.boundary;
                          precomputed_extremum=bias.extremal_cache)
    end

    fs .-= fs_svec
    return fs
end
