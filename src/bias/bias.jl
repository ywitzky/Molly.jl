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
    bad_step::Any                   # lazily-allocated 1-element device Int buffer: first step_n at which
                                     # a deferred finite check failed, or 0 -- see check_bias_finite_deferred!
end

function BiasPotential(cv_type::C, bias_type::B) where {C, B}
    return BiasPotential{C, B}(cv_type, bias_type, uses_builtin_cv_gradient!(cv_type),
                               nothing, nothing, nothing, nothing, nothing, nothing, nothing)
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

# Compile-time-stable partition of a heterogeneous `general_inters` tuple into its BiasPotential
# entries and everything else, preserving each group's relative order. A runtime `filter` isn't
# type-stable on a `Tuple` with mixed element types, so this recurses like `Base.map`/`Base.tail`
# do instead. Used by force.jl's cuda_graph_capturing branch to batch every attached
# BiasPotential's captured-path tail (see bias_batched_tail!) while leaving any other
# general_inters type (e.g. a future non-bias GPU-capturable interaction) on its current,
# unbatched, per-inter AtomsCalculators.forces! call, unchanged.
@inline split_biases(::Tuple{}) = (), ()
@inline function split_biases(t::Tuple)
    rest_biases, rest_others = split_biases(Base.tail(t))
    x = first(t)
    return x isa BiasPotential ? ((x, rest_biases...), rest_others) : (rest_biases, (x, rest_others...))
end

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

function ensure_bias_finite_buffer!(bias::BiasPotential)
    if bias.bad_step === nothing
        bias.bad_step = similar(bias.d_buf, Int, 1)
        bias.bad_step .= 0
    end
    return nothing
end

# One grid-stride kernel, no output allocation: every thread that finds a non-finite element
# writes step_n into bad_step[1] (only if it's still 0). Concurrent writers only ever write the
# *same* step_n, so the race is harmless and needs no atomic. This replaces an earlier
# `mapreduce(...; dims=1)` + broadcast version -- `mapreduce` with `dims` allocates a fresh output
# array on every call, and doing that inside a captured CUDA graph region (3x per bias per step)
# shifts the pool's allocation pattern between calls, degrading @captured's cuGraphExecUpdate to a
# slow path that scales badly with n_bias (confirmed live: capture speedup collapsed from ~2.1-2.7x
# to <1x as n_bias grew). This kernel allocates nothing after bad_step itself is created.
@kernel inbounds=true function bias_finite_check_kernel!(bad_step, @Const(value), step_n::Int)
    idx = @index(Global, Linear)
    if idx <= length(value) && !bias_all_finite(value[idx]) && bad_step[1] == 0
        bad_step[1] = step_n
    end
end

# Records a non-finite `value` (an N_atoms- or 1-element device array -- d_buf/grad/fs_svec all
# qualify) without reading anything back to the host and without allocating (see
# bias_finite_check_kernel! above): the *first* bad step wins and every check after that is a
# no-op write of the same value -- no host branch, no sync, safe to call every step. Used in place
# of `check_bias_finite` for array-valued checks when `defer_finite_check=true` (Langevin's
# use_cuda_graph path, both its captured and periodic-fallback steps -- see AtomsCalculators.forces!
# below); the actual error, if any, is only raised later by check_bias_finite_periodic's cheap
# 1-element readback.
function check_bias_finite_deferred!(value::AbstractArray, bias::BiasPotential, step_n::Integer)
    backend = get_backend(value)
    n = length(value)
    kernel! = bias_finite_check_kernel!(backend, min(n, 256))
    kernel!(bias.bad_step, value, Int(step_n); ndrange=n)
    return nothing
end

# --- Batched captured-path tail across every attached BiasPotential at once ------------------
#
# cv_gradient! stays per-bias (genuinely CV-type-specific: different CVs need different reduction
# kernels -- see bias_cv_step! below). bias_gradient (LinearBias/SquareBias/FlatBottomSquareBias/
# PeriodicFlatBottomBias dispatch) and the remaining tail (finite check + fs_svec = d_bias_buf .*
# grad; fs -= fs_svec) are both generic per-element work with no per-bias *kernel* needed -- a
# tuple-recursion-unrolled kernel dispatches each bias's own bias_type/array at compile time (see
# the NTuple recursion helpers below), same trick either way -- so they batch into 3 kernel
# launches total per step instead of 3*n_bias -- see force.jl's cuda_graph_capturing branch for
# the call site.
#
# grad/fs_svec are `NTuple`s of every bias's own buffer: both come from `ustrip_vec.(zero(coords))`/
# `similar(fs)`, so they share one concrete element type across every non-Torsion,
# persistent-buffer CV, making their NTuples homogeneous. d_buf/d_bias_buf/bad_step differ in
# Unitful eltype per bias (different CV/energy units) -- their NTuples are heterogeneous but still
# compile-time-resolved, same as any ordinary Julia function taking a heterogeneous tuple.
# Recursion (not a runtime `for i in eachindex(t)` over a heterogeneous tuple) is what keeps
# indexing type-stable/GPU-codegen-safe here -- the standard Julia idiom (see e.g. Base.map's own
# tuple recursion). Both helpers below assume at least one bias (only called when
# length(biases) > 0), so the base case is the 1-tuple, not the empty tuple.
@inline function _bias_apply_recurse(grads::Tuple{Any}, fs_svecs::Tuple{Any}, d_bias_bufs::Tuple{Any}, i)
    v = d_bias_bufs[1][1] * grads[1][i]
    fs_svecs[1][i] = v
    return v
end
@inline function _bias_apply_recurse(grads::Tuple, fs_svecs::Tuple, d_bias_bufs::Tuple, i)
    v = d_bias_bufs[1][1] * grads[1][i]
    fs_svecs[1][i] = v
    return v + _bias_apply_recurse(Base.tail(grads), Base.tail(fs_svecs), Base.tail(d_bias_bufs), i)
end

@kernel inbounds=true function bias_batched_apply_kernel!(fs, grads::NTuple{N}, fs_svecs::NTuple{N},
                                                            d_bias_bufs::NTuple{N}) where N
    i = @index(Global, Linear)
    fs[i] -= _bias_apply_recurse(grads, fs_svecs, d_bias_bufs, i)
end

@inline function _bias_check_recurse(values::Tuple{Any}, bad_steps::Tuple{Any}, i, step_n)
    v, bs = values[1], bad_steps[1]
    if i <= length(v) && !bias_all_finite(v[i]) && bs[1] == 0
        bs[1] = step_n
    end
    return nothing
end
@inline function _bias_check_recurse(values::Tuple, bad_steps::Tuple, i, step_n)
    v, bs = values[1], bad_steps[1]
    if i <= length(v) && !bias_all_finite(v[i]) && bs[1] == 0
        bs[1] = step_n
    end
    return _bias_check_recurse(Base.tail(values), Base.tail(bad_steps), i, step_n)
end

# One launch checks every attached bias's d_buf/grad/fs_svec at once (each shorter than
# ndrange=max(n_atoms) simply reads out of its own bounds check, same guard
# bias_finite_check_kernel! already uses per-array); each bias's own bad_step is written
# independently -- see check_bias_finite_deferred! for why concurrent same-value writes need no
# atomic.
@kernel inbounds=true function bias_batched_finite_kernel!(values::NTuple{N}, bad_steps::NTuple{N},
                                                             step_n::Int) where N
    i = @index(Global, Linear)
    _bias_check_recurse(values, bad_steps, i, step_n)
end

# Per-bias half of the captured-path tail: computes bias.grad/d_buf, the one genuinely
# CV-type-specific step that can't batch across biases (different CVs need different reduction
# kernels -- see cv.jl). bias_gradient_kernel! (bias-type-specific dispatch, e.g. Linear/Square/
# FlatBottom/Periodic) used to run here too, one ndrange=1 launch per bias; it's now folded into
# bias_batched_tail!'s batched-gradient kernel below (same recursion-over-heterogeneous-tuple
# trick as the force-apply/finite-check batching -- a scalar function dispatching on a
# compile-time-resolved tuple slot batches exactly like an array-valued one does). Does not touch
# fs (beyond using it as a shape/eltype template for fs_svec, matching the original single-launch
# path exactly); bias_batched_tail! does the batched, dispatch-free remainder for every bias at
# once.
#
# `do_check`: whether to run the finite-check kernels at all -- gates all 3 (2 here, 1 more in
# bias_batched_tail!) together. Needed by the two-graph capture-once design (see
# captured_forces_once!, MollyCUDAExt.jl): the no-check graph must never launch a finite-check
# kernel (so it stays free of a baked-in step_n it could never update once captured); the
# with-check graph is captured once with `step_n` fixed to a sentinel (not the real step number --
# it's only ever launched again by *replaying* the already-captured graph, so a value baked in at
# capture time would go stale) and simulate!'s step loop reports "somewhere in the last
# finite_check_every steps" rather than an exact step on failure, using its own host-side step
# counter (see simulators.jl).
function bias_cv_step!(bias::BiasPotential, sys, coords, fs, step_n, do_check::Bool=true)
    ensure_bias_buffers!(bias, coords)
    ensure_bias_dist_scratch!(bias, coords, sys.atoms)
    ensure_bias_gradient_buffer!(bias)
    # Always ensured (not gated on do_check): bias.bad_step must already exist by the time the
    # with-check graph is captured, and ensure_bias_finite_buffer! is cheap/idempotent once
    # allocated -- see simulate!'s warm-up call (src/simulators.jl) for the same reasoning applied
    # to bias.grad/d_buf/d_bias_buf.
    ensure_bias_finite_buffer!(bias)
    cv_gradient!(bias.grad, bias.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                extremal_cache=bias.extremal_cache, scratch=bias.dist_scratch)
    if do_check
        check_bias_finite_deferred!(bias.d_buf, bias, step_n)
        check_bias_finite_deferred!(bias.grad, bias, step_n)
    end
    # `similar(fs)` (not `bias.grad`) because fs_svec's eltype is d_bias_buf .* grad's
    # (force-with-units), same as fs's own eltype -- grad itself is unitless.
    bias.fs_svec === nothing && (bias.fs_svec = similar(fs))
    return nothing
end

@inline function _bias_gradient_recurse(d_bias_bufs::Tuple{Any}, d_bufs::Tuple{Any}, bias_types::Tuple{Any})
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return nothing
end
@inline function _bias_gradient_recurse(d_bias_bufs::Tuple, d_bufs::Tuple, bias_types::Tuple)
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return _bias_gradient_recurse(Base.tail(d_bias_bufs), Base.tail(d_bufs), Base.tail(bias_types))
end

# Batches bias_gradient_kernel! (see bias_cv_step!'s docstring) across every bias -- bias_types is
# a plain (non-CuArray) NTuple of bias_type values, passed through to the device as kernel
# arguments like `boundary`/other bitstype-ish scalar args elsewhere in this codebase.
@kernel inbounds=true function bias_batched_gradient_kernel!(d_bias_bufs::NTuple{N}, @Const(d_bufs::NTuple{N}),
                                                               bias_types::NTuple{N}) where N
    idx = @index(Global, Linear)
    idx == 1 && _bias_gradient_recurse(d_bias_bufs, d_bufs, bias_types)
end

# Batched tail: after every bias in `biases` has already run bias_cv_step! this step (which
# already ran check_bias_finite_deferred! on d_buf/grad individually -- see bias_cv_step! above),
# computes every bias's bias_gradient (d_bias_buf), applies all their force contributions to `fs`,
# and checks the one array the tail itself produces (fs_svec) -- 3 kernel launches total (instead
# of 3*length(biases): 1 bias_gradient + 1 finite-check + 1 apply per bias) -- see the NTuple
# recursion helpers above.
function bias_batched_tail!(fs, biases::Tuple, step_n, do_check::Bool=true)
    isempty(biases) && return nothing
    grads       = map(b -> b.grad,       biases)
    fs_svecs    = map(b -> b.fs_svec,    biases)
    d_bias_bufs = map(b -> b.d_bias_buf, biases)
    bias_types  = map(b -> b.bias_type,  biases)
    backend = get_backend(fs)
    n_atoms = length(first(grads))
    grad_kernel! = bias_batched_gradient_kernel!(backend, 1)
    grad_kernel!(d_bias_bufs, map(b -> b.d_buf, biases), bias_types; ndrange=1)
    apply_kernel! = bias_batched_apply_kernel!(backend, min(n_atoms, 256))
    apply_kernel!(fs, grads, fs_svecs, d_bias_bufs; ndrange=n_atoms)
    if do_check
        bad_steps = map(b -> b.bad_step, biases)
        check_kernel! = bias_batched_finite_kernel!(backend, min(n_atoms, 256))
        check_kernel!(fs_svecs, bad_steps, Int(step_n); ndrange=n_atoms)
    end
    return nothing
end

# Periodic, out-of-graph readback for a BiasPotential using deferred finite checks (see
# check_bias_finite_deferred! above) -- call this every `finite_check_every` steps from simulate!'s
# step loop. A no-op until check_bias_finite_deferred! has actually run at least once.
function check_bias_finite_periodic(bias::BiasPotential)
    bias.bad_step === nothing && return nothing
    bad_step = only(from_device(bias.bad_step))
    if bad_step != 0
        error("BiasPotential with CV $(typeof(bias.cv_type)) and bias $(typeof(bias.bias_type)) " *
              "first produced a non-finite value at step $bad_step.")
    end
    return nothing
end

# Same as check_bias_finite_periodic, but for every attached BiasPotential at once: one host sync
# instead of n_bias separate ones. check_bias_finite_periodic itself is a `from_device` (CUDA
# sync) round trip regardless of payload size (tens of microseconds of driver/sync overhead, not
# the 8 bytes actually transferred), so calling it once per bias every `finite_check_every` steps
# (simulate!'s step loop, src/simulators.jl) was paying that fixed cost n_bias times where once
# suffices -- confirmed via profile_host_overhead.jl's per-step overhead accounting. `vcat` of the
# (tiny, n_bias-length) bad_step buffers is one cheap device-side kernel launch, then a single
# from_device does the one sync this whole function needs. Skips any bias whose bad_step hasn't
# been allocated yet (not yet warmed up).
function check_bias_finite_periodic_batched!(biases)
    live = Tuple(b for b in biases if b isa BiasPotential && b.bad_step !== nothing)
    isempty(live) && return nothing
    bad_steps_h = from_device(reduce(vcat, map(b -> b.bad_step, live)))
    for (bias, bad_step) in zip(live, bad_steps_h)
        if bad_step != 0
            error("BiasPotential with CV $(typeof(bias.cv_type)) and bias $(typeof(bias.bias_type)) " *
                  "first produced a non-finite value at step $bad_step.")
        end
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
    defer_finite_check::Bool = false,
    kwargs...
)
    coords = bias_coords(sys, bias.cv_type, buffers, step_n)

    # cuda_graph_capturing=true: zero-host-sync path for use inside a captured CUDA graph (see
    # simulators.jl's use_cuda_graph option). check_bias_finite itself host-syncs/branches --
    # illegal during graph capture -- so finite checks here always go through
    # check_bias_finite_deferred! (device-resident, no sync); check_bias_finite_periodic (above),
    # called from simulate!'s step loop outside the captured region, does the actual (cheap,
    # 1-element) readback and raises the error if anything was ever recorded.
    if cuda_graph_capturing
        # needs_vir is excluded from the captured region entirely (see simulators.jl), so no
        # calculate_virial! call here. Single-bias call: bias_cv_step! (per-bias, CV/bias-type-
        # specific) then bias_batched_tail! on a 1-tuple -- same 2 kernel launches the batched
        # multi-bias path (force.jl) uses, so a lone BiasPotential costs no more than it did before
        # this was split out. See bias_batched_tail!/force.jl's cuda_graph_capturing branch for the
        # actual n_bias>1 batching this enables.
        bias_cv_step!(bias, sys, coords, fs, step_n)
        bias_batched_tail!(fs, (bias,), step_n)
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
    # d_coords (== bias.grad when persistent) is N_atoms-sized -- bias_all_finite's plain `all(...)`
    # forces a host sync on every call, which dominates a biased Langevin step's cost (confirmed by
    # profiling; see the plan this refers to). defer_finite_check routes it through
    # check_bias_finite_deferred! instead (device-resident, no sync); Langevin's step loop passes
    # defer_finite_check=true and reads the deferred result back cheaply via check_bias_finite_periodic.
    if defer_finite_check
        ensure_bias_finite_buffer!(bias)
        check_bias_finite_deferred!(d_coords, bias, step_n)
    else
        check_bias_finite(d_coords, "CV gradient", bias; cv_sim=cv_sim)
    end

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
    if defer_finite_check
        check_bias_finite_deferred!(fs_svec, bias, step_n)
    else
        check_bias_finite(
            fs_svec,
            "bias force",
            bias;
            cv_sim=cv_sim,
            max_abs_component = bias_max_abs_ustrip(fs_svec),
        )
    end

    if needs_vir && bias.cv_type.has_virial
        calculate_virial!(buffers.virial, bias.cv_type, coords, -fs_svec, sys.atoms, sys.boundary;
                          precomputed_extremum=bias.extremal_cache)
    end

    fs .-= fs_svec
    return fs
end
