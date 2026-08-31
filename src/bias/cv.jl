# Calculate collective variables

export
    CalcMinDist,
    CalcMaxDist,
    CalcCMDist,
    CalcSingleDist,
    CalcDist,
    calculate_cv,
    cv_gradient,
    calculate_cv!,
    cv_gradient!,
    CalcRg,
    CalcRMSD,
    CalcTorsion

# Does not account for periodic boundary conditions, assumes appropriate unwrapping
function center_of_mass(coords, atoms)
    com = similar(coords, 1)
    center_of_mass!(coords, atoms, com)
    return only(from_device(com))
end

function center_of_mass!(coords, atoms, com, mass_total_buf=nothing)
    masses = mass.(atoms)
    # sum(masses; dims=1), not sum(masses): the former stays device-resident (a 1-element
    # array); the latter forces a blocking device->host sync to return a host scalar. The
    # optional mass_total_buf lets callers that also need the total mass (CalcCMDist, CalcRg)
    # reuse this reduction instead of recomputing sum(mass.(atoms)) themselves.
    mtot = sum(masses; dims=1)
    com .= sum(masses .* coords; dims=1) ./ mtot
    mass_total_buf === nothing || (mass_total_buf .= mtot)
    return nothing
end

function calculate_virial(cv, args...; kwargs...) end

function pairwise_displacement_matrix(coords_1::AbstractArray{SVector{D, C}},
                                      coords_2::AbstractArray{SVector{D, C}},
                                      calc_type,
                                      boundary) where {D, C}
    c1_col = reshape(coords_1, length(coords_1), 1)
    c2_row = reshape(coords_2, 1, length(coords_2))
    if calc_type == :closest
        return vector.(c1_col, c2_row, (boundary,))
    else
        return c2_row .- c1_col
    end
end

function pairwise_distance_matrix(coords_1, coords_2, calc_type, boundary)
    return norm.(pairwise_displacement_matrix(coords_1, coords_2, calc_type, boundary))
end

# Finds the pair (i, j) minimizing/maximizing the distance between two groups of atoms,
# using `extremum_fn = findmin`/`findmax`. Returns the indices, the extremal distance,
# and the coords_1[i] -> coords_2[j] displacement vector, all without scalar-indexing
# into `coords_1`/`coords_2` (safe for CuArray input).
#
# CPU / generic fallback: materializes the full group_a x group_b displacement matrix. Fine on
# CPU (no memory ceiling anywhere near typical group sizes, no kernel-launch-count concern); on
# GPU this is replaced below by `extremal_pair_fused`, which avoids materializing the O(Na*Nb)
# matrix entirely (see that function's docstring for why).
function extremal_pair_dense(coords_1, coords_2, calc_type, extremum_fn, boundary)
    diffs = pairwise_displacement_matrix(coords_1, coords_2, calc_type, boundary)
    dist_matrix = norm.(diffs)
    d, idx = extremum_fn(dist_matrix)
    i, j = Tuple(idx)
    r_ij = only(from_device(diffs[i:i, j:j]))
    return i, j, d, r_ij
end

# `coords_1`/`coords_2` are typically `@view coords[cv.atom_inds_1]`-style fancy-index views
# (indexed by a `Vector{Int}`, not a range) -- a `SubArray` wrapping a `CuArray` this way is *not*
# itself an `AbstractGPUArray` (only range-indexed views are), so a plain `::AbstractGPUArray`
# dispatch would silently miss the fused GPU path for exactly the call pattern `cv_gradient!`
# actually uses. Unwrap recursively via `parent` instead of dispatching on the wrapper type.
is_gpu_resident(x::AbstractGPUArray) = true
is_gpu_resident(x::SubArray) = is_gpu_resident(parent(x))
is_gpu_resident(x) = false

function extremal_pair(coords_1, coords_2, calc_type, extremum_fn, boundary)
    if is_gpu_resident(coords_1)
        return extremal_pair_fused(coords_1, coords_2, calc_type, extremum_fn, boundary)
    else
        return extremal_pair_dense(coords_1, coords_2, calc_type, extremum_fn, boundary)
    end
end

# GPU-native `extremal_pair`: avoids ever materializing the dense group_a x group_b displacement
# matrix (`extremal_pair_dense`'s approach), which is O(group_a * group_b) memory -- large groups
# (tens of thousands of atoms per side) OOM (e.g. 51200 x 51200 SVector{3,Float32} needs ~29GB).
#
# Two-pass approach: Pass 1 (one @kernel launch, one thread per row of coords_1, each thread
# scanning all of coords_2) writes only O(group_a)-sized per-row winners; Pass 2 is a plain
# findmin/findmax over that small array. Compute is still O(group_a * group_b) -- unavoidable, a
# true cutoff-free global extremum requires checking every pair -- but memory drops to O(group_a).
#
# This is the fallback used for ad-hoc calls with no persistent `MinMaxScratch` (e.g. direct
# `calculate_cv`/`cv_gradient` calls with no BiasPotential behind them). BiasPotential's usual path
# -- a persistent `MinMaxScratch` supplied and `coords` GPU-resident -- instead uses
# `mindist_reduce_kernel!`/`mindist_finalize_value_kernel!`/`mindist_finalize_grad_kernel!` below,
# which get rid of `findmin`/`findmax`'s host sync (and the two `from_device` reads after it)
# entirely via an atomic device-side reduction, while keeping the exact same row-level parallelism
# as the two-pass approach above -- see the comment above those kernels for the full design.
#
# `idx1_dev`/`idx2_dev` are device copies of `cv.atom_inds_1`/`atom_inds_2`, uploaded exactly once
# (lazily, in ensure_bias_dist_scratch!, bias.jl) and reused on every subsequent call. This matters
# because `@view coords[cv.atom_inds_1]` -- used by the generic `calculate_cv!` fallback and by
# `extremal_pair_fused` below -- re-uploads the host index Vector to a *fresh* GPU array on every
# single call (confirmed live via `CUDA.@allocated`: 40 bytes for a 5-index group, i.e. exactly
# `sizeof(Int) * 5`), a real per-call allocation independent of kernel-launch count, and one that's
# illegal inside a captured CUDA graph region regardless. `winner_i`/`winner_j`/`r_ij` cache the
# most recent call's winning pair (device-resident), letting `calculate_virial_dist!` recover it
# for `ExtremalPairCache` without redoing the O(group_a * group_b) search.
# Caps the number of parallel workers `mindist_tile_kernel!` uses (see its docstring below) --
# also the finalize kernel's serial-scan length, so this is the one knob trading finalize cost
# against tile parallelism. 4096 keeps a single-thread scan over that many candidates in the
# few-microsecond range while still giving a tiny/lopsided group (e.g. na=5, nb=100000) roughly
# 800x more concurrent workers than the old one-thread-per-row design ever could.
const MINDIST_TILE_CAP = 4096

mutable struct MinMaxScratch{IV, DV, JV, RV, SV, R1V}
    idx1_dev::IV    # device copy of cv.atom_inds_1 (uploaded once)
    idx2_dev::IV    # device copy of cv.atom_inds_2 (uploaded once)
    out_dist::DV    # O(T): tile kernel's per-worker winning distance (T = min(na*nb, MINDIST_TILE_CAP))
    out_i::JV       # O(T): tile kernel's per-worker winning row (index into group A)
    out_j::JV       # O(T): tile kernel's per-worker winning column (index into group B)
    out_disp::RV    # O(T): tile kernel's per-worker winning displacement
    winner_i::SV    # 1-element: finalize kernel's global winning row
    winner_j::SV    # 1-element: finalize kernel's global winning column
    r_ij::R1V       # 1-element: finalize kernel's global winning displacement
end

# Caches the (i, j, d, r_ij) result of an `extremal_pair` call made inside `cv_gradient!` for
# CalcMinDist/CalcMaxDist, so a subsequent `calculate_virial_dist!` call in the same timestep (on
# the same, unchanged `coords`) can reuse it instead of recomputing the O(group_a * group_b)
# extremal search a second time. Populated by BiasPotential (bias.jl), unused (kwarg default
# `nothing`) by any other caller.
mutable struct ExtremalPairCache
    valid::Bool
    i::Int
    j::Int
    d::Any
    r_ij::Any
end

@kernel inbounds=true function extremal_pair_row_kernel!(out_dist, out_j, out_disp,
                                                          @Const(coords_1), @Const(coords_2),
                                                          boundary, closest::Bool, ::Val{is_min}) where is_min
    i = @index(Global, Linear)
    if i <= length(coords_1)
        ci = coords_1[i]
        r1 = closest ? vector(ci, coords_2[1], boundary) : coords_2[1] - ci
        best_d, best_j, best_disp = norm(r1), 1, r1
        for j in 2:length(coords_2)
            rij = closest ? vector(ci, coords_2[j], boundary) : coords_2[j] - ci
            d = norm(rij)
            better = is_min ? (d < best_d) : (d > best_d)
            if better
                best_d, best_j, best_disp = d, j, rij
            end
        end
        out_dist[i] = best_d
        out_j[i] = best_j
        out_disp[i] = best_disp
    end
end

function extremal_pair_fused(coords_1, coords_2, calc_type, extremum_fn, boundary)
    na = length(coords_1)
    out_dist = similar(coords_1, eltype(eltype(coords_1)), na)
    out_j = similar(coords_1, Int, na)
    out_disp = similar(coords_1, na)

    closest = calc_type == :closest
    is_min = extremum_fn === findmin
    backend = get_backend(coords_1)
    kernel! = extremal_pair_row_kernel!(backend, min(na, 256))
    kernel!(out_dist, out_j, out_disp, coords_1, coords_2, boundary, closest, Val(is_min); ndrange=na)

    d, i = extremum_fn(out_dist)
    j = only(from_device(out_j[i:i]))
    r_ij = only(from_device(out_disp[i:i]))
    return i, j, d, r_ij
end

# --------------------------------------------------------------
# Fused path for CalcMinDist/CalcMaxDist's calculate_cv!/cv_gradient!, used whenever a persistent
# `MinMaxScratch` is supplied (BiasPotential's usual case) and `coords` is GPU-resident. Plain
# KernelAbstractions kernels throughout -- no atomics, no host syncs.
#
# Earlier version of this comment described a one-thread-per-row design: `ndrange = group_a`,
# each thread serially scanning all of group_b for its row's winner. That parallelises over
# group_a only -- fine when group_a is itself large, but the realistic case for a CV is a small,
# fixed reference group (a handful of atoms) checked against a much bigger group_b, and in that
# case group_a-many threads is nowhere near enough parallelism: e.g. na=5 gives only 5 concurrent
# workers each doing an O(nb) serial scan, using a vanishing fraction of the device regardless of
# how large nb grows. Fixed by tiling over the *flattened* na*nb pair space instead of just rows:
#  1. `mindist_tile_kernel!` -- ndrange=T=min(na*nb, MINDIST_TILE_CAP) workers, each grid-striding
#     over a disjoint slice of the full na*nb pairs (not just one row), keeping a running local
#     winner. T is capped (not just na) so parallelism now scales with the *total* amount of work
#     regardless of how lopsided group_a/group_b are, and writes only O(T)-bounded output
#     (`out_dist`/`out_i`/`out_j`/`out_disp`), not O(na).
#  2. `mindist_finalize_value_kernel!`/`mindist_finalize_grad_kernel!` -- ndrange=1, serial scan
#     over the T (<=MINDIST_TILE_CAP) worker outputs, not over na -- bounded regardless of group
#     size, unlike the old design's O(group_a) scan (that finalize kernel becoming the actual
#     bottleneck, once group_a itself grew large, is what motivated this rework).
#  3. `mindist_clear_grad_kernel!` -- ndrange=group_a+group_b, one thread per atom, clearing any
#     stale nonzero `grad` entry from a previous call's different winning pair. Previously this
#     was a serial loop folded into kernel B; same O(group) bottleneck class as (2), so it gets
#     its own parallel kernel, launched before the (now O(1)-ish) finalize kernel writes the new
#     winning pair's 2 nonzero entries.
# `calculate_cv!` needs (1)+(2) (2 launches, same count as before); `cv_gradient!` needs all of
# (1)-(3) (3 launches, one more than before -- the trade for (2) and (3) no longer scaling with
# group size). Two more things this avoids, same as previously:
#  * `extremal_pair_fused` (above) needs `findmin`/`findmax` (a host sync) plus two `from_device`
#    slice reads afterward on top of its own kernel launch; the kernels below need none.
#  * `@view coords[cv.atom_inds_1]` (used by `calculate_cv!`'s generic fallback, and by
#    `extremal_pair_fused`'s callers) re-uploads the host index Vector to a fresh GPU array on
#    *every* call (confirmed live via CUDA.@allocated). `MinMaxScratch.idx1_dev`/`idx2_dev` are
#    uploaded once and reused, so these kernels index into the *full* `coords`/`grad` arrays
#    directly and never touch `@view`.
#
# `mindist_tile_kernel!` grid-strides over the *flattened* na*nb pair space: worker `tid` (of T
# total) visits pairs `tid`, `tid+T`, `tid+2T`, ... (converted back to (i, j) via div/mod on group
# B's length), tracking a running local winner. T = min(na*nb, MINDIST_TILE_CAP) guarantees every
# worker's first iteration (k=tid<=T<=na*nb) is in range, so there's no sentinel/uninitialized-
# winner case to special-case for workers that would otherwise get 0 pairs.
@kernel inbounds=true function mindist_tile_kernel!(out_dist, out_i, out_j, out_disp, @Const(coords),
                                                     @Const(idx1), @Const(idx2), boundary,
                                                     closest::Bool, ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    T = length(out_dist)
    nb = length(idx2)
    total = length(idx1) * nb

    k = tid
    i = (k - 1) ÷ nb + 1
    j = (k - 1) % nb + 1
    ci, cj = coords[idx1[i]], coords[idx2[j]]
    r1 = closest ? vector(ci, cj, boundary) : cj - ci
    best_d, best_i, best_j, best_disp = norm(r1), i, j, r1
    k += T
    while k <= total
        i = (k - 1) ÷ nb + 1
        j = (k - 1) % nb + 1
        ci, cj = coords[idx1[i]], coords[idx2[j]]
        rij = closest ? vector(ci, cj, boundary) : cj - ci
        d = norm(rij)
        better = is_min ? (d < best_d) : (d > best_d)
        if better
            best_d, best_i, best_j, best_disp = d, i, j, rij
        end
        k += T
    end
    out_dist[tid] = best_d
    out_i[tid] = best_i
    out_j[tid] = best_j
    out_disp[tid] = best_disp
end

@kernel inbounds=true function mindist_finalize_value_kernel!(dist_val, @Const(out_dist), ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    if tid == 1
        T = length(out_dist)
        best_d = out_dist[1]
        for k in 2:T
            d = out_dist[k]
            (is_min ? (d < best_d) : (d > best_d)) && (best_d = d)
        end
        dist_val[1] = best_d
    end
end

# Clears every candidate atom's `grad` entry in parallel (one thread per atom, ndrange=na+nb) --
# split out from the old finalize kernel's serial O(na+nb) clearing loop, same reasoning as the
# tile-vs-row split above. Must run (and, being on the same backend queue, does run, by launch
# order -- see mindist_gradient_fused! below) before mindist_finalize_grad_kernel! writes the new
# winning pair's 2 nonzero entries, or it would wipe them out again.
@kernel inbounds=true function mindist_clear_grad_kernel!(grad, @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    na = length(idx1)
    z = zero(eltype(grad))
    if tid <= na
        grad[idx1[tid]] = z
    else
        grad[idx2[tid - na]] = z
    end
end

@kernel inbounds=true function mindist_finalize_grad_kernel!(grad, d_buf, winner_i, winner_j, r_ij_buf,
                                                              @Const(out_dist), @Const(out_i), @Const(out_j),
                                                              @Const(out_disp), @Const(idx1), @Const(idx2),
                                                              ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    if tid == 1
        T = length(out_dist)
        best_d, best_slot = out_dist[1], 1
        for k in 2:T
            d = out_dist[k]
            (is_min ? (d < best_d) : (d > best_d)) && ((best_d, best_slot) = (d, k))
        end
        best_i, best_j, best_r = out_i[best_slot], out_j[best_slot], out_disp[best_slot]

        d_buf[1] = best_d
        winner_i[1] = best_i
        winner_j[1] = best_j
        r_ij_buf[1] = best_r
        if best_d > zero(best_d)
            dir = best_r / best_d
            grad[idx1[best_i]] = -dir
            grad[idx2[best_j]] = dir
        end
    end
end

function mindist_calculate_cv_fused!(dist_val, scratch::MinMaxScratch, coords, boundary, closest::Bool,
                                     is_min::Val)
    backend = get_backend(coords)
    T = length(scratch.out_dist)
    kernel_a! = mindist_tile_kernel!(backend, min(T, 256))
    kernel_a!(scratch.out_dist, scratch.out_i, scratch.out_j, scratch.out_disp, coords,
             scratch.idx1_dev, scratch.idx2_dev, boundary, closest, is_min; ndrange=T)
    kernel_b! = mindist_finalize_value_kernel!(backend, 1)
    kernel_b!(dist_val, scratch.out_dist, is_min; ndrange=1)
    return nothing
end

# Launches the tile + clear + finalize kernels (zero host syncs) and, only if `extremal_cache` was
# actually supplied (BiasPotential's forces! path always supplies one for CalcMinDist/CalcMaxDist;
# ad-hoc calculate_cv/cv_gradient calls don't), does one small readback afterward to populate it
# for calculate_virial_dist!'s reuse -- see MinMaxScratch's docstring above. This readback never
# runs during CUDA graph capture: virial steps are excluded from the captured region entirely (see
# simulators.jl's check_cuda_graph_legality/step loop), so it can't reappear there even though it's
# a host sync.
function mindist_gradient_fused!(grad, d_buf, scratch::MinMaxScratch, coords, boundary, closest::Bool,
                                 is_min::Val, extremal_cache)
    backend = get_backend(coords)
    T = length(scratch.out_dist)
    kernel_a! = mindist_tile_kernel!(backend, min(T, 256))
    kernel_a!(scratch.out_dist, scratch.out_i, scratch.out_j, scratch.out_disp, coords,
             scratch.idx1_dev, scratch.idx2_dev, boundary, closest, is_min; ndrange=T)

    na, nb = length(scratch.idx1_dev), length(scratch.idx2_dev)
    kernel_clear! = mindist_clear_grad_kernel!(backend, min(na + nb, 256))
    kernel_clear!(grad, scratch.idx1_dev, scratch.idx2_dev; ndrange=na + nb)

    kernel_b! = mindist_finalize_grad_kernel!(backend, 1)
    kernel_b!(grad, d_buf, scratch.winner_i, scratch.winner_j, scratch.r_ij, scratch.out_dist, scratch.out_i,
             scratch.out_j, scratch.out_disp, scratch.idx1_dev, scratch.idx2_dev, is_min; ndrange=1)

    if extremal_cache !== nothing
        extremal_cache.valid = true
        extremal_cache.i, extremal_cache.j = only(from_device(scratch.winner_i)), only(from_device(scratch.winner_j))
        extremal_cache.d, extremal_cache.r_ij = only(from_device(d_buf)), only(from_device(scratch.r_ij))
    end
    return nothing
end

function check_calc_type(calc_type)
    if !(calc_type in (:closest, :raw))
        throw(ArgumentError("calc_type argument must be :closest or :raw, found $calc_type"))
    end
end

"""
    CalcMinDist(calc_type=:closest)

Bias the minimum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMinDist
    calc_type::Symbol

    function CalcMinDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMinDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(md, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(md::CalcMinDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    # Routed through extremal_pair (not a plain minimum(pairwise_distance_matrix(...))) so this
    # also benefits from the GPU-native fused kernel below -- i/j/r_ij are unused here, but
    # discarding them costs nothing extra since the kernel already computes them regardless.
    # (This is the ad-hoc/no-persistent-scratch path -- see calculate_cv!'s CalcMinDist/CalcMaxDist
    # override below for the single-kernel path used when a BiasPotential's MinMaxScratch exists.)
    _, _, d, _ = extremal_pair(coords_1, coords_2, md.calc_type, findmin, boundary)
    dist_val .= d
    return nothing
end

"""
    CalcMaxDist(calc_type=:closest)

Bias the maximum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMaxDist
    calc_type::Symbol

    function CalcMaxDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMaxDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(md, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(md::CalcMaxDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    _, _, d, _ = extremal_pair(coords_1, coords_2, md.calc_type, findmax, boundary)
    dist_val .= d
    return nothing
end

"""
    CalcCMDist(calc_type=:closest)

Bias the distance between the centers of mass of two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

Should generally be used with molecule unwrapping since it assumes that the atoms
within each group are in the same periodic box.
If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcCMDist
    calc_type::Symbol

    function CalcCMDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(cd::CalcCMDist, coords_1, coords_2, boundary,
                             atoms_1, atoms_2, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(cd, coords_1, coords_2, dist_val, boundary, atoms_1, atoms_2, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(cd::CalcCMDist, coords_1, coords_2, dist_val, boundary,
                              atoms_1, atoms_2, args...; kwargs...)
    com_1 = similar(coords_1, 1)
    com_2 = similar(coords_2, 1)
    center_of_mass!(coords_1, atoms_1, com_1)
    center_of_mass!(coords_2, atoms_2, com_2)
    if cd.calc_type == :closest
        dist_val .= norm.(vector.(com_1, com_2, (boundary,)))
    else
        dist_val .= norm.(com_2 .- com_1)
    end
    return nothing
end

# --------------------------------------------------------------
# Fused path for CalcCMDist's calculate_cv!/cv_gradient!, used whenever a persistent
# `CMDistScratch` is supplied (BiasPotential's usual case) and `coords` is GPU-resident.
#
# Originally this was a *single* ndrange=1 thread doing both group sums (and, for cv_gradient!,
# the per-atom gradient write) serially, on the theory that O(group_a + group_b) is cheap enough
# to not need row-parallelism. That's wrong at real GPU-relevant group sizes: a single lane doing
# a strictly serial loop gets none of the device's parallelism, so wall time grows linearly with
# group size with ~1/1000th of the GPU active -- confirmed by measurement to scale horribly
# compared to the (embarrassingly parallel) CPU loop it's supposed to beat. Fixed by splitting
# into 3 kernels, each doing the maximum useful amount of parallel work at every stage:
#   1. `cmdist_reduce_kernel!` -- ndrange=max(T1,T2) threads (T1/T2 = min(group size, 1024)), each
#      doing a grid-stride partial reduction of its group's mass/mass-weighted-position into a
#      small (<=1024-element) `partial_mass*`/`partial_wpos*` buffer. This is the real fix: turns
#      an O(group) *serial* scan into an O(group/T) *parallel* one.
#   2. `cmdist_finalize_kernel!` -- ndrange=1, but now only sums the small (<=1024-element)
#      partial-reduction buffers, not the raw group (bounded cost regardless of group size).
#      Writes `dist_val`, plus `dir_buf`/`mtot1_buf`/`mtot2_buf` (small persistent device
#      buffers, only read by step 3, so `cv_gradient!` never needs a host sync to get them there).
#   3. `cmdist_grad_write_kernel!` -- ndrange=group_a+group_b, one thread per atom, writing that
#      atom's gradient entry in parallel instead of serially (the write is O(group), same
#      class of bottleneck as the reduction, so it gets the same treatment).
# `calculate_cv!` only needs steps 1-2 (2 launches); `cv_gradient!` needs all 3. `idx1_dev`/
# `idx2_dev` (persistent device copies of cv.atom_inds_1/2, uploaded once in
# ensure_bias_dist_scratch!, bias.jl) exist for the same reason as MinMaxScratch's: `@view
# coords[cv.atom_inds_1]` re-uploads the host index Vector on every call otherwise.
mutable struct CMDistScratch{IV, MV, WV, DV, SV}
    idx1_dev::IV
    idx2_dev::IV
    partial_mass1::MV
    partial_wpos1::WV
    partial_mass2::MV
    partial_wpos2::WV
    dir_buf::DV
    mtot1_buf::SV
    mtot2_buf::SV
end

# Kept for CalcRg (below) and the CPU/no-scratch fallback path -- a single serial O(group) sum,
# fine there since Rg's group is typically not the pathological case this file's CMDist rework
# above was fixed for, and the CPU path is already an ordinary (parallel-over-cores) loop.
@inline function cmdist_com(coords, atoms, idx)
    n = length(idx)
    m1 = mass(atoms[idx[1]])
    acc, mtot = coords[idx[1]] * m1, m1
    for k in 2:n
        mk = mass(atoms[idx[k]])
        acc += coords[idx[k]] * mk
        mtot += mk
    end
    return acc / mtot, mtot
end

@kernel inbounds=true function cmdist_reduce_kernel!(pmass1, pwpos1, pmass2, pwpos2,
                                                      @Const(coords), @Const(atoms),
                                                      @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    T1 = length(pmass1)
    if tid <= T1
        na = length(idx1)
        acc, mtot = zero(eltype(pwpos1)), zero(eltype(pmass1))
        k = tid
        while k <= na
            mk = mass(atoms[idx1[k]])
            acc += coords[idx1[k]] * mk
            mtot += mk
            k += T1
        end
        pmass1[tid] = mtot
        pwpos1[tid] = acc
    end
    T2 = length(pmass2)
    if tid <= T2
        nb = length(idx2)
        acc, mtot = zero(eltype(pwpos2)), zero(eltype(pmass2))
        k = tid
        while k <= nb
            mk = mass(atoms[idx2[k]])
            acc += coords[idx2[k]] * mk
            mtot += mk
            k += T2
        end
        pmass2[tid] = mtot
        pwpos2[tid] = acc
    end
end

@kernel inbounds=true function cmdist_finalize_kernel!(dist_val, dir_buf, mtot1_buf, mtot2_buf,
                                                        @Const(pmass1), @Const(pwpos1),
                                                        @Const(pmass2), @Const(pwpos2),
                                                        boundary, closest::Bool)
    tid = @index(Global, Linear)
    if tid == 1
        T1, T2 = length(pmass1), length(pmass2)
        mtot1, wpos1 = pmass1[1], pwpos1[1]
        for k in 2:T1
            mtot1 += pmass1[k]
            wpos1 += pwpos1[k]
        end
        mtot2, wpos2 = pmass2[1], pwpos2[1]
        for k in 2:T2
            mtot2 += pmass2[k]
            wpos2 += pwpos2[k]
        end
        com1, com2 = wpos1 / mtot1, wpos2 / mtot2
        r12 = closest ? vector(com1, com2, boundary) : com2 - com1
        d = norm(r12)
        dist_val[1] = d
        # Unconditional division (no `d > 0` branch): kernel C (cmdist_grad_write_kernel! below)
        # only ever reads dir_buf inside its own `d > 0` branch, so a NaN/Inf here when d==0 is
        # never observed -- and keeping this branchless avoids a same-vs-different-units ternary
        # (r12/d is dimensionless, zero(r12) is not).
        dir_buf[1] = r12 / d
        mtot1_buf[1] = mtot1
        mtot2_buf[1] = mtot2
    end
end

@kernel inbounds=true function cmdist_grad_write_kernel!(grad, @Const(d_buf), @Const(dir_buf),
                                                          @Const(mtot1_buf), @Const(mtot2_buf),
                                                          @Const(atoms), @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    na = length(idx1)
    d = d_buf[1]
    if d > zero(d)
        dir = dir_buf[1]
        if tid <= na
            grad[idx1[tid]] = -dir * (mass(atoms[idx1[tid]]) / mtot1_buf[1])
        else
            k = tid - na
            grad[idx2[k]] = dir * (mass(atoms[idx2[k]]) / mtot2_buf[1])
        end
    else
        z = zero(eltype(grad))
        if tid <= na
            grad[idx1[tid]] = z
        else
            k = tid - na
            grad[idx2[k]] = z
        end
    end
end

"""
    CalcSingleDist(calc_type=:closest)

Bias the distance between two atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcSingleDist
    calc_type::Symbol

    function CalcSingleDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(sd::CalcSingleDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(sd, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(sd::CalcSingleDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    if length(coords_1) > 1 || length(coords_2) > 1
        throw(ArgumentError("CalcSingleDist can only be used with atom groups containing one atom"))
    end
    c1, c2 = only(from_device(coords_1)), only(from_device(coords_2))
    #c1 = coords_1
    #c2 = coords_2
    if sd.calc_type == :closest
        dist_val .= norm(vector(c1, c2, boundary))
    else
        dist_val .= norm(c2 - c1)
    end
end

"""
    CalcDist(atom_inds_1, atom_inds_2, dist_type=CalcMinDist(), correction=:pbc)

Bias the distance between two atoms or groups of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds_1`: indices of the atom(s) in the first group.
- `atom_inds_2`: indices of the atom(s) in the second group.
- `dist_type=CalcMinDist()`: type of distance to calculate.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. If using multiple atoms in
    a group, they should generally be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcDist{DT}
    atom_inds_1::Vector{Int}
    atom_inds_2::Vector{Int}
    dist_type::DT
    correction::Symbol
    has_virial::Bool

    function CalcDist(atom_inds_1, atom_inds_2, dist_type::DT=CalcMinDist(),
                      correction=:pbc, has_virial = true) where DT
        check_correction_arg(correction)
        return new{DT}(atom_inds_1, atom_inds_2, dist_type, correction, has_virial)
    end
end

"""
    calculate_cv(cv, coords, atoms, boundary, velocities; kwargs...)

Calculate the value of a collective variable (CV) with the current system state.

New CV types should implement this function.
This function does not apply the molecule correction over the boundaries; if
required, `coords` can be obtained from `unwrap_molecules` first.
The gradient of this function with respect to coordinates, used to calculate forces,
is by default calculated with automatic differentiation when Enzyme is imported.
Alternatively, the `cv_gradient` function can be defined for a new CV type.
"""
function calculate_cv(cv::CalcDist, coords, atoms, boundary, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
    return only(from_device(buff))
end

"""
    calculate_cv!(cv, coords, atoms, boundary, buff, velocities; kwargs...)

Mutating counterpart to [`calculate_cv`](@ref): writes the CV value into the preallocated
1-element `buff` instead of allocating and returning it.
"""
function calculate_cv!(cv::CalcDist, coords, atoms, boundary, buff, args...; kwargs...)
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# Fused reduce+finalize path, used whenever a persistent `MinMaxScratch` is supplied
# (BiasPotential's usual case) and `coords` is GPU-resident -- see mindist_reduce_kernel!'s
# docstring above for why this bypasses the generic method above entirely rather than just calling
# dist_between_groups! (that generic method's `@view coords[cv.atom_inds_1]` lines are exactly the
# per-call allocation this path exists to avoid). Falls back to the generic method otherwise (CPU,
# or no scratch).
function calculate_cv!(cv::CalcDist{<:Union{CalcMinDist, CalcMaxDist}}, coords, atoms, boundary, buff,
                       args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        is_min = cv.dist_type isa CalcMinDist
        mindist_calculate_cv_fused!(buff, scratch, coords, boundary, closest, Val(is_min))
        return nothing
    end
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# Single-kernel fused path, used whenever a persistent `CMDistScratch` is supplied (BiasPotential's
# usual case) and `coords` is GPU-resident -- see CMDistScratch's docstring above. Falls back to
# the generic method otherwise (CPU, or no scratch).
function calculate_cv!(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, buff, args...;
                       scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        backend = get_backend(coords)
        T1, T2 = length(scratch.partial_mass1), length(scratch.partial_mass2)
        reduce! = cmdist_reduce_kernel!(backend, min(max(T1, T2), 256))
        reduce!(scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
               coords, atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=max(T1, T2))
        finalize! = cmdist_finalize_kernel!(backend, 1)
        finalize!(buff, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
                  scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
                  boundary, closest; ndrange=1)
        return nothing
    end
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# CalcSingleDist always involves exactly 2 known-index atoms: a single-thread GPU kernel doing
# the whole computation avoids the ~6-8 separate broadcast kernel launches (each with a fixed
# ~15-20us dispatch floor, regardless of data size) that the generic broadcast-based path above
# costs on a CuArray. Indices are plain host Ints (from cv.atom_inds_1/2, never device data), so
# this needs no from_device/to_device sync at all -- genuinely a single kernel launch.
@kernel inbounds=true function single_dist_cv_kernel!(d_buf, @Const(coords), i, j, boundary,
                                                       closest::Bool)
    idx = @index(Global, Linear)
    if idx == 1
        r_ij = closest ? vector(coords[i], coords[j], boundary) : coords[j] - coords[i]
        d_buf[1] = norm(r_ij)
    end
end

@kernel inbounds=true function single_dist_cv_gradient_kernel!(grad, d_buf, @Const(coords),
                                                                i, j, boundary, closest::Bool)
    idx = @index(Global, Linear)
    if idx == 1
        r_ij = closest ? vector(coords[i], coords[j], boundary) : coords[j] - coords[i]
        d = norm(r_ij)
        d_buf[1] = d
        if d > zero(d)
            dir = r_ij / d
            grad[i] = -dir
            grad[j] = dir
        else
            z = zero(r_ij) / oneunit(d)
            grad[i] = z
            grad[j] = z
        end
    end
end

function calculate_cv!(cv::CalcDist{CalcSingleDist}, coords::AbstractGPUArray, atoms, boundary,
                       buff::AbstractGPUArray, args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    closest = cv.dist_type.calc_type == :closest
    backend = get_backend(coords)
    kernel! = single_dist_cv_kernel!(backend, 1)
    kernel!(buff, coords, i, j, boundary, closest; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the distance between two atoms.
#
# Mathematics:
# Let the coordinates of the two atoms be r_i and r_j.
# The minimum image vector from atom i to atom j is r_{ij} = r_j - r_i.
# The distance is given by d = |r_{ij}|.
# The gradients with respect to the atomic coordinates are:
# ∇_{r_i} d = -r_{ij}/d,    ∇_{r_j} d = r_{ij}/d

@doc raw"""
    cv_gradient(cv, coords, atoms, boundary, velocities; kwargs...)

Calculates the analytical gradient of a collective variable (CV) with respect to the
system coordinates.

Returns a tuple containing the gradient (as an array of vectors) and the current
value of the CV.
When Enzyme is imported this defaults to using AD, but an explicit method
can be provided for a given CV type and is defined for built-in CVs.
The AD approach should work with and without units.

Supported CV Types:
- `CalcDist`: Distance between two atoms or groups (Min, Max, Center of Mass, or Single).
- `CalcRg`: Radius of gyration of a group of atoms.
- `CalcRMSD`: Root-mean-square deviation from a reference structure using Kabsch alignment.
- `CalcTorsion`: Torsion (dihedral) angle defined by four atoms.

Allocates the gradient array and a 1-element CV-value buffer, then delegates to
[`cv_gradient!`](@ref), which writes into them in place. Call `cv_gradient!` directly
with reused buffers to avoid the per-call allocation (e.g. across repeated timesteps).
"""
function cv_gradient(cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

"""
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, velocities; kwargs...)

Mutating counterpart to [`cv_gradient`](@ref): writes the gradient into the preallocated
`grad` (same shape/backend as `coords`) and the CV value into the preallocated 1-element
`d_buf`, instead of allocating and returning them.
"""
function cv_gradient!(grad, d_buf, cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    c1 = @view coords[i:i]
    c2 = @view coords[j:j]

    r_ij = cv.dist_type.calc_type == :closest ? vector.(c1, c2, (boundary,)) : c2 .- c1
    d = norm.(r_ij)
    d_buf .= d

    mask = d .> zero(eltype(d))
    d_safe = ifelse.(mask, d, oneunit.(d))
    dir = r_ij ./ d_safe
    grad[i:i] .= ifelse.(mask, .-dir, zero(dir))
    grad[j:j] .= ifelse.(mask, dir, zero(dir))

    return nothing
end

# GPU fast path: one kernel launch (see single_dist_cv_gradient_kernel! above) instead of the
# ~6-8 broadcast kernel launches of the generic method above.
function cv_gradient!(grad::AbstractGPUArray, d_buf::AbstractGPUArray,
                      cv::CalcDist{CalcSingleDist}, coords::AbstractGPUArray, atoms, boundary,
                      args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    closest = cv.dist_type.calc_type == :closest
    backend = get_backend(coords)
    kernel! = single_dist_cv_gradient_kernel!(backend, 1)
    kernel!(grad, d_buf, coords, i, j, boundary, closest; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the minimum distance between two groups of atoms.
#
# Mathematics:
# Let A and B be two sets of atoms.
# The minimum distance is defined by the specific pair (i*, j*) ∈ A x B that 
# minimizes d_{i,j} = |r_{i,j}|.
# The gradient evaluates to zero for all atoms except i* and j*, for which it reduces 
# to the single distance gradient:
# ∇_{r_{i*}} d = -r_{i*j*}/d,    ∇_{r_{j*}} d = r_{i*j*}/d
function cv_gradient(cv::CalcDist{CalcMinDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcMinDist}, coords, atoms, boundary, args...;
                      extremal_cache=nothing, scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        mindist_gradient_fused!(grad, d_buf, scratch, coords, boundary, cv.dist_type.calc_type == :closest,
                                Val(true), extremal_cache)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    i, j, d, r_ij = extremal_pair(c1, c2, cv.dist_type.calc_type, findmin, boundary)
    d_buf .= d
    if extremal_cache !== nothing
        extremal_cache.valid, extremal_cache.i, extremal_cache.j = true, i, j
        extremal_cache.d, extremal_cache.r_ij = d, r_ij
    end

    # Clear the full static candidate set (atom_inds_1 ∪ atom_inds_2) that ANY call to this
    # function could have written to on a PREVIOUS call with this same (possibly persistent,
    # reused across steps) `grad` buffer -- the winning pair can move between calls. O(group
    # size), not O(N_atoms). Harmless-redundant when `grad` was freshly zeroed (the
    # non-persistent-buffer path).
    zg = zero(eltype(grad))
    grad[cv.atom_inds_1] .= (zg,)
    grad[cv.atom_inds_2] .= (zg,)

    if d > zero(d)
        dir = r_ij / d
        gi, gj = cv.atom_inds_1[i], cv.atom_inds_2[j]
        grad[gi:gi] .= (-dir,)
        grad[gj:gj] .= (dir,)
    end

    return nothing
end

# Computes the analytical gradient of the maximum distance between two groups of atoms.
#
# Mathematics:
# Let A and B be two sets of atoms.
# The maximum distance is defined by the specific pair (i*, j*) ∈ A x B that 
# maximizes d_{i,j} = |r_{i,j}|.
# The gradient is equivalent to the single distance gradient applied exclusively 
# to this maximizing pair.
function cv_gradient(cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, args...;
                      extremal_cache=nothing, scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        mindist_gradient_fused!(grad, d_buf, scratch, coords, boundary, cv.dist_type.calc_type == :closest,
                                Val(false), extremal_cache)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    i, j, d, r_ij = extremal_pair(c1, c2, cv.dist_type.calc_type, findmax, boundary)
    d_buf .= d
    if extremal_cache !== nothing
        extremal_cache.valid, extremal_cache.i, extremal_cache.j = true, i, j
        extremal_cache.d, extremal_cache.r_ij = d, r_ij
    end

    # See CalcMinDist's cv_gradient! for why this clear is needed with a reused `grad` buffer.
    zg = zero(eltype(grad))
    grad[cv.atom_inds_1] .= (zg,)
    grad[cv.atom_inds_2] .= (zg,)

    if d > zero(d)
        dir = r_ij / d
        gi, gj = cv.atom_inds_1[i], cv.atom_inds_2[j]
        grad[gi:gi] .= (-dir,)
        grad[gj:gj] .= (dir,)
    end

    return nothing
end

# Computes the analytical gradient of the center-of-mass distance between two groups of atoms.
#
# Mathematics:
# Let M_A and M_B be the total masses of groups A and B.
# Let R_A and R_B be their respective centers of mass, and D = |R_B - R_A|.
# Applying the chain rule through the center of mass definition, the gradients 
# for individual atoms are proportional to their fractional mass:
# ∇_{r_i} D = -(m_i/M_A) * (R_{AB}/D)    ∀ i ∈ A
# ∇_{r_j} D =  (m_j/M_B) * (R_{AB}/D)    ∀ j ∈ B
function cv_gradient(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcCMDist}, coords, atoms, boundary, args...;
                      scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        backend = get_backend(coords)
        T1, T2 = length(scratch.partial_mass1), length(scratch.partial_mass2)
        reduce! = cmdist_reduce_kernel!(backend, min(max(T1, T2), 256))
        reduce!(scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
               coords, atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=max(T1, T2))
        finalize! = cmdist_finalize_kernel!(backend, 1)
        finalize!(d_buf, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
                  scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
                  boundary, closest; ndrange=1)
        na, nb = length(scratch.idx1_dev), length(scratch.idx2_dev)
        write! = cmdist_grad_write_kernel!(backend, min(na + nb, 256))
        write!(grad, d_buf, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
              atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=na + nb)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1_buf = similar(c1, 1)
    com2_buf = similar(c2, 1)
    center_of_mass!(c1, a1, com1_buf)
    center_of_mass!(c2, a2, com2_buf)

    if cv.dist_type.calc_type == :closest
        r_12 = vector.(com1_buf, com2_buf, (boundary,))
    else
        r_12 = com2_buf .- com1_buf
    end

    d = norm.(r_12)
    d_buf .= d

    mask = d .> zero(eltype(d))
    d_safe = ifelse.(mask, d, oneunit.(d))
    dir = r_12 ./ d_safe

    m1, m2 = mass.(a1), mass.(a2)
    # sum(...; dims=1), not sum(...): stays device-resident (a 1-element array) instead of
    # forcing a blocking device->host sync -- see the same note on center_of_mass! above.
    M1, M2 = sum(m1; dims=1), sum(m2; dims=1)

    grad1 = (.-dir) .* (m1 ./ M1)
    grad2 = dir .* (m2 ./ M2)
    # @inbounds: cv.atom_inds_1/atom_inds_2 are validated at CV-construction time, always valid
    # indices into `grad` -- without it, GPUArrays' fancy-index setindex! bounds check
    # (checkindex -> all(...)) is an *extra* host sync on top of the actual write, and (found
    # directly, verifying CUDA graph capture) raises a device-side exception if this runs inside
    # a captured region at all.
    @inbounds grad[cv.atom_inds_1] = ifelse.(mask, grad1, zero.(grad1))
    @inbounds grad[cv.atom_inds_2] = ifelse.(mask, grad2, zero.(grad2))

    return nothing
end

function calculate_virial!(virial_buff, cv::CalcDist, coords, forces, atoms, boundary;
                           precomputed_extremum=nothing, kwargs...)
    calculate_virial_dist!(virial_buff, cv.dist_type, cv, coords, forces, atoms, boundary;
                           precomputed_extremum=precomputed_extremum)
end

function calculate_virial_dist!(virial_buff, dt::CalcSingleDist, cv, coords, forces, atoms, boundary;
                                kwargs...)
    i = cv.atom_inds_1[1]
    j = cv.atom_inds_2[1]
    f_i = only(from_device(forces[i:i]))
    c_i = only(from_device(coords[i:i]))
    c_j = only(from_device(coords[j:j]))

    if dt.calc_type == :closest
        r_ji = vector(c_j, c_i, boundary)
    else
        r_ji = c_i - c_j
    end

    virial_buff .+= r_ji * transpose(f_i)
end

# `precomputed_extremum`, if supplied (an `ExtremalPairCache` with `valid == true`, populated by
# a `cv_gradient!` call made moments earlier in the same timestep on the same `coords`), skips
# recomputing the O(group_a * group_b) extremal search a second time -- see ExtremalPairCache's
# docstring above.
function calculate_virial_dist!(virial_buff, dt::CalcMinDist, cv, coords, forces, atoms, boundary;
                                precomputed_extremum=nothing, kwargs...)
    if precomputed_extremum !== nothing && precomputed_extremum.valid
        r_ij = precomputed_extremum.r_ij
    else
        c1 = @view coords[cv.atom_inds_1]
        c2 = @view coords[cv.atom_inds_2]
        _, _, _, r_ij = extremal_pair(c1, c2, dt.calc_type, findmin, boundary)
    end
    r_ji = -r_ij

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcMaxDist, cv, coords, forces, atoms, boundary;
                                precomputed_extremum=nothing, kwargs...)
    if precomputed_extremum !== nothing && precomputed_extremum.valid
        r_ij = precomputed_extremum.r_ij
    else
        c1 = @view coords[cv.atom_inds_1]
        c2 = @view coords[cv.atom_inds_2]
        _, _, _, r_ij = extremal_pair(c1, c2, dt.calc_type, findmax, boundary)
    end
    r_ji = -r_ij

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcCMDist, cv, coords, forces, atoms, boundary;
                                kwargs...)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1_buf = similar(c1, 1)
    com2_buf = similar(c2, 1)
    center_of_mass!(c1, a1, com1_buf)
    center_of_mass!(c2, a2, com2_buf)
    com1, com2 = only(from_device(com1_buf)), only(from_device(com2_buf))

    if dt.calc_type == :closest
        r_12 = vector(com2, com1, boundary)
    else
        r_12 = com1 - com2
    end

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_12 * transpose(f_sum)
end

"""
    CalcRg(atom_inds=[], correction=:pbc)

Bias the radius of gyration of a group of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcRg
    atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool

    function CalcRg(atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        return new(atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, atoms, buff, args...; kwargs...)
    return only(from_device(buff))
end

# Fused path for CalcRg's calculate_cv!/cv_gradient!, used whenever a persistent `RgScratch` is
# supplied (BiasPotential's usual case) and `coords` is GPU-resident.
#
# Originally (like CMDist before its own rework, above) a single ndrange=1 thread doing the whole
# O(group) sum plus gradient write -- fine for a handful of atoms, but the same serial bottleneck
# once `group` is large. Unlike CMDist, Rg has a genuine 2-stage data dependency: the sum of
# squared deviations from the center of mass (`Isum`) needs the *already-finalized* center of
# mass as an input, so its reduction can't start until the COM reduction has fully finished --
# there's no way to collapse this into a single parallel pass the way CMDist's one independent
# sum could be. So this needs 2 reduce+finalize pairs back to back, not 1:
#   1. `rg_com_reduce_kernel!` -- ndrange=T=min(n, RG_TILE_CAP), grid-stride partial mass/
#      mass-weighted-position reduction (same shape as cmdist_reduce_kernel!'s per-group body).
#   2. `rg_com_finalize_kernel!` -- ndrange=1, serial sum over the T (bounded) partials, writes
#      `com_buf`/`mtot_buf` (device-resident, no host sync -- read directly by stage 3 below).
#   3. `rg_isum_reduce_kernel!` -- ndrange=T, grid-stride partial reduction of
#      `sum_abs2(r_k - com) * m_k` now that `com`/`mtot` are known, into `partial_isum`.
#   4. `rg_finalize_value_kernel!`/`rg_finalize_grad_kernel!` -- ndrange=1, serial sum over the T
#      `partial_isum` entries, writes the CV value (and, for the gradient path, `d_buf`).
#   5. (gradient only) `rg_grad_write_kernel!` -- ndrange=n, one thread per atom, writing that
#      atom's gradient entry in parallel (same reasoning as CMDist's grad-write kernel).
# `calculate_cv!` needs stages 1-4 (4 launches); `cv_gradient!` needs all 5. More launches than
# CMDist's fix (2/3) -- an inherent cost of the extra sequential dependency, not slack left on the
# table -- but every stage is now genuinely parallel (or an O(T)-bounded scan) regardless of group
# size, instead of one O(group) serial thread. `idx_dev` (persistent device copy of the atom
# indices used, uploaded once in ensure_bias_dist_scratch!, bias.jl -- covering the "atom_inds=[]
# means all atoms" case too, via a materialized `1:n_atoms`) avoids `@view coords[cv.atom_inds]`'s
# per-call re-upload.
const RG_TILE_CAP = 1024

mutable struct RgScratch{IV, MV, WV, IsV, CV, MtV}
    idx_dev::IV
    partial_mass::MV
    partial_wpos::WV
    partial_isum::IsV
    com_buf::CV
    mtot_buf::MtV
end

@kernel inbounds=true function rg_com_reduce_kernel!(pmass, pwpos, @Const(coords), @Const(atoms), @Const(idx))
    tid = @index(Global, Linear)
    T = length(pmass)
    n = length(idx)
    acc, mtot = zero(eltype(pwpos)), zero(eltype(pmass))
    k = tid
    while k <= n
        mk = mass(atoms[idx[k]])
        acc += coords[idx[k]] * mk
        mtot += mk
        k += T
    end
    pmass[tid] = mtot
    pwpos[tid] = acc
end

@kernel inbounds=true function rg_com_finalize_kernel!(com_buf, mtot_buf, @Const(pmass), @Const(pwpos))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pmass)
        mtot, wpos = pmass[1], pwpos[1]
        for k in 2:T
            mtot += pmass[k]
            wpos += pwpos[k]
        end
        com_buf[1] = wpos / mtot
        mtot_buf[1] = mtot
    end
end

# Two separate kernels, not one with a boundary/use_pbc flag: `calculate_cv!` (below) has no
# `boundary` available to pass at all (matches the original `rg_value_kernel!`'s signature and
# `radius_gyration`'s CPU definition, neither of which apply a PBC correction here), while
# `cv_gradient!` does and uses `vector(com, coords[k], boundary)` (matching the original
# `rg_gradient_kernel!`). That value/gradient asymmetry predates this rework and is preserved
# as-is, not "fixed", to avoid any change in observable behaviour.
@kernel inbounds=true function rg_isum_reduce_value_kernel!(pisum, @Const(coords), @Const(atoms),
                                                             @Const(idx), @Const(com_buf))
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(idx)
    com = com_buf[1]
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(coords[idx[k]] - com) * mass(atoms[idx[k]])
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rg_isum_reduce_grad_kernel!(pisum, @Const(coords), @Const(atoms),
                                                            @Const(idx), @Const(com_buf), boundary)
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(idx)
    com = com_buf[1]
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(vector(com, coords[idx[k]], boundary)) * mass(atoms[idx[k]])
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rg_finalize_value_kernel!(dist_val, @Const(pisum), @Const(mtot_buf))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        dist_val[1] = sqrt(Isum / mtot_buf[1])
    end
end

@kernel inbounds=true function rg_finalize_grad_kernel!(d_buf, @Const(pisum), @Const(mtot_buf))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        d_buf[1] = sqrt(Isum / mtot_buf[1])
    end
end

@kernel inbounds=true function rg_grad_write_kernel!(grad, @Const(d_buf), @Const(coords), @Const(atoms),
                                                      @Const(idx), @Const(com_buf), @Const(mtot_buf), boundary)
    tid = @index(Global, Linear)
    rg = d_buf[1]
    if rg > zero(rg)
        factor = 1 / (mtot_buf[1] * rg)
        grad[idx[tid]] = factor * mass(atoms[idx[tid]]) * vector(com_buf[1], coords[idx[tid]], boundary)
    else
        grad[idx[tid]] = zero(eltype(grad))
    end
end

function calculate_cv!(cv::CalcRg, coords, atoms, buff, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        T = length(scratch.partial_mass)
        reduce_com! = rg_com_reduce_kernel!(backend, min(T, 256))
        reduce_com!(scratch.partial_mass, scratch.partial_wpos, coords, atoms, scratch.idx_dev; ndrange=T)
        finalize_com! = rg_com_finalize_kernel!(backend, 1)
        finalize_com!(scratch.com_buf, scratch.mtot_buf, scratch.partial_mass, scratch.partial_wpos; ndrange=1)
        reduce_isum! = rg_isum_reduce_value_kernel!(backend, min(T, 256))
        reduce_isum!(scratch.partial_isum, coords, atoms, scratch.idx_dev, scratch.com_buf; ndrange=T)
        finalize_val! = rg_finalize_value_kernel!(backend, 1)
        finalize_val!(buff, scratch.partial_isum, scratch.mtot_buf; ndrange=1)
        return nothing
    end
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    buff .= radius_gyration(coords_used, atoms_used)
    return nothing
end

# Computes the analytical gradient of the radius of gyration.
#
# Mathematics:
# The mass-weighted radius of gyration is:
# R_g = sqrt( (1/M) * Σ_k m_k |r_k - R_COM|^2 )
# 
# Differentiating with respect to the coordinates of atom k yields:
# ∇_{r_k} R_g = [m_k / (M * R_g)] * (r_k - R_COM)
#
# Note: The derivative of the center of mass R_COM with respect to r_k cancels out
# in the summation due to the definition of the center of mass.
function cv_gradient(cv::CalcRg, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcRg, coords, atoms, boundary, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        T = length(scratch.partial_mass)
        reduce_com! = rg_com_reduce_kernel!(backend, min(T, 256))
        reduce_com!(scratch.partial_mass, scratch.partial_wpos, coords, atoms, scratch.idx_dev; ndrange=T)
        finalize_com! = rg_com_finalize_kernel!(backend, 1)
        finalize_com!(scratch.com_buf, scratch.mtot_buf, scratch.partial_mass, scratch.partial_wpos; ndrange=1)
        reduce_isum! = rg_isum_reduce_grad_kernel!(backend, min(T, 256))
        reduce_isum!(scratch.partial_isum, coords, atoms, scratch.idx_dev, scratch.com_buf, boundary; ndrange=T)
        finalize_grad! = rg_finalize_grad_kernel!(backend, 1)
        finalize_grad!(d_buf, scratch.partial_isum, scratch.mtot_buf; ndrange=1)
        n = length(scratch.idx_dev)
        write! = rg_grad_write_kernel!(backend, min(n, 256))
        write!(grad, d_buf, coords, atoms, scratch.idx_dev, scratch.com_buf, scratch.mtot_buf, boundary; ndrange=n)
        return nothing
    end

    atom_inds_used = iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds
    c_used = @view coords[atom_inds_used]
    a_used = @view atoms[atom_inds_used]

    com_buf = similar(c_used, 1)
    center_of_mass!(c_used, a_used, com_buf)
    m_used = mass.(a_used)
    # sum(...; dims=1), not sum(...): stays device-resident -- see the center_of_mass! note above.
    M_total = sum(m_used; dims=1)

    r_ic_all = vector.(com_buf, c_used, (boundary,))
    rg_sq = sum(sum_abs2.(r_ic_all) .* m_used; dims=1) ./ M_total
    rg = sqrt.(rg_sq)
    d_buf .= rg

    # Device-resident masked write (mirrors CalcDist{CalcSingleDist}'s ifelse-masking above)
    # instead of a host `if rg > zero(rg)` branch -- `rg` is now a 1-element device array, so a
    # host branch on it would force a sync. This also fixes a stale-buffer hazard for free (same
    # cost either way, since it's one fused broadcast regardless of the mask outcome): with a
    # *reused* grad buffer, the old host-branch version left a previous call's stale gradient at
    # `atom_inds_used` on a degenerate (rg == 0) step, since the unconditional-false branch never
    # wrote anything; this version always writes, zero on the degenerate branch.
    mask = rg .> zero(eltype(rg))
    rg_safe = ifelse.(mask, rg, oneunit.(rg))
    inv_factor = 1 ./ (M_total .* rg_safe)
    factor = ifelse.(mask, inv_factor, zero.(inv_factor))
    grad[atom_inds_used] .= factor .* m_used .* r_ic_all

    return nothing
end

# For Rg and also for the RMSD the forces applied to the atoms 
# are dependent only on the relative configuration of said
# atoms, making them translationally invariant. Therefore:
#
# Σ F_i = 0
#
# We can exploit this fact to obtain the virial by computing 
#
# Ξ = Σ (r_i - r_COM) ⊗ F_i; 
#
# rearranging:
#
# Ξ = Σ ( r_i ⊗ F_i ) - r_COM ⊗ Σ F_i = Σ r_i ⊗ F_i
#
# which is equivalent to the standard definition of the virial!
# Note: we cannot just compute Σ r_i ⊗ F_i as this will give 
# different results depending on the choice of origin of coordinates.

function calculate_virial!(virial_buff, cv::CalcRg, coords, forces, atoms, boundary; kwargs...)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    a_used = @view atoms[ids]

    # Calculate Center of Mass of the group to define relative coordinates
    com_buf = similar(c_used, 1)
    center_of_mass!(c_used, a_used, com_buf)

    # Accumulate sum( (r_i - r_com) * F_i^T )
    r_ic_all = vector.(com_buf, c_used, (boundary,))
    virial_buff .+= sum(r_ic_all .* transpose.(f_used))
end

"""
    CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc)

Bias the root-mean-square deviation (RMSD) between the coordinates of a group of atoms
and a set of reference coordinates.

Given as an argument to [`BiasPotential`](@ref).
The two sets of coordinates are superimposed using the Kabsch algorithm.

# Arguments
- `ref_coords`: reference coordinates. Should be constructed with an array type matching the
    `System` this CV will be used with (a plain `Array` for CPU, or the same GPU array type,
    e.g. `CuArray`/`ROCArray`, as the system's coordinates for GPU) — this is not converted
    automatically, the same convention already implicitly expected of `atoms`/`coords`/
    `velocities` elsewhere.
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `ref_atom_inds=[]`: indices of the reference coordinates to use, `[]` uses all coordinates.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcRMSD{RC}
    ref_coords::RC
    atom_inds::Vector{Int}
    ref_atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool

    function CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        RC = typeof(ref_coords)
        new{RC}(ref_coords, atom_inds, ref_atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcRMSD, coords, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, buff, args...; kwargs...)
    return only(from_device(buff))
end

# Persistent scratch for CalcRMSD's calculate_cv!/cv_gradient!, used whenever supplied
# (BiasPotential's usual case) and `coords` is GPU-resident. Unlike the other CV types above, the
# Kabsch alignment inside `rmsd`/`kabsch_deviations` (analysis.jl) always needs a host LAPACK SVD
# -- fundamentally, permanently host-sync-bound, no GPU alternative -- so this doesn't chase that
# cost down to zero. What it does remove:
#  1. `rmsd_coords`'s `coords[atom_inds_used]` re-*allocates* a gathered copy *and* re-uploads
#     `atom_inds_used` on every call (same `@view`/fancy-index cost as everywhere else in this
#     file). `idx_dev`/`coords_used` replace it with a persistent index array plus a single
#     `gather_kernel!` launch into a reused buffer.
#  2. `cv.ref_coords[ref_atom_inds_used]` redundantly re-slices data that can never change after
#     construction -- computed once, here, as `ref_coords_used`.
#  3. `kabsch_rotation_nograd` (analysis.jl) syncs *both* coordinate sets to host every call, but
#     the reference side is exactly as constant as (2) -- `ref_kabsch` precomputes
#     `kabsch_centered(ref_coords_used)` once, here, so every subsequent call's `cached_1=` kwarg
#     skips host-syncing the reference a second (and third, and...) time.
#  4. Everything *after* the SVD -- `kabsch_deviations`'s broadcast, `mean(sum_abs2, diffs)` (a
#     second host sync in its own right: `mean` with no `dims=` on a device array), and the
#     `grad[atom_inds_used] = ...` scatter write -- used to be 3 separate device broadcast kernels
#     plus that second host sync. `rmsd_isum_reduce_kernel!` + `rmsd_finalize_*_kernel!` +
#     (gradient only) `rmsd_grad_write_kernel!` below do all of it device-side once `rot`/
#     `trans_1`/`trans_2` (host scalars, from the SVD step) are known, so the *only* unavoidable
#     sync left is the SVD's own `from_device` on the current (non-reference) coordinates.
#
#     Originally this was one more single ndrange=1 thread doing the whole O(group) sum (and, for
#     cv_gradient!, the gradient write) serially -- same disease as CMDist/Rg had, fixed the same
#     way: `rmsd_isum_reduce_kernel!` grid-strides over T=min(group, RMSD_TILE_CAP) workers into
#     `partial_isum`; `rmsd_finalize_value_kernel!`/`rmsd_finalize_grad_kernel!` serially sum only
#     those T (bounded) partials; `rmsd_grad_write_kernel!` (gradient only) writes each atom's
#     gradient entry in parallel (ndrange=group) instead of serially. No separate "reduce the
#     alignment target" stage is needed here the way Rg needed one for its center of mass --
#     `rot`/`trans_1`/`trans_2` are already fully known (from the host-side SVD) before any of
#     these kernels run, so this is a 1-stage reduction, not Rg's 2-stage one.
const RMSD_TILE_CAP = 1024

mutable struct RmsdScratch{IV, CV, RCV, KV, PV}
    idx_dev::IV
    coords_used::CV
    ref_coords_used::RCV
    ref_kabsch::KV
    partial_isum::PV
end

@kernel inbounds=true function gather_kernel!(dst, @Const(src), @Const(idx))
    k = @index(Global, Linear)
    if k <= length(idx)
        dst[k] = src[idx[k]]
    end
end

@kernel inbounds=true function rmsd_isum_reduce_kernel!(pisum, @Const(ref_used), @Const(coords_used),
                                                         rot, trans_1, trans_2)
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(ref_used)
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(rot * (ref_used[k] - trans_1) - (coords_used[k] - trans_2))
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rmsd_finalize_value_kernel!(dist_val, @Const(pisum), n)
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        dist_val[1] = sqrt(Isum / n)
    end
end

@kernel inbounds=true function rmsd_finalize_grad_kernel!(d_buf, @Const(pisum), n)
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        d_buf[1] = sqrt(Isum / n)
    end
end

@kernel inbounds=true function rmsd_grad_write_kernel!(grad, @Const(d_buf), @Const(ref_used), @Const(coords_used),
                                                        @Const(idx), rot, trans_1, trans_2)
    tid = @index(Global, Linear)
    rmsd_val = d_buf[1]
    if rmsd_val > zero(rmsd_val)
        n = length(idx)
        factor = 1 / (n * rmsd_val)
        diff_k = rot * (ref_used[tid] - trans_1) - (coords_used[tid] - trans_2)
        grad[idx[tid]] = -factor * diff_k
    else
        grad[idx[tid]] = zero(eltype(grad))
    end
end

function calculate_cv!(cv::CalcRMSD, coords, buff, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        n = length(scratch.idx_dev)
        kernel! = gather_kernel!(backend, min(n, 256))
        kernel!(scratch.coords_used, coords, scratch.idx_dev; ndrange=n)
        rot, trans_1, trans_2 = kabsch_rotation_nograd(scratch.ref_coords_used, scratch.coords_used;
                                                        cached_1=scratch.ref_kabsch)
        T = length(scratch.partial_isum)
        reduce! = rmsd_isum_reduce_kernel!(backend, min(T, 256))
        reduce!(scratch.partial_isum, scratch.ref_coords_used, scratch.coords_used, rot, trans_1, trans_2; ndrange=T)
        finalize! = rmsd_finalize_value_kernel!(backend, 1)
        finalize!(buff, scratch.partial_isum, n; ndrange=1)
        return nothing
    end
    coords_used, ref_coords_used = rmsd_coords(cv, coords)
    buff .= rmsd(ref_coords_used, coords_used)
    return nothing
end

# Select the atoms of the system and of the reference used by a CalcRMSD collective variable
function rmsd_coords(cv::CalcRMSD, coords)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    ref_atom_inds_used = (iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords)
                                                           : cv.ref_atom_inds)
    return coords[atom_inds_used], cv.ref_coords[ref_atom_inds_used]
end

function calculate_cv_ustrip!(unit_arr, args...)
    cv = calculate_cv(args...)
    # Enzyme requires a unitless value to be returned
    # We strip the unit, store it and add it back on later
    unit_arr[1] = unit(cv)
    return ustrip(cv)
end

# Computes the analytical gradient of the optimal Root-Mean-Square Deviation (RMSD)
# using Kabsch alignment.
#
# Mathematics:
# Let r_k^{sys} be the current system coordinates and R_COM^{sys} be their centroid. 
# Let r_k^{ref} be the centered reference coordinates.
# The optimally aligned RMSD distance is:
# d_{RMSD} = sqrt( (1/N) * Σ_{k=1}^N |(r_k^{sys} - R_COM^{sys}) - Q r_k^{ref}|^2 )
#
# Because the rotation matrix Q optimally minimizes the distance, the derivative of Q 
# with respect to coordinates vanishes.
# The exact analytical gradient for an evaluated atom k simplifies to:
# ∇_{r_k} d_{RMSD} = [1 / (N * d_{RMSD})] * ((r_k^{sys} - R_COM^{sys}) - Q r_k^{ref})
function cv_gradient(cv::CalcRMSD, coords, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcRMSD, coords, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        n = length(scratch.idx_dev)
        kernel! = gather_kernel!(backend, min(n, 256))
        kernel!(scratch.coords_used, coords, scratch.idx_dev; ndrange=n)
        rot, trans_1, trans_2 = kabsch_rotation_nograd(scratch.ref_coords_used, scratch.coords_used;
                                                        cached_1=scratch.ref_kabsch)
        T = length(scratch.partial_isum)
        reduce! = rmsd_isum_reduce_kernel!(backend, min(T, 256))
        reduce!(scratch.partial_isum, scratch.ref_coords_used, scratch.coords_used, rot, trans_1, trans_2; ndrange=T)
        finalize! = rmsd_finalize_grad_kernel!(backend, 1)
        finalize!(d_buf, scratch.partial_isum, n; ndrange=1)
        write! = rmsd_grad_write_kernel!(backend, min(n, 256))
        write!(grad, d_buf, scratch.ref_coords_used, scratch.coords_used, scratch.idx_dev,
              rot, trans_1, trans_2; ndrange=n)
        return nothing
    end

    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used, ref_c_used = rmsd_coords(cv, coords)
    N = length(c_used)

    # Deviations of the rotated reference from the current coordinates
    diffs = kabsch_deviations(ref_c_used, c_used)
    rmsd_val = sqrt(mean(sum_abs2, diffs))
    d_buf .= rmsd_val

    if rmsd_val > zero(rmsd_val)
        factor = 1 / (N * rmsd_val)
        grad[atom_inds_used] = (-factor,) .* diffs
    end

    return nothing
end

function calculate_virial!(virial_buff, cv::CalcRMSD, coords, forces, atoms, boundary; kwargs...)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    
    # RMSD with centering is translationally invariant.
    # We use the centroid of the current configuration as the reference point.
    # sum/length rather than mean(): mean() triggers scalar indexing (via
    # first()) when c_used is a @view of a CuArray
    com = sum(c_used) / length(c_used)

    # Accumulate sum( (r_i - r_centroid) * F_i^T )
    r_ic_all = vector.((com,), c_used, (boundary,))
    virial_buff .+= sum(r_ic_all .* transpose.(f_used))
end

"""
    CalcTorsion(atom_inds::AbstractVector{Int}=[], correction=:pbc, has_virial::Bool=true;
                gradient_singularity_tol=1e-6)

A collective variable that calculates the torsion angle (dihedral) defined by four atoms.

The angle is defined by the intersection of the planes formed by atoms (i, j, k) and (j, k, l), where the indices are given by `atom_inds`.
The torsion gradient is regularized near collinear geometries using
`gradient_singularity_tol`, a dimensionless relative tolerance applied to the
bond-vector norms.

# Fields
- `atom_inds::AbstractVector{Int}`: The indices of the four atoms (i, j, k, l) defining the torsion.
- `correction::Symbol`: The method used to handle periodic boundary conditions. Defaults to `:pbc`.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native bonded-topology
    traversal.
- `has_virial::Bool`: Whether the virial contribution should be calculated for this collective variable. Defaults to `true`.
- `gradient_singularity_tol::Float64`: Relative tolerance used to cap torsion gradients near collinear geometries.
"""
struct CalcTorsion
    atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool
    gradient_singularity_tol::Float64

    function CalcTorsion(atom_inds=[], correction=:pbc, has_virial=true;
                         gradient_singularity_tol=1e-6)
        check_correction_arg(correction)
        tol = Float64(gradient_singularity_tol)
        if !isfinite(tol) || tol <= 0
            throw(ArgumentError("gradient_singularity_tol must be finite and positive, got $(gradient_singularity_tol)."))
        end
        return new(atom_inds, correction, has_virial, tol)
    end
end

function calculate_cv(cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    buff = similar(coords, FT, 1)
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
    return only(from_device(buff))
end

function calculate_cv!(cv::CalcTorsion, coords, atoms, boundary, buff, args...; kwargs...)
    pts = from_device(coords[cv.atom_inds])
    buff .= torsion_angle(pts[1], pts[2], pts[3], pts[4], boundary)
    return nothing
end

# CalcTorsion always involves exactly 4 known-index atoms, same as CalcSingleDist's 2 -- a
# single-thread GPU kernel avoids the from_device host sync the generic method above pays on
# every call (not just multiple launches: `coords[cv.atom_inds]` there is a full device->host
# sync, since `pts` is then used as plain host StaticArrays values).
@kernel inbounds=true function torsion_cv_kernel!(d_buf, @Const(coords), i, j, k, l, boundary)
    idx = @index(Global, Linear)
    if idx == 1
        d_buf[1] = torsion_angle(coords[i], coords[j], coords[k], coords[l], boundary)
    end
end

function calculate_cv!(cv::CalcTorsion, coords::AbstractGPUArray, atoms, boundary,
                       buff::AbstractGPUArray, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    backend = get_backend(coords)
    kernel! = torsion_cv_kernel!(backend, 1)
    kernel!(buff, coords, i, j, k, l, boundary; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the torsion (dihedral) angle defined by four atoms.
#
# Mathematics:
# Let the four atoms be i, j, k, l. Define bond vectors: 
# b_1 = r_j - r_i,  b_2 = r_k - r_j,  b_3 = r_l - r_k.
# Define normal vectors to the planes: 
# m = b_1 x b_2,    n = b_2 x b_3. 
# 
# The gradients are evaluated via the chain rule on:
# ϕ = atan2( |b_2|(b_1 · n), m · n )
#
# This yields:
# ∇_{r_i} ϕ =   (|b_2| / |m|^2) * m
# ∇_{r_l} ϕ = - (|b_2| / |n|^2) * n
# ∇_{r_j} ϕ = - (1 + (b_1 · b_2)/|b_2|^2) * ∇_{r_i} ϕ + ((b_2 · b_3)/|b_2|^2) * ∇_{r_l} ϕ
# ∇_{r_k} ϕ =   ((b_1 · b_2)/|b_2|^2) * ∇_{r_i} ϕ - (1 + (b_2 · b_3)/|b_2|^2) * ∇_{r_l} ϕ
function check_torsion_bond_norm(norm_value, label::AbstractString)
    if !isfinite(ustrip(norm_value)) || norm_value <= zero(norm_value)
        throw(ArgumentError("CalcTorsion cannot compute a finite gradient because $(label) " *
                            "has non-positive or non-finite length ($(norm_value))."))
    end
    return norm_value
end

function cv_gradient(cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords)) / oneunit(eltype(eltype(coords)))
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    d_buf = similar(coords, FT, 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    pts = from_device(coords[[i, j, k, l]])
    ri, rj, rk, rl = pts[1], pts[2], pts[3], pts[4]

    b1 = vector(ri, rj, boundary)
    b2 = vector(rj, rk, boundary)
    b3 = vector(rk, rl, boundary)

    m = cross(b1, b2)
    n = cross(b2, b3)

    b1_norm = check_torsion_bond_norm(norm(b1), "bond i-j")
    b2_norm = check_torsion_bond_norm(norm(b2), "bond j-k")
    b3_norm = check_torsion_bond_norm(norm(b3), "bond k-l")
    FT = typeof(float(ustrip(b2_norm)))
    tol = FT(cv.gradient_singularity_tol)
    length_scale = max(b1_norm, b2_norm, b3_norm)
    norm_floor = tol * length_scale
    b1_norm_eff = max(b1_norm, norm_floor)
    b2_norm_eff = max(b2_norm, norm_floor)
    b3_norm_eff = max(b3_norm, norm_floor)

    m_sq = sum(abs2, m)
    n_sq = sum(abs2, n)
    b2_sq = b2_norm^2
    m_sq_eff = max(m_sq, (tol * b1_norm_eff * b2_norm_eff)^2)
    n_sq_eff = max(n_sq, (tol * b2_norm_eff * b3_norm_eff)^2)
    b2_sq_eff = max(b2_sq, b2_norm_eff^2)

    phi = torsion_angle(ri, rj, rk, rl, boundary)
    d_buf .= phi

    grad_i =  (b2_norm_eff / m_sq_eff) * m
    grad_l = -(b2_norm_eff / n_sq_eff) * n

    b1_dot_b2 = dot(b1, b2)
    b3_dot_b2 = dot(b3, b2)

    grad_j = -(1 + b1_dot_b2 / b2_sq_eff) * grad_i + (b3_dot_b2 / b2_sq_eff) * grad_l
    grad_k = (b1_dot_b2 / b2_sq_eff) * grad_i - (1 + b3_dot_b2 / b2_sq_eff) * grad_l

    grad[[i, j, k, l]] = -[grad_i, grad_j, grad_k, grad_l]

    return nothing
end

# GPU fast path: one kernel launch instead of the from_device host sync + host StaticArrays math
# the generic method above pays on every call. Mirrors single_dist_cv_gradient_kernel! above.
#
# check_torsion_bond_norm's CPU throw (ArgumentError on degenerate bond-length geometry) has no
# GPU equivalent -- kernels can't throw catchable exceptions. Rather than silently floor and
# succeed (which would produce a plausible-but-wrong gradient, diverging from the CPU path's
# fail-loud behaviour), the degenerate case writes NaN into the affected atoms' gradient instead.
# check_bias_finite (src/bias/bias.jl) already errors on a non-finite CV gradient every step in
# the normal (non-graph-capture) path, so this still fails loud -- via the existing finite-check
# mechanism instead of a `throw`.
@kernel inbounds=true function torsion_cv_gradient_kernel!(grad, d_buf, @Const(coords),
                                                            i, j, k, l, boundary, tol)
    idx = @index(Global, Linear)
    if idx == 1
        ri, rj, rk, rl = coords[i], coords[j], coords[k], coords[l]
        b1 = vector(ri, rj, boundary)
        b2 = vector(rj, rk, boundary)
        b3 = vector(rk, rl, boundary)
        m = cross(b1, b2)
        n = cross(b2, b3)
        b1n, b2n, b3n = norm(b1), norm(b2), norm(b3)
        d_buf[1] = torsion_angle(ri, rj, rk, rl, boundary)

        degenerate = !isfinite(ustrip(b1n)) || b1n <= zero(b1n) ||
                     !isfinite(ustrip(b2n)) || b2n <= zero(b2n) ||
                     !isfinite(ustrip(b3n)) || b3n <= zero(b3n)
        if degenerate
            nan_s = NaN / oneunit(b2n)   # same unit-attaching idiom as `zero(r_ij) / oneunit(d)` above
            nan_v = SVector(nan_s, nan_s, nan_s)
            grad[i] = nan_v; grad[j] = nan_v; grad[k] = nan_v; grad[l] = nan_v
        else
            length_scale = max(b1n, b2n, b3n)
            norm_floor = tol * length_scale
            b1n_eff = max(b1n, norm_floor)
            b2n_eff = max(b2n, norm_floor)
            b3n_eff = max(b3n, norm_floor)

            m_sq_eff = max(sum(abs2, m), (tol * b1n_eff * b2n_eff)^2)
            n_sq_eff = max(sum(abs2, n), (tol * b2n_eff * b3n_eff)^2)
            b2_sq_eff = max(b2n^2, b2n_eff^2)

            grad_i =  (b2n_eff / m_sq_eff) * m
            grad_l = -(b2n_eff / n_sq_eff) * n

            b1_dot_b2 = dot(b1, b2)
            b3_dot_b2 = dot(b3, b2)

            grad_j = -(1 + b1_dot_b2 / b2_sq_eff) * grad_i + (b3_dot_b2 / b2_sq_eff) * grad_l
            grad_k =  (b1_dot_b2 / b2_sq_eff) * grad_i - (1 + b3_dot_b2 / b2_sq_eff) * grad_l

            grad[i] = -grad_i; grad[j] = -grad_j; grad[k] = -grad_k; grad[l] = -grad_l
        end
    end
end

function cv_gradient!(grad::AbstractGPUArray, d_buf::AbstractGPUArray, cv::CalcTorsion,
                      coords::AbstractGPUArray, atoms, boundary, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    tol = FT(cv.gradient_singularity_tol)
    backend = get_backend(coords)
    kernel! = torsion_cv_gradient_kernel!(backend, 1)
    kernel!(grad, d_buf, coords, i, j, k, l, boundary, tol; ndrange=1)
    return nothing
end

function calculate_virial!(virial_buff, cv::CalcTorsion, coords, forces, atoms, boundary; kwargs...)
    ids = cv.atom_inds
    pts = from_device(coords[ids])
    fs = from_device(forces[ids])
    c1, c2, c3, c4 = pts[1], pts[2], pts[3], pts[4]
    f1, f3, f4 = fs[1], fs[3], fs[4]
    r_ji = vector(c2, c1, boundary) # r_i - r_j
    r_jk = vector(c2, c3, boundary) # r_k - r_j
    r_jl = vector(c2, c4, boundary) # r_l - r_j

    virial_buff .+= r_ji * transpose(f1) +
                    r_jk * transpose(f3) +
                    r_jl * transpose(f4)
end

# --------------------------------------------------------------
# Persistent-buffer support for BiasPotential (src/bias/bias.jl).
#
# Not every CV type has a buffer-writing calculate_cv!/cv_gradient! -- a custom, user-defined CV
# type that only implements calculate_cv falls back to the generic Enzyme-AD cv_gradient
# (ext/MollyEnzymeExt.jl), which has no buffer-writing equivalent. This trait, computed once at
# BiasPotential construction (no coords needed), gates BiasPotential's persistent-buffer path to
# exactly the CV types below; everything else keeps using the allocating calculate_cv/cv_gradient
# wrappers unchanged.
uses_builtin_cv_gradient!(::CalcDist) = true
uses_builtin_cv_gradient!(::CalcRg) = true
uses_builtin_cv_gradient!(::CalcRMSD) = true
uses_builtin_cv_gradient!(::CalcTorsion) = true
uses_builtin_cv_gradient!(::Any) = false

# Buffer-shape helpers, deduplicating the grad/d_buf allocation pattern repeated across every
# allocating cv_gradient/calculate_cv wrapper above. Used both by those wrappers and by
# BiasPotential's lazy persistent-buffer initialization (bias.jl).
zero_cv_grad_buffer(cv, coords)  = ustrip_vec.(zero(coords))
zero_cv_value_buffer(cv, coords) = similar(coords, eltype(eltype(coords)), 1)
# CalcTorsion's CV value/gradient are unitless (an angle), unlike the other CV types
# (length-valued).
zero_cv_grad_buffer(cv::CalcTorsion, coords) =
    ustrip_vec.(zero(coords)) / oneunit(eltype(eltype(coords)))
zero_cv_value_buffer(cv::CalcTorsion, coords) =
    similar(coords, typeof(float(ustrip(oneunit(eltype(eltype(coords)))))), 1)
zero_cv_gradient_buffers(cv, coords) = (zero_cv_grad_buffer(cv, coords), zero_cv_value_buffer(cv, coords))

# `calculate_cv!`'s positional-argument prefix before `buff` varies by CV type (CalcRg omits
# `boundary`; CalcRMSD omits both `atoms` and `boundary`) -- `buff` is a *required named*
# parameter for all of them, unlike `cv_gradient!`/`calculate_cv`, where any extra positional
# args are absorbed harmlessly by a trailing `args...`, so a uniform `(cv, coords, atoms,
# boundary, buff, args...)` call would silently misassign `buff`'s slot for CalcRg/CalcRMSD.
# This gives callers that need to invoke `calculate_cv!` generically (BiasPotential; also used
# the same way in Julia_Benchmark/Profile_CVs.jl) one uniform calling convention.
calculate_cv_buffered!(cv::CalcRMSD, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, buff; kwargs...)
calculate_cv_buffered!(cv::CalcRg, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, atoms, buff; kwargs...)
calculate_cv_buffered!(cv, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
