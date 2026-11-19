module UnmixBatch

using SpectralUnmixing, LinearAlgebra, Base.Threads
export unmix_matrix

BLAS.set_num_threads(1)

function unmix_matrix(
    lib, rfl, unc, class_idx, options;
    mode, n_mc, num_endmembers, normalization, optimization,
    max_combinations, combination_type,
    n_cores::Integer = nthreads(),
)
    R = Matrix{Float64}(rfl)
    U = unc === nothing ? nothing : Matrix{Float64}(unc)
    size(R, 2) == size(lib.spectra, 2) ||
        throw(DimensionMismatch("pixels have $(size(R, 2)) bands, library has $(size(lib.spectra, 2))"))

    n = size(R, 1)
    k = length(lib.class_valid_keys) + 1   # classes + brightness
    e = size(lib.spectra, 1) + 1           # endmembers + brightness

    mr  = fill(NaN, n, k); mv  = fill(NaN, n, k)
    cfr = fill(NaN, n, e); cfv = fill(NaN, n, e)

    ntasks = clamp(n_cores, 1, nthreads())
    ntasks < n_cores &&
        @warn "Requested $n_cores cores but Julia has only $(nthreads()) threads"

    next = Atomic{Int}(1)
    @sync for _ in 1:ntasks
        @spawn while true
            i = atomic_add!(next, 1)   # returns the old value
            i > n && break
            r = R[i:i, :]
            all(isfinite, r) || continue
            u = U === nothing ? nothing : U[i:i, :]
            m, v, c, cv = unmix_pixel(lib, r, u, class_idx, options, mode, n_mc,
                                      num_endmembers, normalization, optimization,
                                      max_combinations, combination_type)
            mr[i, :]  = m
            cfr[i, :] = c
            v  === nothing || (mv[i, :]  = v)
            cv === nothing || (cfv[i, :] = cv)
        end
    end
    return mr, mv, cfr, cfv
end

end
