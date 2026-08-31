# The MRISubspaceRecon.jl lane: A^H y, kernel creation, kernel application.
#
# Run as a subprocess by the benchmark harness; prints its phase timings on a
# line beginning "BENCHMARK " for the parent to read. The parent watches what
# the process costs in host and device memory, so nothing is measured here but
# time.
#
#   julia --project=. lane.jl <prepared.npz> <size> <cpu|cuda> [repeats]

using NPZ
using JSON
using LinearAlgebra
using MRISubspaceRecon

const path = ARGS[1]
const size_n = parse(Int, ARGS[2])
const device = ARGS[3]
const repeats = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 5
const maps_path = length(ARGS) > 4 ? ARGS[5] : ""

held = npzread(path)
# numpy (coils, frames, shots, points) -> (samples, frames, coils), shot fastest
kspace = permutedims(held["kspace"], (3, 4, 2, 1))
Ncoil = size(kspace, 4)
Nframe = size(kspace, 3)
Nsamp = size(kspace, 1) * size(kspace, 2)
data = reshape(kspace, Nsamp, Nframe, Ncoil)

# numpy (frames, shots, points, 3) -> (3, samples, frames), same sample order
trj = reshape(permutedims(held["trajectory"], (4, 2, 3, 1)), 3, Nsamp, Nframe)
dcf = reshape(permutedims(held["density"], (2, 3, 1)), Nsamp, Nframe)
U = Float32.(real.(held["basis"]))   # (frames, rank); real by construction, see prepare.py

img_shape = (size_n, size_n, size_n)
println("data $(size(data))  trj $(size(trj))  U $(size(U))  -> $(img_shape)")

# The same sensitivities the other lane is given, so neither is measured
# against a calibration fitted to its own conventions.
stacked = npzread(maps_path)["maps"]          # (coils, nx, ny, nz)
cmaps = [Array{ComplexF32}(stacked[c, :, :, :]) for c in 1:Ncoil]

# The trajectory is already in (-0.5, 0.5), which is what this package wants.
# The weights are applied to the whole concatenated sample vector at once, in
# the same order the trajectory is flattened: sample fastest, frame slowest.
weights = vec(dcf)

if device == "cuda"
    using CUDA
    data = CuArray(data)
    trj = CuArray(trj)
    weights = CuArray(weights)
    cmaps = [CuArray(c) for c in cmaps]
    # The basis is read inside a GPU kernel, so it has to be there too.
    U = CuArray(U)
end

sync() = device == "cuda" ? CUDA.synchronize() : nothing
seconds = Dict{String,Float64}()

sync(); t0 = time()
xbp = calculate_backprojection(data, trj, cmaps; U=U, density_compensation=weights)
sync(); seconds["adjoint"] = time() - t0
xbp = nothing
GC.gc()

sync(); t0 = time()
A = NFFTNormalOp(img_shape, trj, U; cmaps=cmaps)
sync(); seconds["create"] = time() - t0

# One coil per application, which is the unit the other lane measures too.
Ncoef = size(U, 2)
x = zeros(ComplexF32, prod(img_shape) * Ncoef)
if device == "cuda"
    x = CuArray(x)
end
y = similar(x)
mul!(y, A, x)                          # warm
sync(); t0 = time()
for _ in 1:repeats
    mul!(y, A, x)
end
sync(); seconds["apply"] = (time() - t0) / repeats

println("BENCHMARK " * JSON.json(Dict("seconds" => seconds, "extra" => Dict{String,Float64}())))
