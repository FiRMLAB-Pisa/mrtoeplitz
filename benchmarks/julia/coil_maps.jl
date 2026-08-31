# Estimate coil sensitivities once, for every lane to share.
#
# They come from MRISubspaceRecon's own estimator rather than ours, so no lane
# is measured against maps fitted by its own conventions.
#
#   julia --project=. coil_maps.jl <prepared.npz> <size> <out.npz>

using NPZ
using MRISubspaceRecon

const path = ARGS[1]
const size_n = parse(Int, ARGS[2])
const out = ARGS[3]

held = npzread(path)
kspace = permutedims(held["kspace"], (3, 4, 2, 1))
Ncoil = size(kspace, 4)
Nframe = size(kspace, 3)
Nsamp = size(kspace, 1) * size(kspace, 2)
data = reshape(kspace, Nsamp, Nframe, Ncoil)
trj = reshape(permutedims(held["trajectory"], (4, 2, 3, 1)), 3, Nsamp, Nframe)

img_shape = (size_n, size_n, size_n)
println("estimating $(Ncoil) maps on $(img_shape) ...")
flush(stdout)
cmaps = calculate_coil_maps(data, trj, img_shape; verbose=true)

stacked = Array{ComplexF32}(undef, Ncoil, img_shape...)
for c in 1:Ncoil
    stacked[c, :, :, :] = cmaps[c]
end
npzwrite(out, Dict("maps" => stacked))
println("wrote $out  $(size(stacked))")
