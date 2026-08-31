#!/usr/bin/env bash
# Regenerate examples/coil_maps.npz from BART.
#
# Three sensitivity banks that differ in the one way the CoilKernels design
# turns on -- whether the map is band-limited -- all produced by the reference
# implementation rather than by hand.
#
#   simulated : bart's own analytic 8-channel head coil
#   espirit   : ESPIRiT maps, eigenvector-normalised
#   nlinv     : NLINV maps, whose Sobolev weighting is a band limit
#
# Needs BART on PATH and its python/ on PYTHONPATH.
set -euo pipefail
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

bart coils --H2D8C -n 8 "$out/sens2d"          # the simulated array
bart phantom -x 128 -S 8 "$out/coilimgs"       # an object seen through it
bart phantom -x 128 "$out/obj"
bart fft -u 3 "$out/coilimgs" "$out/ksp"
bart ecalib -m 1 "$out/ksp" "$out/maps_espirit"
bart nlinv -i 12 "$out/ksp" "$out/reco" "$out/maps_nlinv"

python - "$out" <<'PY'
import sys
import numpy as np
import cfl

root = sys.argv[1] + "/"
def load(name):
    bank = cfl.readcfl(root + name).squeeze()
    if bank.ndim == 3:
        bank = bank.transpose(2, 0, 1)
    return np.ascontiguousarray(bank.astype(np.complex64))

banks = {
    "simulated": load("sens2d"),
    "espirit": load("maps_espirit"),
    "nlinv": load("maps_nlinv"),
}
banks = {k: (v / np.abs(v).max()).astype(np.complex64) for k, v in banks.items()}
np.savez_compressed("examples/coil_maps.npz", object=load("obj"), **banks)
print("wrote examples/coil_maps.npz")
PY
