#!/bin/bash
# Batch submission for the ConvMixer arm (M-05, D023).
#
# Why sbatch rather than the `cbrun -t rocky -- srun -c 16 --` one-liner in the README: those
# are interactive and die with the submitting shell, which is fine for a 4-minute pilot but not
# for a multi-hour full run on a contended partition. The schedule, manifest, and seed are
# unchanged -- only the submission mechanism differs.
#
# `-C x48-m7a` pins the same node class the frozen T-03 runs used (AMD EPYC 9R14, 48-core, 16
# threads allocated). That is not cosmetic: `seconds_per_frame` is a reported column in
# docs/RESULTS.md, and a run placed on an Intel m7i node would produce a throughput number that
# cannot be compared with the frozen EDSR and U-Net rows.
#
# Usage, from the repository root:
#   cbrun -t rocky -- sbatch scripts/slurm_convmixer.sbatch <model> <experiment>
# where <model> names a configs/model/<model>_x4.yaml (`convmixer`, `convmixer_nonorm`) and
# <experiment> is `pilot` or `full`. Both default to the published full run.

#SBATCH --job-name=convmixer
#SBATCH --partition=cpu
#SBATCH --constraint=x48-m7a
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j-convmixer.out

set -euo pipefail

MODEL="${1:-convmixer}"
EXPERIMENT="${2:-full}"
REPO=/cb/home/fuxic/ws/shallow-water-super-resolution
cd "$REPO"
mkdir -p logs

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-?}" \
     "model=${MODEL} experiment=${EXPERIMENT}"
python -c "import torch; print('bf16', torch.cpu._is_avx512_bf16_supported(), 'threads', torch.get_num_threads())"

exec python -m swe_sr.train \
    --config "configs/model/${MODEL}_x4.yaml" \
    --experiment "configs/experiment/${EXPERIMENT}.yaml"
