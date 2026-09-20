#!/usr/bin/env bash
set -euo pipefail

PHASE_ROOT=${PHASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-${PHASE_ROOT}/outputs/ablation_smoke_20pct_10ep}
DINO_TFNO_SHA256=${DINO_TFNO_SHA256:-a96152ba4dc4b341d9a336c4c619e55835e1776bda8655824f96d25398d59e7d}
QUEUE=${QUEUE:-gpuhopper}

: "${DATA_ROOT:?Set DATA_ROOT to the canonical decaying-turbulence data root}"
: "${POSEIDON_ROOT:?Set POSEIDON_ROOT to the pinned POSEIDON checkout}"
: "${POSEIDON_DEPS:?Set POSEIDON_DEPS to the POSEIDON dependency directory}"
: "${MHD_PINO_DEPS:?Set MHD_PINO_DEPS to the MHD dependency directory}"
: "${CONTAINER:?Set CONTAINER to the supported Apptainer image}"
: "${DINO_TFNO_CHECKPOINT:?Set DINO_TFNO_CHECKPOINT to the verified tFNO artifact}"
: "${PROJECT:?Set PROJECT to your PBS project code}"

for path in "$PHASE_ROOT" "$DATA_ROOT" "$POSEIDON_ROOT" "$POSEIDON_DEPS" "$MHD_PINO_DEPS"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done
for path in "$CONTAINER" "$DINO_TFNO_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file does not exist: $path" >&2
    exit 1
  fi
done
if [[ -e "$RUN_ROOT/submitted_jobs.txt" ]]; then
  echo "Run root already contains a submission record: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"/{pbs,logs,features,stats,training/checkpoints}

write_job() {
  local stage=$1
  local payload=$2
  local script="$RUN_ROOT/pbs/${stage}.pbs"
  cat > "$script" <<JOB
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l mem=128GB
#PBS -l ngpus=1
#PBS -l jobfs=20GB
#PBS -l storage=scratch/ek9
#PBS -l wd
#PBS -N ${stage}
#PBS -j oe
#PBS -o ${RUN_ROOT}/logs/${stage}.out

set -euo pipefail
module load apptainer

apptainer exec --nv \
  --bind "${PHASE_ROOT}:/workspace" \
  --bind "${DATA_ROOT}:/data:ro" \
  --bind "${RUN_ROOT}:/run" \
  --bind "${POSEIDON_ROOT}:/poseidon" \
  --bind "${POSEIDON_DEPS}:/opt/poseidon_deps" \
  --bind "${MHD_PINO_DEPS}:/opt/mhd_pino" \
  --bind "$(dirname "${DINO_TFNO_CHECKPOINT}"):/external:ro" \
  "${CONTAINER}" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    export PYTHONPATH=/workspace/src:/workspace:/poseidon:/opt/poseidon_deps:/opt/mhd_pino:/apps/python3/3.11.7/lib/python3.11/site-packages
    export DATA_ROOT=/data
    export FEATURE_ROOT=/run/features
    export STATS_ROOT=/run/stats
    export OUTPUT_ROOT=/run/training
    ${payload}
  '
JOB
  chmod 755 "$script"
}

write_job phase_ab01_tfno '
mkdir -p /run/stats/ablation_smoke_20pct_10ep
python scripts/compute_statistics.py trajectory \
  --input /data/mhd_Re1000_N1000/mhd_data_3channel.npy \
  --output /run/stats/ablation_smoke_20pct_10ep/tfno_train_only_stats_Re1000.npz \
  --train-size 900 --seed 42 --split-mode single_re_seed42 --sub-t 4
python scripts/train_tfno.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/tfno_re1000.yaml
'

write_job phase_ab02_dino '
mkdir -p /run/stats/ablation_smoke_20pct_10ep
echo "'"${DINO_TFNO_SHA256}"'  /external/tfno_Re1000.pt" | sha256sum -c -
python scripts/compute_statistics.py trajectory \
  --input /data/mhd_Re1000_N1000/mhd_data_3channel.npy \
  --output /run/stats/ablation_smoke_20pct_10ep/dino_train_only_stats_Re1000.npz \
  --train-size 900 --seed 42 --split-mode single_re_seed42 --sub-t 4
python scripts/generate_diffusion_features.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/dino_conditioner_re1000.yaml \
  --checkpoint /external/tfno_Re1000.pt \
  --output-root /run/features/ablation_smoke_20pct_10ep/dino_re1000 \
  --batch-size 4 --num-workers 0
python scripts/compute_statistics.py diffusion \
  --input /run/features/ablation_smoke_20pct_10ep/dino_re1000/train \
  --output-prefix /run/stats/ablation_smoke_20pct_10ep/dino_re1000 \
  --prediction-mode direct
python scripts/train_dino.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/dino_re1000.yaml
'

write_job phase_ab03_scot0 '
python scripts/train_scot.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/scot_without_tl_re1000.yaml
'

write_job phase_ab04_scotTL '
python scripts/train_scot.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/scot_with_tl_re1000.yaml
'

write_job phase_ab05_naiveMR '
python scripts/train_scot.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/naive_multi_re.yaml
'

write_job phase_ab06_gatedMR '
python scripts/train_scot.py \
  --config configs/acceptance/ablation_smoke_20pct_10ep/gated_adapter_multi_re.yaml
'

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "Wrote PBS scripts to $RUN_ROOT/pbs; no jobs submitted."
  exit 0
fi

job_tfno=$(qsub "$RUN_ROOT/pbs/phase_ab01_tfno.pbs")
job_dino=$(qsub "$RUN_ROOT/pbs/phase_ab02_dino.pbs")
job_scot0=$(qsub "$RUN_ROOT/pbs/phase_ab03_scot0.pbs")
job_scot_tl=$(qsub "$RUN_ROOT/pbs/phase_ab04_scotTL.pbs")
job_naive=$(qsub -W "depend=afterok:${job_scot_tl}" "$RUN_ROOT/pbs/phase_ab05_naiveMR.pbs")
job_gated=$(qsub -W "depend=afterok:${job_scot_tl}" "$RUN_ROOT/pbs/phase_ab06_gatedMR.pbs")

{
  echo "tfno=${job_tfno}"
  echo "dino=${job_dino}"
  echo "scot_without_tl=${job_scot0}"
  echo "scot_with_tl=${job_scot_tl}"
  echo "naive_multi_re=${job_naive}"
  echo "gated_adapter_multi_re=${job_gated}"
} | tee "$RUN_ROOT/submitted_jobs.txt"
