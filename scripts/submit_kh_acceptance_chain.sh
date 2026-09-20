#!/usr/bin/env bash
set -euo pipefail

PHASE_ROOT=${PHASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-${PHASE_ROOT}/outputs/kh_20pct_10ep}
QUEUE=${QUEUE:-gpuhopper}

: "${DATA_ROOT:?Set DATA_ROOT to the canonical multi-Re KH data root}"
: "${POSEIDON_ROOT:?Set POSEIDON_ROOT to the pinned POSEIDON checkout}"
: "${POSEIDON_DEPS:?Set POSEIDON_DEPS to the POSEIDON dependency directory}"
: "${MHD_PINO_DEPS:?Set MHD_PINO_DEPS to the MHD dependency directory}"
: "${CONTAINER:?Set CONTAINER to the supported Apptainer image}"
: "${PROJECT:?Set PROJECT to your PBS project code}"

for path in "$PHASE_ROOT" "$DATA_ROOT" "$POSEIDON_ROOT" "$POSEIDON_DEPS" "$MHD_PINO_DEPS"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$CONTAINER" ]]; then
  echo "Container does not exist: $CONTAINER" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT/submitted_jobs.txt" ]]; then
  echo "Run root already contains a submission record: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"/{pbs,logs,features,stats,training/checkpoints}

write_job() {
  local stage=$1
  local queue_name=$2
  local walltime=$3
  local payload=$4
  local script="$RUN_ROOT/pbs/${stage}.pbs"
  cat > "$script" <<EOF
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -q ${queue_name}
#PBS -l walltime=${walltime}
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
EOF
  chmod 755 "$script"
}

write_job phase18_srscot "$QUEUE" 03:00:00 '
python scripts/train_scot.py \
  --config configs/acceptance/kh_20pct_10ep/single_re_scot.yaml
'

write_job phase18_srprep gpuvolta 01:00:00 '
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/acceptance/kh_20pct_10ep/single_re_scot.yaml \
  --conditioner-checkpoint /run/training/checkpoints/kh_20pct_10ep/single_re_scot_re1000.pt \
  --output-root /run/features/acceptance/kh_20pct_10ep/single_re/phase_re1000 \
  --batch-size 4 \
  --num-workers 4 \
  --single-re 1000
python scripts/compute_statistics.py diffusion \
  --input /run/features/acceptance/kh_20pct_10ep/single_re/phase_re1000/train \
  --output-prefix /run/stats/acceptance/kh_20pct_10ep/single_re/phase_re1000 \
  --prediction-mode residual
'

write_job phase18_srphase "$QUEUE" 01:00:00 '
python scripts/train_dino.py \
  --config configs/acceptance/kh_20pct_10ep/single_re_phase.yaml
'

write_job phase18_mrscot "$QUEUE" 06:00:00 '
python scripts/train_scot.py \
  --config configs/acceptance/kh_20pct_10ep/multi_re_scot.yaml
'

write_job phase18_mrprep gpuvolta 01:00:00 '
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/acceptance/kh_20pct_10ep/multi_re_scot.yaml \
  --conditioner-checkpoint /run/training/checkpoints/kh_20pct_10ep/multi_re_scot_t0_5.pt \
  --output-root /run/features/acceptance/kh_20pct_10ep/multi_re/phase_t0_5 \
  --batch-size 4 \
  --num-workers 4
python scripts/compute_statistics.py diffusion-per-re \
  --input /run/features/acceptance/kh_20pct_10ep/multi_re/phase_t0_5/train \
  --output-dir /run/stats/acceptance/kh_20pct_10ep/multi_re/phase_t0_5 \
  --prediction-mode residual
'

write_job phase18_mrphase "$QUEUE" 06:00:00 '
python scripts/train_dino.py \
  --config configs/acceptance/kh_20pct_10ep/multi_re_phase.yaml
'

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "Wrote PBS scripts to $RUN_ROOT/pbs; no jobs submitted."
  exit 0
fi

job_sr_scot=$(qsub "$RUN_ROOT/pbs/phase18_srscot.pbs")
job_sr_prep=$(qsub -W "depend=afterok:${job_sr_scot}" "$RUN_ROOT/pbs/phase18_srprep.pbs")
job_sr_phase=$(qsub -W "depend=afterok:${job_sr_prep}" "$RUN_ROOT/pbs/phase18_srphase.pbs")
job_mr_scot=$(qsub -W "depend=afterok:${job_sr_phase}" "$RUN_ROOT/pbs/phase18_mrscot.pbs")
job_mr_prep=$(qsub -W "depend=afterok:${job_mr_scot}" "$RUN_ROOT/pbs/phase18_mrprep.pbs")
job_mr_phase=$(qsub -W "depend=afterok:${job_mr_prep}" "$RUN_ROOT/pbs/phase18_mrphase.pbs")

{
  echo "single_re_scot=${job_sr_scot}"
  echo "single_re_feature_stats=${job_sr_prep}"
  echo "single_re_phase=${job_sr_phase}"
  echo "multi_re_scot=${job_mr_scot}"
  echo "multi_re_feature_stats=${job_mr_prep}"
  echo "multi_re_phase=${job_mr_phase}"
} | tee "$RUN_ROOT/submitted_jobs.txt"
