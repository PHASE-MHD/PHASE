# POSEIDON dependency

PHASE uses the scOT architecture from
[POSEIDON](https://github.com/camlab-ethz/poseidon) at commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61`. POSEIDON remains an external
dependency and is not vendored in this repository.

## Install scOT

Install the pinned revision without replacing the PHASE environment
dependencies:

```bash
git clone https://github.com/camlab-ethz/poseidon.git external/poseidon
git -C external/poseidon checkout b8fa28f59bd7f7673323f28d11a12c6f3a215c61
python -m pip install --no-deps -e external/poseidon
```

Verify the installation:

```bash
python -c "from scOT.model import ScOT, ScOTConfig; print('scOT import OK')"
```

## Pretrained weights

Transfer-learning configurations load `camlab-ethz/Poseidon-T` from Hugging
Face. Set `HF_HOME` to control the cache location:

```bash
export HF_HOME=/path/to/huggingface/cache
```

Populate this cache before offline training. The scOT-without-transfer-learning
ablation requires the scOT package but does not load pretrained weights. The
tFNO and DINO baselines require neither scOT nor Poseidon-T.

PHASE retains the pretrained velocity representation and expands the
input/output pathways for magnetic variables. Exact transfer-learning and
optimizer settings are defined by the canonical YAML files under `configs/`.

Please cite POSEIDON when using the scOT architecture or Poseidon-T weights.
