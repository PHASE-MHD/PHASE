# POSEIDON dependency

PHASE uses the scOT architecture from
[POSEIDON](https://github.com/camlab-ethz/poseidon) at commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61`. POSEIDON remains an external dependency.

## Install

```bash
git clone https://github.com/camlab-ethz/poseidon.git external/poseidon
git -C external/poseidon checkout b8fa28f59bd7f7673323f28d11a12c6f3a215c61
python -m pip install --no-deps -e external/poseidon
```

## Pretrained weights

Transfer-learning configurations load the external `camlab-ethz/Poseidon-T`
checkpoint through the POSEIDON API.

PHASE retains the pretrained velocity representation and expands the
input/output pathways for magnetic variables. Exact transfer-learning and
optimizer settings are defined by the canonical YAML files under `configs/`.
