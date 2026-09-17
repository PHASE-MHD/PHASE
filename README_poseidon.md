# POSEIDON dependency and transfer learning

PHASE uses the scalable Operator Transformer (scOT) architecture from
[POSEIDON](https://github.com/camlab-ethz/poseidon). POSEIDON is an external
dependency and is not vendored in this repository.

The tFNO and DINO baselines do not require POSEIDON. All scOT-based ablations
require the `scOT` package, including the model trained without transfer
learning. Transfer-learning experiments additionally require the pretrained
`camlab-ethz/Poseidon-T` weights.

## 1. Install the scOT package

The PHASE experiments used POSEIDON commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61`. Install that exact revision:

```bash
python -m pip install \
  "scOT @ git+https://github.com/camlab-ethz/poseidon.git@b8fa28f59bd7f7673323f28d11a12c6f3a215c61"
```

POSEIDON pins several older package versions in its own metadata. When using a
PHASE environment that already provides compatible versions of PyTorch,
Transformers, Accelerate, and the other dependencies, install only the scOT
package itself to avoid replacing that environment:

```bash
git clone https://github.com/camlab-ethz/poseidon.git external/poseidon
git -C external/poseidon checkout b8fa28f59bd7f7673323f28d11a12c6f3a215c61
python -m pip install --no-deps -e external/poseidon
```

Do not commit `external/poseidon/` into PHASE. It is an independently
maintained upstream project and should remain an external dependency.

Verify the architecture import:

```bash
python -c "from scOT.model import ScOT, ScOTConfig; print('scOT import OK')"
```

## 2. Obtain the pretrained Poseidon-T weights

The transfer-learning configurations identify the pretrained model as:

```yaml
model_params:
  poseidon_model: "camlab-ethz/Poseidon-T"
  load_pretrained_poseidon: true
```

At first use, Hugging Face downloads the model into its local cache. Internet
access is therefore required unless the cache has already been populated. The
cache location can be controlled with `HF_HOME`:

```bash
export HF_HOME=/path/to/huggingface/cache
```

If authentication is required by the execution environment, authenticate
before training:

```bash
huggingface-cli login
```

Test both the package and pretrained-weight download before launching a PHASE
training run:

```bash
python - <<'PY'
from scOT.model import ScOT

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
print("Loaded Poseidon-T")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
PY
```

For offline training, run this check once on a networked machine using the same
`HF_HOME`, then make that populated cache available on the compute node.

## 3. How PHASE uses POSEIDON

The PHASE MHD wrappers import `ScOT` and `ScOTConfig` from the installed
package. For transfer learning, they load `Poseidon-T`, retain its pretrained
velocity representation, and expand the input/output pathways for the magnetic
variables. The PHASE configurations then assign separate optimizer groups to
the pretrained parameters and newly introduced MHD parameters.

The scOT model without transfer learning still needs the installed `scOT`
package for the architecture, but it must disable pretrained-weight loading:

```yaml
model_params:
  poseidon_model: null
  load_pretrained_poseidon: false
```

## 4. Ablation dependency summary

| Ablation | scOT package | Poseidon-T weights |
| --- | --- | --- |
| tFNO | No | No |
| DINO | No | No |
| scOT without transfer learning | Yes | No |
| scOT with transfer learning | Yes | Yes |
| Naive multi-regime scOT | Yes | Yes |
| Gated-adapter multi-regime scOT | Yes | Yes |
| Four-channel scOT with Helmholtz projection and physics losses | Yes | Yes |
| PHASE with residual diffusion | Yes | Yes |

## 5. Attribution

Users of PHASE should also cite the POSEIDON paper and repository. See the
upstream POSEIDON README for its current citation information and license.
