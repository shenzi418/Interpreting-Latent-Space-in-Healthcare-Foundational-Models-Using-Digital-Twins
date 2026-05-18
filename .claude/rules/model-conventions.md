*Apply when editing `net1d.py`, `finetune_model.py`, `scripts/finetune_*.py`, or writing a new training entry-point script.*

## Net1D / ECGFounder shape contract
- Input: 12-lead ECG, shape `(B, 12, L)`. Leads must be in target order — see `data-pipeline.md`.
- Backbone init args (preserved across every entry point): `base_filters=64, filter_list=[64,160,160,400,400,1024,1024], m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16, stride=2, groups_width=16, use_bn=False, use_do=False`.
- Output features: 1024-d after global avg pool, before `model.dense`.
- `ConvAdapter1D`: zero-initialised 1×1-down → nonlinearity → 1×1-up residual block; only trainable backbone component in linear-probe mode.

## `linear_prob=True + use_adapter=True` is BUGGED in `ft_12lead_ECGFounder`
- Combination freezes `model.dense` along with the backbone (`finetune_model.py:52-58`) — silent broken training.
- Workaround used in Exp 7 shared-head: pass `linear_prob=False`, then run:
  ```python
  freeze_backbone_except_adapters(model)
  for p in model.dense.parameters():
      p.requires_grad = True
  ```
- `ft_multihead_ECGFounder` (`finetune_model.py:135-140`) freezes `model.backbone` only — unaffected.

## `sys.path` injection (every entry-point script)
- Project is not pip-installable. Copy this verbatim into the first ~10 lines of any new `scripts/` or `analysis/` entry point:
  ```python
  SCRIPT_DIR = Path(__file__).resolve().parent
  REPO_ROOT = SCRIPT_DIR.parent
  if str(REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(REPO_ROOT))
  if str(SCRIPT_DIR) not in sys.path:
      sys.path.insert(0, str(SCRIPT_DIR))
  ```
- Skipping makes `from net1d import …` / `from losses.mmd import …` fail with `ModuleNotFoundError`.
- Tag imports after the path injection with `# pylint: disable=wrong-import-position`.