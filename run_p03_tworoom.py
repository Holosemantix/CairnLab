import os
import sys
from pathlib import Path

# Fix dataset root for tworoom
os.environ['STABLEWM_HOME'] = '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms'

REPO_ROOT = Path('/opt/huawei/explorer-env/dataset/ag_data/code/wm_exp')
sys.path.insert(0, str(REPO_ROOT))

from tools.repr_analysis.run_full_diagnostics import run_full_diagnostics

MODEL_SPECS = {
    'LeWM-base': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm/tworoom_lewm_epoch_9_object.ckpt',
    'LeWM-fixed-std': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_noise_std_0_005/tworoom_lewm_noise_std_0_005_epoch_9_object.ckpt',
    'LeWM-perframe-p05': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_noise_0to005_p05/tworoom_lewm_noise_0to005_p05_epoch_9_object.ckpt',
    'LeWM-perframe-p1': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_noise_0to005_p1/tworoom_lewm_noise_0to005_p1_epoch_9_object.ckpt',
    'SWM-base': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425_epoch_9_object.ckpt',
    'SWM-fixed-std': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt',
    'SWM-perframe-p05': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt',
    'SWM-perframe-p1': '/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt',
}

missing = [k for k, v in MODEL_SPECS.items() if not Path(v).exists()]
if missing:
    print(f"Missing checkpoints: {missing}")
    sys.exit(1)

SAVE_DIR = Path('/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Starting P0.3 diagnostics for tworoom...")
print(f"STABLEWM_HOME={os.environ['STABLEWM_HOME']}")
print(f"Models: {list(MODEL_SPECS.keys())}")
print(f"Save dir: {SAVE_DIR}")

result = run_full_diagnostics(
    models=MODEL_SPECS,
    dataset='tworoom',
    stds=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
    rollout_steps=(1, 2, 4, 8),
    state_key=None,
    n_sequences=256,
    future_steps=8,
    frameskip=5,
    img_size=224,
    seed=3072,
    device='cuda',
    save_dir=SAVE_DIR,
    plot=True,
    skip_noise=False,
    skip_predictor=False,
    skip_resolution=False,
    predictor_history_noise_only=True,
)

print("\n=== Diagnostics Summary ===")
summary = result.get('diagnostics_summary')
if summary is not None:
    import pandas as pd
    print(pd.DataFrame(summary).to_string(index=False))
else:
    print("N/A")
print(f"\nResults saved to: {SAVE_DIR}")
