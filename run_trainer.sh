#!/usr/bin/env bash

# ==========================================
# Train + Eval sweep + Noise table，一站式
# ==========================================
#
# 必填 env vars (与旧脚本一致):
#   dataset_name              tworoom | pusht | cube | reacher
#   trainer_file              train.py | train_swm.py
#   config                    swm | lewm (also accepts swm.yaml | lewm.yaml)
#   output_model_name         模型名后缀（最终落盘 `${dataset_name}_${output_model_name}`）
#   num_eval                  每次 eval 的 episode 数量
#   STABLEWM_HOME             checkpoint 根目录
#
# 可选 Hydra overrides (与旧脚本一致，留空则不下发):
#   encoder_projection_head_type, loss_regularizer_*, loss_uniformity_*,
#   loss_temporal_hinge_*, loss_inverse_dynamics_weight,
#   loss_transition_distance_weight, loss_pred_*, loss_rollout_*,
#   seed, wm_embed_dim, wm_inference_*, image_noise_std_min/max/apply_to_val
#
# 新增 env vars:
#   image_noise_noise_prob    每帧加噪概率 (默认 1.0；<1 制造 clean+noisy 混合)
#   eval_corruption_stds      eval sweep 噪声列表，空格分隔
#                              默认 "0.0 0.03 0.05 0.08"
#                              传 "" 跳过 eval sweep（仍跑 noise table）
#   eval_corruption_apply_to  eval sweep 加噪目标，逗号分隔；'+' 表示同一组里多目标
#                              默认 "pixels+goal,pixels,goal"
#                              （"pixels+goal" 表示同时加噪两端）
#   frameskip                 数据加载 frameskip；默认 5（与训练 data config 一致）
#   eval_gpus                 GPU id 列表，空格分隔；默认自动探测全部
#   noise_table_stds          诊断扫的 std；默认 0.0~0.10 一组（仍由本字段控制）
#   diagnostic_rollout_steps  predictor 自回归 rollout 步数；默认 "1 2 4 8"
#   skip_eval_sweep           设 1 跳过 eval sweep
#   skip_noise_table          legacy 名，等价于 skip_diagnostics
#   skip_diagnostics          设 1 跳过整套诊断（noise/predictor/resolution）
#   diagnostic_skip_predictor 设 1 仅跳过 predictor_sensitivity
#   diagnostic_skip_resolution 设 1 仅跳过 task_resolution
#   eval_epoch                用于 eval 的 epoch 编号；默认读取训练 config 的 trainer.max_epochs
#
# 用法示例：
#   dataset_name=tworoom trainer_file=train_swm.py config=swm \
#     output_model_name=perframe_0to05_p1 num_eval=50 \
#     image_noise_std_min=0.0 image_noise_std_max=0.05 image_noise_noise_prob=1.0 \
#     eval_corruption_stds="0.0 0.05 0.08" \
#     bash run_trainer.sh
#
# 在结果目录会得到：
#   eval_results/<label>.log              每个 eval 的完整 stdout
#   eval_results/<label>_results.txt      该 eval 的 metrics 文本
#   eval_results/diagnostics/             noise + predictor + task_resolution
#       noise_sensitivity.{csv,json}
#       geometry_summary.{csv,json}
#       predictor_sensitivity.{csv,json}
#       task_resolution.{csv,json}
#       diagnostics_summary.json          per-checkpoint 一行 roll-up
#       *.png                             curves & geometry tradeoff plots
#   eval_results/summary.txt              所有 eval + diagnostics 的摘要
# ==========================================

set -u  # treat unset vars as errors after the unsets below
set -o pipefail

# 切到脚本所在目录，确保所有相对路径（config/、tools/ 等）一致解析，
# 避免从其它 cwd 调用本脚本时 diagnostics 读不到 train data config。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 1. Hydra 参数构建 (沿用原脚本) ----------
CMD_ARGS=()
add_override() {
    local key="$1"
    local value="${2:-}"
    if [ -n "$value" ]; then
        CMD_ARGS+=("$key=$value")
    fi
}

# Backward-compatible Hydra config name. Older callers may pass `lewm.yaml`;
# internally we need the stem for both config/train/<name>.yaml lookup and
# `--config-name=<name>`.
config_name="${config##*/}"
config_name="${config_name%.yaml}"
config_name="${config_name%.yml}"

# data: hydra data group; dataset_dirname: STABLEWM_HOME/lewm-<dirname>;
# default_h5_name: 真实 HDF5 dataset 名称（必须与 config/train/data/<data>.yaml
# 和 config/eval/<dataset_name>.yaml 中的 name 一致）。在此显式列出避免依赖
# cwd 的 grep/sed 解析。
case "${dataset_name}" in
    tworoom) data="tworoom"; dataset_dirname="tworooms"; default_h5_name="tworoom"                    ;;
    pusht)   data="pusht";   dataset_dirname="pusht";    default_h5_name="pusht_expert_train"         ;;
    cube)    data="ogb";     dataset_dirname="cube";     default_h5_name="ogbench/cube_single_expert" ;;
    reacher) data="dmc";     dataset_dirname="reacher";  default_h5_name="reacher"                    ;;
    *) echo "错误: 未知的 dataset_name '${dataset_name}'"; exit 1 ;;
esac

# 从训练数据配置读取默认 frameskip，支持环境变量覆盖
_dataset_cfg="${SCRIPT_DIR}/config/train/data/${data}.yaml"
if [ -f "${_dataset_cfg}" ]; then
    _default_frameskip=$(grep -m1 '^[[:space:]]*frameskip:' "${_dataset_cfg}" | sed 's/.*:[[:space:]]*\([0-9]*\).*/\1/')
    frameskip="${frameskip:-${_default_frameskip:-5}}"
    # 双保险：若 yaml 中的 name 与 case 里硬编码不一致，提示一下
    _yaml_h5_name=$(grep -m1 '^[[:space:]]*name:' "${_dataset_cfg}" | sed 's/.*:[[:space:]]*\([^[:space:]]*\).*/\1/')
    if [ -n "${_yaml_h5_name}" ] && [ "${_yaml_h5_name}" != "${default_h5_name}" ]; then
        echo "[warn] config/train/data/${data}.yaml name=${_yaml_h5_name} 与脚本内 default_h5_name=${default_h5_name} 不一致，使用脚本值；请同步两处"
    fi
else
    frameskip="${frameskip:-5}"
fi
diagnostic_dataset_name="${diagnostic_dataset_name:-${default_h5_name}}"

# Eval 默认使用训练 config 中的 trainer.max_epochs 对应 checkpoint。
_train_cfg="${SCRIPT_DIR}/config/train/${config_name}.yaml"
if [ -f "${_train_cfg}" ]; then
    _config_max_epochs=$(awk '
        /^[^[:space:]]/ { in_trainer=($1=="trainer:") }
        in_trainer && /^[[:space:]]*max_epochs:/ {
            sub(/#.*/, "")
            sub(/.*:[[:space:]]*/, "")
            print
            exit
        }
    ' "${_train_cfg}")
else
    echo "[eval] training config not found: ${_train_cfg}"
    exit 1
fi
if [ -z "${_config_max_epochs}" ]; then
    echo "[eval] trainer.max_epochs not found in ${_train_cfg}"
    exit 1
fi
eval_epoch="${eval_epoch:-${_config_max_epochs}}"
echo "[eval] using checkpoint epoch ${eval_epoch} (trainer.max_epochs from config/train/${config_name}.yaml)"

output_model_name="${dataset_name}_${output_model_name}"

add_override "data" "${data}"
add_override "data.dataset.frameskip" "${frameskip}"
add_override "seed" "${seed:-}"
add_override "output_model_name" "${output_model_name}"
add_override "subdir" "ckpt/${output_model_name}"
add_override "encoder.projection_head.type" "${encoder_projection_head_type:-}"
add_override "loss.regularizer.type" "${loss_regularizer_type:-}"
add_override "loss.regularizer.weight" "${loss_regularizer_weight:-}"
add_override "loss.regularizer.scope" "${loss_regularizer_scope:-}"
add_override "loss.rollout.weight" "${loss_rollout_weight:-}"
add_override "loss.rollout.steps" "${loss_rollout_steps:-}"
add_override "loss.uniformity.t" "${loss_regularizer_t:-}"
add_override "loss.uniformity.mode" "${loss_uniformity_mode:-}"
add_override "loss.uniformity.temporal_exclusion" "${loss_uniformity_temporal_exclusion:-}"
add_override "loss.temporal_hinge.weight" "${loss_temporal_hinge_weight:-}"
add_override "loss.temporal_hinge.margin" "${loss_temporal_hinge_margin:-}"
add_override "loss.temporal_hinge.squared" "${loss_temporal_hinge_squared:-}"
add_override "loss.temporal_hinge.dynamic.enabled" "${loss_temporal_hinge_dynamic_enabled:-}"
add_override "loss.temporal_hinge.dynamic.base_margin" "${loss_temporal_hinge_dynamic_base_margin:-}"
add_override "loss.temporal_hinge.dynamic.min_margin" "${loss_temporal_hinge_dynamic_min_margin:-}"
add_override "loss.temporal_hinge.dynamic.max_margin" "${loss_temporal_hinge_dynamic_max_margin:-}"
add_override "loss.inverse_dynamics.weight" "${loss_inverse_dynamics_weight:-}"
add_override "loss.transition_distance.weight" "${loss_transition_distance_weight:-}"
add_override "loss.pred.space" "${loss_pred_space:-}"
add_override "loss.pred.type" "${loss_pred_type:-}"
add_override "wm.embed_dim" "${wm_embed_dim:-}"
add_override "wm.inference.rollout_state_space" "${wm_inference_rollout_state_space:-}"
add_override "wm.inference.cost_space" "${wm_inference_cost_space:-}"
add_override "wm.inference.cost_type" "${wm_inference_cost_type:-}"
add_override "image_noise.std_min" "${image_noise_std_min:-}"
add_override "image_noise.std_max" "${image_noise_std_max:-}"
add_override "image_noise.noise_prob" "${image_noise_noise_prob:-}"
add_override "image_noise.apply_to_val" "${image_noise_apply_to_val:-}"

# ---------- 2. 训练 ----------
swanlab login -k "${SWANLAB_API_KEY}"
# Defensive: if STABLEWM_HOME already points to a lewm-* subdir, go up one level first
if [[ "$(basename "$STABLEWM_HOME")" == lewm-* ]]; then
    export STABLEWM_HOME="$(dirname "$STABLEWM_HOME")/lewm-${dataset_dirname}"
else
    export STABLEWM_HOME="${STABLEWM_HOME}/lewm-${dataset_dirname}"
fi

echo "==================================================="
echo "[train] starting ${trainer_file} for ${output_model_name}"
echo "==================================================="
python ${trainer_file} --config-name="${config_name}" \
    logger_backend=swanlab \
    swanlab.enabled=True \
    "${CMD_ARGS[@]}"

train_status=$?
if [ $train_status -ne 0 ]; then
    echo "[train] failed with status ${train_status}; skipping eval sweep"
    exit $train_status
fi

# ---------- 3. Eval / Noise 通用准备 ----------
ckpt_rel="ckpt/${output_model_name}/${output_model_name}_epoch_${eval_epoch}"
ckpt_abs="${STABLEWM_HOME}/${ckpt_rel}_object.ckpt"
results_dir="${STABLEWM_HOME}/ckpt/${output_model_name}/eval_results"
mkdir -p "${results_dir}"

if [ ! -f "${ckpt_abs}" ]; then
    echo "[eval] checkpoint not found: ${ckpt_abs}"
    echo "[eval] aborting downstream steps"
    exit 1
fi

# GPU 探测
if [ -z "${eval_gpus:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        eval_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' ')
    else
        eval_gpus="0"
    fi
fi
read -ra gpu_array <<< "${eval_gpus}"
n_gpus=${#gpu_array[@]}
echo "[gpu] using GPUs: ${gpu_array[*]} (count=${n_gpus})"

# ---------- 4. Eval Sweep ----------
run_one_eval() {
    local job="$1"
    local gpu="$2"
    IFS='|' read -ra parts <<< "$job"
    local label="${parts[0]}"
    local std="${parts[1]}"
    local mode="${parts[2]}"

    local args=(
        "--config-name=${dataset_name}.yaml"
        "policy=${ckpt_rel}"
        "eval.num_eval=${num_eval}"
        "output.filename=${results_dir}/${label}_metrics.txt"
    )
    if [ "$mode" != "none" ]; then
        args+=("eval.corruption.std=${std}")
        local apply_list="${mode//+/,}"
        args+=("eval.corruption.apply_to=[${apply_list}]")
    fi

    echo "[eval] start  gpu=${gpu} label=${label} std=${std} mode=${mode}"
    CUDA_VISIBLE_DEVICES=${gpu} python eval.py "${args[@]}" \
        > "${results_dir}/${label}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[eval] done   gpu=${gpu} label=${label}"
    else
        echo "[eval] FAIL   gpu=${gpu} label=${label} (rc=${rc}; see ${results_dir}/${label}.log)"
    fi
}

if [ "${skip_eval_sweep:-0}" != "1" ]; then
    eval_corruption_stds="${eval_corruption_stds-0.0 0.03 0.05 0.08}"
    eval_corruption_apply_to="${eval_corruption_apply_to:-pixels+goal,pixels,goal}"

    jobs=()
    for std in $eval_corruption_stds; do
        is_zero=$(awk -v s="$std" 'BEGIN{print (s+0==0)?1:0}')
        if [ "$is_zero" = "1" ]; then
            jobs+=("clean|0.0|none")
        else
            IFS=',' read -ra modes <<< "${eval_corruption_apply_to}"
            for mode in "${modes[@]}"; do
                local_label="$(echo "${mode}" | tr '+' '_')_std${std}"
                jobs+=("${local_label}|${std}|${mode}")
            done
        fi
    done

    total=${#jobs[@]}
    echo "==================================================="
    echo "[eval sweep] ${total} jobs across ${n_gpus} GPUs"
    echo "==================================================="

    i=0
    while [ $i -lt $total ]; do
        pids=()
        for ((k=0; k<n_gpus && i<total; k++)); do
            run_one_eval "${jobs[$i]}" "${gpu_array[$k]}" &
            pids+=($!)
            ((i++))
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
    done
else
    echo "[eval sweep] skipped (skip_eval_sweep=1)"
fi

# ---------- 5. Full Latent-Geometry Diagnostics ----------
# Unified entry: noise_sensitivity + predictor_sensitivity + task_resolution.
# Output dir: ${results_dir}/diagnostics/
#   noise_sensitivity.{csv,json}, geometry_summary.{csv,json}, *.png
#   predictor_sensitivity.{csv,json}
#   task_resolution.{csv,json}
#   diagnostics_summary.json   (per-checkpoint roll-up; consumed by P0.7)
#
# Backward-compat env vars:
#   skip_noise_table=1         skips the entire diagnostics suite (legacy name)
#   skip_diagnostics=1         same as above (preferred)
#   noise_table_stds           still used; passed as --stds
#   diagnostic_rollout_steps   default "1 2 4 8"
#   diagnostic_skip_predictor=1 / diagnostic_skip_resolution=1   per-tool overrides
if [ "${skip_diagnostics:-${skip_noise_table:-0}}" != "1" ]; then
    noise_table_stds="${noise_table_stds:-0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1}"
    diagnostic_rollout_steps="${diagnostic_rollout_steps:-1 2 4 8}"
    diag_args=(
        "--model" "${output_model_name}=${ckpt_abs}"
        "--dataset" "${diagnostic_dataset_name}"
        "--stds" ${noise_table_stds}
        "--rollout-steps" ${diagnostic_rollout_steps}
        "--frameskip" "${frameskip}"
        "--save-dir" "${results_dir}/diagnostics"
        "--plot"
    )
    [ "${diagnostic_skip_predictor:-0}" = "1" ] && diag_args+=("--skip-predictor")
    [ "${diagnostic_skip_resolution:-0}" = "1" ] && diag_args+=("--skip-resolution")

    echo "==================================================="
    echo "[diagnostics] running full suite on ${ckpt_abs}"
    echo "==================================================="
    CUDA_VISIBLE_DEVICES=${gpu_array[0]} python -m tools.repr_analysis.run_full_diagnostics \
        "${diag_args[@]}" 2>&1 | tee "${results_dir}/diagnostics.log"
else
    echo "[diagnostics] skipped (skip_diagnostics=1)"
fi

# ---------- 6. Summary ----------
summary_file="${results_dir}/summary.txt"
{
    echo "===== ${output_model_name} eval summary ====="
    echo "ckpt: ${ckpt_abs}"
    echo "dataset: ${dataset_name}    num_eval: ${num_eval}    epoch: ${eval_epoch}"
    echo
    echo "----- eval metrics -----"
    for log in "${results_dir}"/*.log; do
        [ -e "$log" ] || continue
        base=$(basename "$log" .log)
        [ "$base" = "noise_table" ] && continue
        [ "$base" = "diagnostics" ] && continue
        echo
        echo "== ${base} =="
        # 抓 metrics line（dict 形式）
        if grep -m1 "^{" "$log" >/dev/null 2>&1; then
            grep "^{" "$log" | tail -1
        else
            grep -i "metrics\|success" "$log" | tail -3
        fi
    done
    if [ -f "${results_dir}/diagnostics/geometry_summary.csv" ]; then
        echo
        echo "----- geometry summary -----"
        cat "${results_dir}/diagnostics/geometry_summary.csv"
    fi
    if [ -f "${results_dir}/diagnostics/diagnostics_summary.json" ]; then
        echo
        echo "----- diagnostics roll-up -----"
        cat "${results_dir}/diagnostics/diagnostics_summary.json"
    fi
} > "${summary_file}"

echo "==================================================="
echo "[done] artifacts in:"
echo "  ${results_dir}/"
echo "  - per-eval logs:    *.log"
echo "  - per-eval metrics: *_metrics.txt"
echo "  - diagnostics:      diagnostics/"
echo "      * noise_sensitivity.csv / .json"
echo "      * geometry_summary.csv / .json"
echo "      * predictor_sensitivity.csv / .json"
echo "      * task_resolution.csv / .json"
echo "      * diagnostics_summary.json"
echo "      * noise_ratio_curve_goal.png"
echo "      * noise_angle_curve_goal.png"
echo "      * geometry_tradeoff_goal.png"
echo "  - summary:          summary.txt"
echo "==================================================="

# ---------- 7. Cleanup ----------
rm -rf "${STABLEWM_HOME}/ckpt/${output_model_name}"/*.mp4
