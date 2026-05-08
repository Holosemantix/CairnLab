# LeWM 去 BN 的可行性与诊断

> **状态**：未决，留作后续实验候选。本文记录"去掉 LeWM projection head 末端 BatchNorm1d"的动机、已观察现象、机制分析与诊断 / 实验建议。
> **关系**：与 plan_v3 / plan_adaptive_resolution 主线无强依赖；属"BN train/test 不一致"这条独立线索。
> **触发场景**：用户尝试 `encoder.projection_head.norm_fn=none` 训练，发现 SIGReg loss 居高不下、加权无效、target.detach() 也无效，怀疑梯度被掐断。

---

## 1. 动机

LeWM projector 末端是 `BatchNorm1d(embed_dim)`。BN 有两个角色：

1. **scale anchor**：强制每维 `mean=0, std=1`，让 SIGReg 的固定 knot 位置落在分布合理区间。
2. **anti-collapse 隐式作用**：通过 batch 耦合阻止 trivial collapse（所有样本输出常数）。

代价：

- **train/test 不一致**：训练用 batch stats、推理用 running stats，前若干 epoch running stats 未收敛会有 train/eval gap（短期失配，通常 1–2 epoch 后稳定）。
- 推理阶段 ckpt 加载顺序有要求，eval-time 行为依赖 BN 运行统计是否被正确 freeze。

如果能去 BN，这两个代价同时消除。

---

## 2. 已观察现象（用户报告）

`encoder.projection_head.norm_fn=none` 直接训练时：

1. **SIGReg 高位不变**：SIGReg loss 维持在某个高常数附近，多 epoch 不下降。
2. **调大 SIGReg 权重也没用**：`loss.sigreg.weight` 加倍 / 加 10 倍，SIGReg loss 数值没明显变化（不是按比例缩放，是真的不动）。
3. **`target.detach()` 没影响**：`loss.target_stop_grad=true` 切换后，loss 曲线、SIGReg 曲线、pred_loss 曲线都基本一致。

---

## 3. 机制分析（最可能的因果）

### 3.1 SIGReg 是 scale-敏感的

SIGReg 实现要点（`module.py::SIGReg`）：用一组随机投影方向 `num_proj=1024`、固定 17 个 knot 位置在 z-score 空间打 score-matching 风格 loss。**knots 位置预设了输入大约处于"每维 std≈1"的状态**。

如果 z 实际尺度漂走（每维 std=10⁻³ 或 10² 都可能），knots 全部落在分布两个尾部之外，loss 值被卡在一个**与梯度方向无关**的高常数：

- **SIGReg loss 高且不动** → 知道 1
- **加权无效**：因为 gradient 方向本身指向"噪声方向"而非"减小 loss"的方向，乘以更大权重还是噪声 → 知道 2
- **target.detach() 无效**：因为问题不在 pred ↔ target 的 gradient 路径，而在 SIGReg 本身的 input 失配 → 知道 3

三条观察在"SIGReg knots 失配"假设下都能解释。

### 3.2 替代假设（次概率）

| 假设 | 是否符合观察 | 评估 |
|---|---|---|
| Encoder 梯度被 BN 去除后变成 0 | 不完全符合：pred_loss 应该仍可下降 | 需诊断验证 |
| Projector 输出 collapse 到常数 | 部分符合：SIGReg 可能算到 NaN 或固定值；但 pred_loss 也会塌缩 | 需看 z 范数曲线 |
| Optimizer LR 太低跟不上 SIGReg 梯度 | 加权无效不支持此假设（加权应该等比放大梯度） | 排除 |

最可能仍是 **3.1 SIGReg knots 失配**。

---

## 4. 诊断步骤（决定要不要继续 BN-free 之前）

### 4.1 加梯度监控

代码端可临时（不推 main）加一个 Lightning callback 或在 `lejepa_forward` 末尾打 grad norm，分组按：

- `model.encoder` (ViT backbone)
- `model.projector`
- `model.predictor`
- `model.pred_proj`
- `model.action_encoder`

或更简的方式：直接用 swanlab 的 `swanlab.watch(model, log='gradients', log_freq=N)`（在 `train.py::run` 中 `manager()` 之前调用）。

### 4.2 加 z 尺度监控

在 `lejepa_forward` 中：

```python
with torch.no_grad():
    z = output["emb"]                              # (B, T, D)
    output["z_norm_mean"] = z.norm(dim=-1).mean()  # 整体范数
    output["z_per_dim_std"] = z.std(dim=(0, 1)).mean()  # 每维 std 的均值
    output["z_per_dim_std_max"] = z.std(dim=(0, 1)).max()
    output["z_per_dim_std_min"] = z.std(dim=(0, 1)).min()
```

把 `z_per_dim_std_*` 加入 `metrics_dict` 过滤器即可在 swanlab 看到。

### 4.3 跑 5 分钟即可定性

`norm_fn=none` 跑 ~200 step：

| 现象 | 结论 |
|---|---|
| `grad/encoder_l2` 正常但 `z_per_dim_std` 漂出 [0.3, 3]，SIGReg loss 高 | **3.1 SIGReg knots 失配**（最可能）→ 换 anchor 即可 |
| `grad/projector_l2` ≈ 0 且 `z_norm_mean` → 0 或 → ∞ | **collapse**：projector 输出退化，需要更强 anti-collapse |
| `grad/encoder_l2` 远小于其他组 | 梯度被某一头抢光，需要 grad clip 调小 |

---

## 5. 解决方案（如果决定继续 BN-free）

按改动量从小到大排：

| 方案 | 改动 | 说明 / 取舍 |
|---|---|---|
| **A. LayerNorm at projector tail** | `encoder.projection_head.norm_fn=layernorm`（已支持 `resolve_norm_fn`） | per-token 归一，无 train/test gap；但**逐维独立约束**比 BN 弱，knots 仍可能略微失配，需观察 SIGReg loss |
| **B. RMSNorm 替代** | 自定义 norm_fn | 类似 A 但更便宜 |
| **C. L2-normalize 输出（=SWM）** | `encoder.projection_head.norm_fn=none` + 末端 `F.normalize` | 这就是 SphericalJEPA 路线；如果效果接近 SWM-base，证明 BN 的关键作用就是 scale anchor |
| **D. 给 SIGReg 输入手动 z-score** | `train.py::lejepa_forward` 调 SIGReg 前手动 `(z - z.mean(0)) / (z.std(0)+eps)`（detach mean/std） | 最贴近"BN 但去掉 train/test 不一致"的语义；不改架构 |
| **E. 改 SIGReg 实现使其 scale-invariant** | `module.py::SIGReg` knots 改成自适应（基于当前 batch z 的 quantile） | 最深入；可能引入数值不稳 |

**首推方案 D**：保留 LeWM 的全部架构、只把 BN 在数值上等价为"前向 z-score + 不在 ckpt 里存 running stats"，对 SIGReg 的 input 分布要求几乎完全保留，且消除 train/test 不一致。

**次推方案 A**：最小改动验证 LayerNorm 是否够用；不够再走 D。

---

## 6. 推荐行动顺序

1. **先做 4.1 + 4.2 的诊断**（加 grad norm + z-scale 监控），跑 5 分钟 BN-free 训练，确认是不是 SIGReg knots 失配。**不在 main 落库这个临时监控**——只在调试分支或本地跑一下。
2. 根据诊断结论：
   - 若是 SIGReg knots 失配（最可能）：先试 **方案 A (LayerNorm)**，10 epoch full eval；不够再上 **方案 D**。
   - 若是 collapse：BN 本身在做 anti-collapse，去 BN 必须配 **L2-normalize（SWM）** 或 stop-grad + 非对称架构（LeWM 的 SimSiam-style 已有 `loss.target_stop_grad`，但需要同时加 predictor 路径不对称）。
   - 若是梯度断流：检查 head 之间梯度量级，调 `gradient_clip_val`。
3. 任何方案落地前都要证明**不退步于 LeWM-base**（TwoRoom 93 / PushT 87.33 / Reacher 57.67 / Cube 72.33，详 plan_v3 §2.2）。

---

## 7. 现状结论

**当前不建议 main 分支默认去 BN**：

- 用户实际观察的失败现象有清晰机制解释（SIGReg knots 失配）。
- 解决方案 A/D 都需要先完成诊断 + 小规模验证。
- LeWM+noise 已经是强 baseline（plan_v3 §0），动 projector 架构有引入回退风险。
- BN 的 train/test 不一致虽真实但量级小（1–2 epoch 后稳定），改动收益不明。

**保留为后续候选**：如果 sigma-conditioned JEPA Pilot-1B 落地后想进一步追求架构简化（去除 train/test gap），再回来按 §6 顺序做。

---

## 8. 维护说明

- 本文件不影响 plan_v3 / plan_adaptive_resolution 主线，单独存在以便后续单点决策。
- 实验落地时把诊断结果与方案选择追加到 §4 / §5 / §6 对应位置。
- 如方案 A/D 跑出正/负结论，更新 §7"现状结论"段。
