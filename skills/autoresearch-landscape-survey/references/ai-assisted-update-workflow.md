# AI-Assisted Update Workflow

这个 skill 仓库的目标不是让 Python 完全替代调研，而是让 Python 负责格式化、去重、预分类和 prompt 打包，让 AI 同事负责读仓库、识别机制、判断差异、写分析。

## 为什么需要 AI 介入

AutoResearch 项目经常把机制散落在 README、docs、skills、prompts、pipeline code、tests、examples 和 paper 里。仅凭 YAML 字段或关键词无法可靠判断：

- 它是只生成文本，还是能真实执行代码/训练？
- reviewer 是语言层面的，还是绑定 run artifact？
- claim verification 是引用核验，还是 run-level evidence gate？
- multi-agent 是角色扮演，还是 producer/verifier/judge 分权？
- 是否有新的属性需要更新 taxonomy？

因此新项目流程必须包含 AI/human deep reading。

## 推荐流程

```bash
# 1. 填 intake YAML
cp skills/autoresearch-landscape-survey/assets/templates/new_project_intake.yaml reports/intake/new_project.yaml

# 2. 快速查重和找相似项目
python skills/autoresearch-landscape-survey/scripts/check_new_project.py \
  --input reports/intake/new_project.yaml --format markdown \
  --output reports/intake/new_project_precheck.md

# 3. 从 YAML 做启发式 facet 预测
python skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py \
  --input reports/intake/new_project.yaml --format yaml \
  --output reports/intake/new_project_facets.yaml

# 4. 生成给 AI 同事的深度阅读 prompt bundle
python skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py \
  --input reports/intake/new_project.yaml \
  --output reports/intake/new_project_ai_review_prompt.md

# 5. AI 同事读取仓库和论文，返回结构化报告
# 6. 人类/主 AI 更新 data/project_registry.yaml
# 7. 重新生成整体一览和报告 prompt
python skills/autoresearch-landscape-survey/scripts/render_project_overview.py
python skills/autoresearch-landscape-survey/scripts/render_landscape_report.py
python skills/autoresearch-landscape-survey/scripts/prepare_ai_report_prompt.py
```

## AI 同事必须回答的问题

1. 是否已经调研过或是已有项目的 fork/variant？
2. 它的 field、workflow、execution、verification、accountability、agent topology 是什么？
3. 它有哪些 advertised capability，哪些有 repo/paper/test/artifact 证据？
4. 它最像哪些项目，和它们不同在哪里？
5. 它是否带来新 facet？
6. 它是否改变我们 Accountable Research CI 的设计？

## 报告生成方式

`render_project_overview.py` 和 `render_landscape_report.py` 只负责稳定格式化 registry。更深入的判断应通过：

```bash
python skills/autoresearch-landscape-survey/scripts/prepare_ai_report_prompt.py
```

生成 `reports/ai_landscape_report_prompt.md`，再交给 AI 把项目地图写成新的调研报告。
