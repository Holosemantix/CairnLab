# AutoResearchClaw Codex Model Pinning

CairnLab validation uses AutoResearchClaw as an external AutoResearch system.
When AutoResearchClaw uses ACP/Codex, the ACP session can be recreated between
pipeline stages. If the session is recreated without an explicit model, Codex
may fall back to a local default. In the observed validation run, that fallback
was `gpt-5.3-codex`.

Use the pinned wrapper when running AutoResearchClaw:

```yaml
llm:
  primary_model: "gpt-5.5[medium]"
  acp:
    acpx_command: "/opt/huawei/explorer-env/dataset/ag_data/code/CairnLab/scripts/acpx-codex-gpt55-medium.sh"
```

The wrapper delegates to AutoResearchClaw's normal `acpx-node22` launcher while
prepending:

```bash
--model gpt-5.5[medium]
```

Launcher resolution stays outside the CairnLab core:

1. use `CAIRNLAB_ACPX_COMMAND` when set;
2. otherwise use a sibling `../AutoResearchClaw/experiments/arc_bench/scripts/acpx-node22`
   checkout when present;
3. otherwise fall back to `acpx-node22` on `PATH`.

Override the model or ACP command without editing the wrapper:

```bash
CAIRNLAB_CODEX_MODEL='gpt-5.5[medium]' \
CAIRNLAB_ACPX_COMMAND=/path/to/acpx-node22 \
python -m researchclaw run ...
```

This is not a CairnLab transition authority feature. It is a validation harness
control that keeps the external AutoResearch system on a selected Codex model so
CairnLab can import the resulting evidence and enforce claim lifecycle rules.

Validation note: on 2026-06-08, resuming the ML01 e2e run from Stage 15 with
this wrapper and `gpt-5.5[medium]` advanced the pipeline through Stage 23. The
run still produced degraded quality and paper-verifier failures, which CairnLab
imports as evidence diagnostics. Model pinning fixes ACP/Codex model drift; it
does not make upstream claims release-ready.
