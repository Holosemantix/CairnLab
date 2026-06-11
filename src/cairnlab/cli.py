from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__
from .adapters import AdapterSelectionError, adapter_names, detect_adapters, select_adapter
from .engine import CairnProject
from .models import Actor, ClaimState
from .transition_explain import explain_transition_decision, render_transition_explanation_text
from .utils import actor_from_string

app = typer.Typer(help="CairnLab Research Claim Kernel CLI")
transition_app = typer.Typer(help="Claim lifecycle transition commands")
adapter_app = typer.Typer(help="AutoResearch manifest adapter commands")
app.add_typer(transition_app, name="transition")
app.add_typer(adapter_app, name="adapter")


@app.command()
def version() -> None:
    typer.echo(f"cairnlab {__version__}")


@app.command()
def init(path: Path = typer.Option(Path("."), "--path")) -> None:
    project = CairnProject.open(path)
    project.init()
    typer.echo(f"Initialized CairnLab project at {path / '.cairn'}")


@app.command("import-case")
def import_case(case_path: Path, path: Path = typer.Option(Path("."), "--path")) -> None:
    project = CairnProject.open(path)
    result = project.import_case(case_path)
    typer.echo(
        f"Imported {result.case_id}: {result.claims} claims, "
        f"{result.evidence} evidence objects, {result.relations} relations"
    )


@app.command("import-external")
def import_external(
    source_path: Path,
    adapter: str = typer.Option("auto", "--adapter", help="Adapter name, or auto for deterministic detection."),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        selected_adapter = select_adapter(source_path, adapter_name=adapter)
        export = selected_adapter.export_case(source_path)
    except AdapterSelectionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    project = CairnProject.open(path)
    result = project.import_claim_case(export.case)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "adapter": selected_adapter.name,
                    "case_id": result.case_id,
                    "claims": result.claims,
                    "evidence": result.evidence,
                    "relations": result.relations,
                    "diagnostics": [item.model_dump(mode="json") for item in export.diagnostics],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(
        f"Imported {result.case_id}: {result.claims} claims, "
        f"{result.evidence} evidence objects, {result.relations} relations"
    )
    for diagnostic in export.diagnostics:
        typer.echo(f"{diagnostic.level}: {diagnostic.message}")


@app.command()
def validate(
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    report = project.validate()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(f"Recommendation: {report.recommendation}")
        for reason in report.reasons:
            typer.echo(f"- {reason}")


@app.command()
def trace(
    object_id: str,
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    result = project.trace(object_id)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"Object: {result.object_id}")
    if result.observed_state is not None:
        typer.echo(f"Observed state: {result.observed_state}")
    if result.authority_state is not None:
        typer.echo(f"Authority state: {result.authority_state}")
    typer.echo(f"Projected state: {result.projected_state}")
    typer.echo("Downstream:")
    for downstream in result.downstream_objects:
        typer.echo(f"- {downstream}")
    typer.echo("Events:")
    for event in result.events:
        typer.echo(f"- {event.type}: {event.reason}")


@app.command("decision-trace")
def decision_trace(
    claim_id: str,
    transition: str | None = typer.Option(None, "--transition"),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    package = project.decision_trace_package(claim_id, transition=transition)
    if json_output:
        typer.echo(package.model_dump_json(indent=2))
        return
    typer.echo(f"Package: {package.package.id}")
    typer.echo(f"Claim: {package.package.claim}")
    typer.echo(f"Export hash: {package.package.export_hash}")
    typer.echo(f"Evidence: {len(package.evidence)}")
    typer.echo(f"Relations: {len(package.relations)}")
    typer.echo(f"Events: {len(package.events)}")


@app.command()
def affected(
    object_id: str,
    reason: str = typer.Option("affected preview", "--reason"),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    plan = project.plan_revert(object_id, reason=reason)
    if json_output:
        typer.echo(plan.model_dump_json(indent=2))
    else:
        for item in plan.affected:
            typer.echo(f"{item.action}: {item.id} -> {item.proposed_event_type}")


@app.command()
def revert(
    target_id: str,
    reason: str = typer.Option(..., "--reason"),
    apply: bool = typer.Option(False, "--apply"),
    plan_only: bool = typer.Option(False, "--plan-only"),
    actor: str = typer.Option("user:unknown", "--actor"),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    del plan_only
    project = CairnProject.open(path)
    actor_model = actor_from_string(actor)
    plan = project.plan_revert(target_id, reason=reason, actor=actor_model)
    if apply:
        events = project.apply_plan(plan)
        if json_output:
            typer.echo("[" + ",".join(event.model_dump_json() for event in events) + "]")
        else:
            typer.echo(f"Appended {len(events)} events")
        return
    if json_output:
        typer.echo(plan.model_dump_json(indent=2))
    else:
        for item in plan.affected:
            typer.echo(f"{item.action}: {item.id} -> {item.proposed_event_type}")


@transition_app.command("request")
def transition_request(
    claim_id: str,
    to: ClaimState = typer.Option(..., "--to"),
    reason: str = typer.Option(..., "--reason"),
    actor: str = typer.Option("user:unknown", "--actor"),
    force: bool = typer.Option(False, "--force"),
    apply: bool = typer.Option(False, "--apply"),
    record_blocked: bool = typer.Option(False, "--record-blocked"),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    actor_model: Actor = actor_from_string(actor)
    decision = project.request_transition(
        claim_id=claim_id,
        target_state=to,
        actor=actor_model,
        reason=reason,
        force=force,
        apply=apply,
        record_blocked=record_blocked,
    )
    if json_output:
        typer.echo(decision.model_dump_json(indent=2))
    else:
        typer.echo(f"Decision: {decision.decision}")
        for reason_item in decision.blocking_reasons:
            typer.echo(f"- {reason_item}")
        if apply and decision.decision == "allowed":
            typer.echo(f"Appended {len(decision.events)} transition event(s)")
        elif record_blocked and decision.decision == "blocked":
            typer.echo(f"Recorded {len(decision.events)} blocked transition event(s)")


@transition_app.command("explain")
def transition_explain(
    claim_id: str,
    to: ClaimState = typer.Option(..., "--to"),
    reason: str = typer.Option(..., "--reason"),
    actor: str = typer.Option("user:unknown", "--actor"),
    force: bool = typer.Option(False, "--force"),
    path: Path = typer.Option(Path("."), "--path"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    project = CairnProject.open(path)
    decision = project.request_transition(
        claim_id=claim_id,
        target_state=to,
        actor=actor_from_string(actor),
        reason=reason,
        force=force,
    )
    explanation = explain_transition_decision(decision)
    if json_output:
        typer.echo(explanation.model_dump_json(indent=2))
        return
    typer.echo(render_transition_explanation_text(explanation))


@adapter_app.command("detect")
def adapter_detect(
    source_path: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    matches = detect_adapters(source_path)
    payload = {
        "path": str(source_path),
        "matches": [adapter.name for adapter in matches],
        "available": list(adapter_names()),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if matches:
        for adapter in matches:
            typer.echo(adapter.name)
    else:
        typer.echo("No adapter detected.")


if __name__ == "__main__":
    app()
