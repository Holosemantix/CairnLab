from __future__ import annotations

from pathlib import Path

import typer

from . import __version__
from .engine import CairnProject
from .models import Actor, ClaimState
from .utils import actor_from_string

app = typer.Typer(help="CairnLab Research Claim Kernel CLI")
transition_app = typer.Typer(help="Claim lifecycle transition commands")
app.add_typer(transition_app, name="transition")


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
    typer.echo(f"Projected state: {result.projected_state}")
    typer.echo("Downstream:")
    for downstream in result.downstream_objects:
        typer.echo(f"- {downstream}")
    typer.echo("Events:")
    for event in result.events:
        typer.echo(f"- {event.type}: {event.reason}")


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
    )
    if json_output:
        typer.echo(decision.model_dump_json(indent=2))
    else:
        typer.echo(f"Decision: {decision.decision}")
        for reason_item in decision.blocking_reasons:
            typer.echo(f"- {reason_item}")


if __name__ == "__main__":
    app()
