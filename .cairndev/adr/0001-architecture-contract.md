# ADR 0001: Architecture Contract as a First-Class Development Artifact

Status: accepted

## Context

Agentic coding tools can implement changes quickly, but project-level design taste is often
repeated manually in prompts. The same preferences should be versioned with the repository
instead of restated in every task.

## Decision

Represent engineering taste as a repo-local, machine-readable design contract in
`.cairndev/contract.yaml`, plus human-readable instructions in `AGENTS.md` and a reusable
Codex skill.

## Consequences

- Agents can read the same contract before every task.
- A CLI can check a subset of the contract deterministically.
- Human review can focus on meaningful architectural tradeoffs.
- The contract must remain small enough to be read and followed.
