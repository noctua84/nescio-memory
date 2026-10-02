# Superpowers: Plans, Specs, and Scratch

This directory holds design documentation and implementation plans for multi-step changes to nescio-memory.

## Layout

**`specs/`** — Design specifications in the form `YYYY-MM-DD-<slug>-design.md`. Each spec documents the **what** and **why** for a change, the tradeoffs, and the invariants that code must preserve. A spec is written before implementation begins.

**`plans/`** — Implementation plans in the form `YYYY-MM-DD-<slug>.md`. Each plan breaks down a spec into numbered, actionable tasks with clear file lists, interfaces, and pass/fail criteria. Plans are the input to agentic workers.

**`../reports/`** (i.e. `docs/reports/`) — Audit and retrospective reports documenting decisions, findings, and outcomes. This is a sibling directory, not under `docs/superpowers/`.

## Scratch Files: Per-Plan Namespacing

Agent workers write task briefs and reports to `.superpowers/sdd/<plan-slug>/task-<N>-{brief,report}.md`, where `<plan-slug>` is the plan filename minus the date prefix and `.md`. For example, the plan `2026-10-01-hnsw-recall.md` produces files like:

```
.superpowers/sdd/hnsw-recall/task-1-brief.md
.superpowers/sdd/hnsw-recall/task-1-report.md
.superpowers/sdd/hnsw-recall/task-2-report.md
```

**Why plan-based namespacing?** Without it, every plan's "task 2" overwrites the same flat file. When task-2-report.md hasn't been written yet, the stale report from a different plan is silently read as fresh — its contents and structure are intact, making it look valid. This happened during the hnsw-recall plan run (#12, recorded in #21): `task-2-report.md` still held a complete report from the stabilization-items-2-7 plan, citing a commit that did not exist on the hnsw-recall branch. Only its header lines revealed it was stale. **Never use the flat `.superpowers/sdd/task-<N>-*.md` form.**

## Verifying a Report Before Using It

Every task report begins with:

```
Branch: <branch-name>
Commit: <commit-sha>
```

Before trusting a report's contents, verify both lines match the current execution:

- **Branch matches:** Check that the report's `Branch:` matches your current branch (`git symbolic-ref -q HEAD` or `git rev-parse --abbrev-ref HEAD`).
- **Commit exists on this branch:** Confirm the commit is an ancestor of `HEAD` with `git merge-base --is-ancestor <commit-sha> HEAD`. Exit 0 means the commit is on this branch; exit 1 means it isn't.

If either check fails, the report is stale — treat it as invalid regardless of what it contains. (This header check is what caught the stale report in #12.)

## Never Commit `.superpowers/`

The `.superpowers/` directory is gitignored and holds **machine-local scratch only**. Task briefs and reports are transient working documents, not durable outcomes. All results must be committed to `plans/`, `specs/`, `docs/reports/`, or the main codebase — never to `.superpowers/`.

Before a task agent finishes, verify that all durable work landed in the repository and that `.superpowers/sdd/` contains only briefs and reports.
