---
name: paper-improver
description: Use when the user uploads an existing research paper (PDF/LaTeX/docx) as a "base paper" and wants it revised or improved using results and code from their project repository. Covers mapping paper claims to repo evidence, finding new or better results not yet included, and producing a revised, evidence-grounded draft with a changelog of what changed and why.
---

# Paper Improver Skill

## Mandate
- The uploaded paper is a base draft, not ground truth. Every number and claim in it must be checked against the codebase before it is kept, changed, or cut.
- Every improvement (new result, stronger baseline, better number, new ablation) must trace to a real file in the repo: a results table, log, metrics JSON, notebook output, or script. Never invent or "smooth over" a number.
- Track every change in a changelog so the user can see exactly what changed, why, and what evidence backs it.

## Process
1. Read the base paper fully and extract: claims, key numbers, tables, figures, related-work list, stated contributions.
2. Inventory the codebase (`scripts/inventory_repo.py` gives a deterministic first pass) to find: results/metrics files, experiment logs, checkpoints, config files, notebooks, existing figures/tables.
3. Build a claim-evidence map. For each claim/number in the paper, find matching evidence in the repo and mark it:
   - `matches` — repo confirms the paper's number
   - `outdated` — repo has a newer/better number than what's in the paper
   - `missing` — no evidence found in the repo
   - `contradicted` — repo evidence disagrees with the paper
4. Identify improvement opportunities: results in the repo that aren't in the paper yet (new experiments, ablations, baselines, larger sweeps), and any paper numbers that are stale versus the repo's latest run.
5. Propose a section-by-section revision plan before touching prose: list additions, edits, and removals, each tagged with the evidence backing it.
6. Apply the revision to a **copy** of the paper — never overwrite the original. Regenerate figures/tables from the underlying data or plotting scripts rather than hand-editing image files.
7. Flag anything that could not be verified instead of silently keeping or dropping it.
8. Deliver: the revised paper, a changelog (what changed + why + evidence source), and a list of open gaps needing human judgment.

## Evidence rules
- A claim only counts as supported if you can point to a specific file (and ideally line/cell/column) in the repo.
- If the repo has no evidence for something the base paper already claims, don't delete it silently — flag it in the changelog as "unverified, kept from original" and let the user decide.
- Never fabricate citations, ablations, baselines, or numbers to fill a gap.
- Generated/illustrative figures (e.g. a redrawn architecture diagram) are fine for exposition but must never stand in for a results figure — those must come from real data.

## Reference
`scripts/inventory_repo.py` — deterministic repo scan for results/metrics/log-like files. Run this before manually reviewing the codebase; don't rely on ad hoc browsing to find evidence.