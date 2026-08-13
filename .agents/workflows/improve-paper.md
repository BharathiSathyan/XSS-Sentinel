---
description: Take an uploaded base research paper plus the project codebase (which holds all results) and produce an improved, evidence-checked revision with a full changelog. Use for "improve/update this paper using our results/code" requests.
---

# /improve-paper

## Steps

1. **Confirm inputs.** Identify the uploaded base paper (PDF/LaTeX/docx) and the codebase root. If either is ambiguous, ask once before proceeding.

2. **Load the rules.** Read `.agents/skills/paper-improver/SKILL.md` first — every step below must follow its evidence rules (no invented numbers, no silent deletions, every change traced to a real file).

3. **Extract the base paper.** Read it directly and write `paper_claims.md`: every claim, key number, table, figure, related-work entry, and stated contribution, listed section by section.

4. **Inventory the codebase.** Run:
   ```
   python .agents/skills/paper-improver/scripts/inventory_repo.py <codebase_root> --out repo_inventory.md
   ```
   Review `repo_inventory.md`, prioritizing ⭐-flagged (results/metrics/log-keyword) files.

5. **Build the claim-evidence map.** Write `claim_evidence_map.md`. For every entry in `paper_claims.md`, find the matching file in `repo_inventory.md` and tag it:
   - `matches` — repo confirms the paper's number
   - `outdated` — repo has a newer/better number
   - `missing` — no supporting evidence found
   - `contradicted` — repo evidence disagrees with the paper

6. **List improvement opportunities.** From the inventory, note: results/ablations/baselines that exist in the repo but aren't in the paper yet, and any paper numbers the repo has since superseded.

7. **Write the revision plan** before touching prose: a section-by-section list of additions, edits, and removals, each tagged with its evidence source (file path from `repo_inventory.md`).

8. **Apply the revision to a copy.** Never overwrite the original base paper — write the revised version to a new file. Regenerate any figures/tables from the underlying data or plotting scripts, not by hand-editing images.

9. **Cross-check.** Re-scan the revised draft and confirm every number/claim still matches an entry in `claim_evidence_map.md`.

10. **Write `changelog.md`.** For each change: what changed, why, and which repo file backs it. Include an explicit "Open gaps" section listing anything still `missing` or `contradicted`.

11. **Present results.** Hand back: the revised paper, `changelog.md`, and `claim_evidence_map.md`. Do not describe the paper as "ready" if any high-value claim is still unresolved — state that plainly instead.

## Notes
- Keep `paper_claims.md`, `repo_inventory.md`, `claim_evidence_map.md`, and `changelog.md` — they're the audit trail, not scratch files to discard.
- If the codebase is large, step 4's script keeps the scan deterministic and cheap; don't substitute ad hoc directory browsing for it.