# XSS Sentinel Paper — Completion Walkthrough

## Deliverables Produced

| File | Location | Purpose |
|------|----------|---------|
| [`XSS_Sentinel_Paper.tex`](file:///c:/Users/bhara/OneDrive/Desktop/XSS-Sentinel/XSS_Sentinel_Paper.tex) | Workspace root | Full IEEE IEEEtran paper |
| [`references.bib`](file:///c:/Users/bhara/OneDrive/Desktop/XSS-Sentinel/references.bib) | Workspace root | 10 IEEE-style BibTeX entries |
| [`claim_evidence_map.md`](file:///C:/Users/bhara/.gemini/antigravity-ide/brain/7ba6cfeb-c00d-4478-8346-ad56d8ffff27/claim_evidence_map.md) | Artifacts dir | Every number traced to results/ file+line |
| [`changelog.md`](file:///C:/Users/bhara/.gemini/antigravity-ide/brain/7ba6cfeb-c00d-4478-8346-ad56d8ffff27/changelog.md) | Artifacts dir | What changed, what's new, open gaps |
| [`repo_inventory.md`](file:///C:/Users/bhara/.gemini/antigravity-ide/brain/7ba6cfeb-c00d-4478-8346-ad56d8ffff27/repo_inventory.md) | Artifacts dir | Full deterministic repo scan |

## Number Verification Summary

- **Total numbers in Tables I–IV**: 82
- **Verified from results/ files**: 82 / 82 (100%)
- **Unverifiable numbers silently included**: **0**

## Table Evidence Sources

| Table | Source File(s) |
|-------|---------------|
| Table I (Baselines) | `results/caxf_{tfidf,char_cnn,sentence_embedding}_results/baselines/train_*.txt` |
| Table II (6-Ensemble × 3-Feature F1) | `results/caxf_*_results/ensemble_comparison_seed_42.txt`, Section 1 |
| Table III (Per-Class F1, TF-IDF) | `results/caxf_tfidf_results/ensemble_comparison_seed_42.txt`, Section 2 |
| Table IV (Ablation) | `results/ensemble_ablation_study_seed_42.txt`, Lines 44-48 |

## Open Gaps (Explicit — Not Silently Included)

1. **FLAN-T5**: mentioned in draft, no code/results in repo → **omitted from paper**
2. **Base paper original metrics**: PDF not parseable → cited as reference only, no numerical claims
3. **Kaggle dataset URL**: best-known URL used in references.bib → verify before submission
4. **MiniLM parameter count**: only model name confirmed → not stated numerically
5. **Author names/affiliation**: not in repo → placeholder in `\author{}` block

## Paper Structure

| Section | Content |
|---------|---------|
| I. Introduction | 5 contributions listed |
| II. Related Work | XSS detection, ensemble methods, SMOTE |
| III. Dataset | 10,917 samples, 70:30 split, SMOTE stats |
| IV. Methodology | CAXF (Eq.1), TF-IDF (Eq.2-3), CharCNN (Eq.4-5, Alg.2), MiniLM (Eq.6, Alg.3), SMOTE (Alg.4), 6 ensemble decision rules (Eq.7-17, Alg.5-6) |
| V. Experimental Setup | Seed, library versions, metrics definition |
| VI. Results | Tables I–IV, key findings |
| VII. Discussion | Obfuscation resilience, DOM dynamics, efficiency, selection guide |
| VIII. Conclusion | Summary + future work |

## LaTeX Compilation Instructions

```bash
cd "c:\Users\bhara\OneDrive\Desktop\XSS-Sentinel"
pdflatex XSS_Sentinel_Paper.tex
bibtex XSS_Sentinel_Paper
pdflatex XSS_Sentinel_Paper.tex
pdflatex XSS_Sentinel_Paper.tex
```

**Note**: `pdflatex` was not found in PATH. Install TeX Live or MiKTeX before compiling.
Syntax validation confirmed: 40 begin/end pairs balanced, 188 dollar signs (even), all `%` are valid LaTeX comments.

## Action Required Before Submission

1. Fill in author names and affiliations in `\author{}` block
2. Verify the Kaggle dataset URL in `references.bib`
3. Fill in the CAXF-LCCDE base paper citation (`caxf_lccde2024`) with actual authors/venue
4. Install pdflatex and compile to confirm no remaining issues
5. Review Open Gaps 1-5 above and decide on each
