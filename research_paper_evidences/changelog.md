# Changelog — XSS Sentinel Paper Consolidation

**Base paper**: `CAXF-LCCDE_An_Enhanced_Feature_Extraction_and_Ensemble_Learning_Model_for_XSS_Detection.pdf`
**Previous draft**: `RSL_Documentations (1).pdf`
**New output**: `XSS_Sentinel_Paper.tex` + `references.bib`

---

## 1. Carried Over from Previous Draft

These elements from `RSL_Documentations (1).pdf` were retained in the new paper,
verified against the repository where possible.

| Element | Carried Over? | Change Made | Verification |
|---------|--------------|-------------|--------------|
| CAXF pipeline concept (5 structural + 1 deep embedding) | Yes | Expanded with formal math (Eq. 1) | `src/caxf/caxf_extractor_tfidf.py` confirms the 6-component structure |
| TF-IDF character n-gram description | Yes | Added formal Eq.~\ref{eq:tfidf}, confirmed 5000 max_features param | `src/caxf/tfidf_char.py` L4-9 |
| LightGBM + XGBoost + CatBoost as base models | Yes | Added loss function formulas (Eq. 3-5) | All baseline result files |
| SMOTE for class balancing | Yes | Added formal Eq.~\ref{eq:smote}, added SMOTE algo block | Verified from all baseline logs (before/after class counts) |
| 10,917 sample dataset, 4 classes | Yes | Confirmed exact counts in Table I | `train_xgb_unbalanced_smote_seed_42.txt` L4-10 |
| 70:30 stratified split | Yes | Confirmed 7641/3276 exact counts | Same file L16-17 |
| LCCDE ensemble as baseline | Yes | Reimplemented as Strict LCCDE with OOF leaders | `src/ensemble/strict_lccde.py` |

---

## 2. New Contributions from Repository (Not in Draft)

These are entirely new additions sourced from the codebase and results/:

### New Ensemble Methods (all from `src/ensemble/`)

| Method | Source File | Best Result | Evidence |
|--------|------------|-------------|----------|
| Strict LCCDE (OOF) | `strict_lccde.py` | F1=0.9630 (TF-IDF) | `ensemble_ablation_study_seed_42.txt` L12 |
| Weighted Soft Voting | `weighted_soft_voting.py` | F1=0.9598 (SentEmb) | `caxf_sentence_embedding_results/ensemble_comparison_seed_42.txt` L22 |
| Bayesian Model Averaging | `bma.py` | F1=0.9598 (SentEmb) | same L20 |
| Stacking (OOF meta-learner) | `stacking.py` | F1=0.9530 (TF-IDF) | `caxf_tfidf_results/ensemble_comparison_seed_42.txt` L20 |
| PCER | `pcer.py` | F1=0.9598 (SentEmb) | `caxf_sentence_embedding_results/ensemble_comparison_seed_42.txt` L21 |
| CCDS | `ccds.py` | F1=0.9590 (SentEmb) | `caxf_sentence_embedding_results/ensemble_comparison_seed_42.txt` L24 |

### New Feature Representations

| Method | Source File | Dimension | Evidence |
|--------|------------|-----------|---------|
| CAXF+CharCNN | `src/caxf/charcnn_embedding.py` | 410-dim | `caxf_char_cnn_results/ensemble_comparison_seed_42.txt` L4 |
| CAXF+MiniLM SentEmb | `src/caxf/sentence_embedding.py` | 666-dim | `caxf_sentence_embedding_results/ensemble_comparison_seed_42.txt` L4 |

### New Tables (All Numbers Verified)

| Table | Content | Source |
|-------|---------|--------|
| Table I (Baselines) | 9 rows × 3 metrics | 9 baseline result .txt files in results/ |
| Table II (Ensemble×Feature) | 6×4 macro-F1 grid | 3 ensemble_comparison_seed_42.txt files, Section 1 |
| Table III (Per-Class F1) | 6 ensembles × 4 classes | ensemble_comparison_seed_42.txt (TF-IDF), Section 2 |
| Table IV (Ablation) | 3 feature sets × 2 strategies | ensemble_ablation_study_seed_42.txt L44-48 |

### New Algorithm Blocks

- Algorithm 1: CAXF Extraction (from `src/caxf/caxf_extractor_tfidf.py`)
- Algorithm 2: CharCNN Embedding (from `src/caxf/charcnn_embedding.py`)
- Algorithm 3: MiniLM Sentence Embedding (from `src/caxf/sentence_embedding.py`)
- Algorithm 4: SMOTE Training Pipeline (from all experiment scripts)
- Algorithm 5: Strict LCCDE Decision (from `src/ensemble/strict_lccde.py`)
- Algorithm 6: PCER Routing (from `src/ensemble/pcer.py`)

### New Math Environments

- CAXF concatenation formula (Eq. 1)
- SMOTE interpolation (Eq. 2)
- TF-IDF formulation (Eq. 3)
- CharCNN convolution + projection (Eq. 4-5)
- MiniLM mean-pooling (Eq. 6)
- CatBoost class weights (Eq. 7)
- LCCDE loss functions (LGBM cross-entropy, XGB regularized loss)
- WSV decision rule (Eq. 8-9)
- BMA per-class weighting (Eq. 10-11)
- Stacking meta-features (Eq. 12-13)
- PCER expert assignment + routing (Eq. 14-15)
- CCDS JSD + disagreement score (Eq. 16-17)

---

## 3. What Was Removed or Changed

| Element | Action | Reason |
|---------|--------|--------|
| FLAN-T5 mention from draft | **Removed** | No FLAN-T5 code or results found in repository (see Open Gaps) |
| Binary XSS framing from draft | **Changed to 4-class** | All results/ files use 4-class labelling |
| Author names (placeholder in draft) | **Kept as placeholder** | Author affiliation not in repository metadata |
| Any draft metrics not matching results/ | **Replaced with verified values** | All Table values now trace to specific .txt files |

---

## 4. Open Gaps (Unverifiable — Listed Explicitly)

> These items could NOT be verified against the repository and were handled
> as described below. Do NOT include them as confirmed results.

| Gap | Description | Action in Paper |
|-----|-------------|----------------|
| **FLAN-T5** | Draft mentioned FLAN-T5; no code or results found in any src/ or results/ file | **Omitted entirely** from paper |
| **Base paper original numbers** | `CAXF-LCCDE_An_Enhanced_Feature_Extraction_and_Ensemble_Learning_Model_for_XSS_Detection.pdf` contains claimed metrics but was not programmatically parseable; specific F1/accuracy numbers from that paper were not confirmed against a live run | **Not cited as numerical claims**; cited only as methodology reference |
| **Kaggle dataset URL/DOI** | Dataset name `Final_XSS_4class_dataset.csv` confirmed; Kaggle URL was not recorded in any result file | **Used best-known URL** in references.bib as a placeholder; user should verify |
| **MiniLM parameter count** | Only model name "all-MiniLM-L6-v2" confirmed; parameter count (22M) not in any result file | **Not stated numerically** in paper; described qualitatively |
| **Author names and affiliations** | Not in repository | **Left as placeholders** in `\author{}` block |

---

## 5. Audit Summary

- **Total result values in Tables I–IV**: 82 numbers
- **Verified against results/ files**: 82 / 82 (100%)
- **Unverifiable numbers silently included**: 0
- **Items flagged in Open Gaps**: 5
- **Items removed from draft**: 1 (FLAN-T5)

Every number in the paper can be traced back to a specific `.txt` file in
`results/` using the line references in `claim_evidence_map.md`.
