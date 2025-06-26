# Ablation Experiments Plan

This document enumerates all ablation studies required to quantify the contribution of each major component in the Adaptive Timeline Segmentation pipeline (see § 2 of the paper).  Every experiment runs **one** modified variant against the **Full Model (FM)** baseline so that performance changes can be attributed unambiguously.

Global settings shared by all experiments:

* **Datasets**  Eight scholarly domains listed in § 3.1.
* **Metrics**  Consensus–Difference score \(\mathcal L\) and, where ground-truth exists, **F1@2yr**.
* **Evaluation protocol**  Re-run complete segmentation pipeline per domain with fixed random seeds.  Report mean metric values across domains and per-domain breakdowns.
* **Optimiser budget**  100 objective evaluations unless stated otherwise.
* **Notation**  Variants are prefixed with "–" when a component is removed; alternative implementations are given explicit tags (e.g. **LIN**, **RAND**).

---

## 1  Keyword-Volatility Shift Detector (KV)

**Research Question**  Does the keyword-based novelty score \(S_{\text{dir}}\) provide unique temporal shift signals beyond those captured by citation dynamics?

**Hypothesis**  Removing KV will reduce \(\mathcal L\) by degrading *Difference* metrics, as emerging topical keywords are no longer detected.

**Method**
1. **FM**  Full model with KV and CG detectors.
2. **–KV**  Disable KV; retain Citation-Gradient (CG) detector unchanged.

---

## 2  Citation-Gradient Shift Detector (CG)

**Research Question**  What is the impact of gradient- and acceleration-based citation anomalies on segmentation quality?

**Hypothesis**  Omitting CG will lower \(\mathcal L\) and F1@2yr because citation surges often coincide with recognisable domain shifts.

**Method**
1. **FM**  Full model.
2. **–CG**  Disable CG; keep KV active.

---

## 3  Confidence Fusion & Validation (FUS)

**Research Question**  Does multiplicative confidence boosting with citation support improve the precision/recall of detected shift years?

**Hypothesis**  Without fusion, more false-positive shift years will pass validation, decreasing overall quality.

**Method**
1. **FM**  Standard fusion (Equation 2 in paper).
2. **–FUS**  Skip fusion; validate directly on \(S_{\text{dir}}\) scores.

---

## 4  Boundary Refinement by Jaccard Crossing (BND)

**Research Question**  Is the boundary scan that selects the first year where keyword similarity flips necessary for accurate segment limits?

**Hypothesis**  Removing boundary refinement will yield mis-aligned or delayed boundaries, reducing both \(\mathcal L\) and F1.

**Method**
1. **FM**  Full pipeline.
2. **–BND**  Adopt each validated shift year itself as the segment boundary (no scan).

---

## 5  Length Constraints & Greedy Merge (LEN)

**Research Question**  How critical are minimum/maximum length constraints (\(3\le\ell\le50\)) and greedy merging for preventing pathological segmentations?

**Hypothesis**  Disabling constraints will increase variance in segment counts, hurting consensus and possibly difference scores.

**Method**
1. **FM**  With constraints.
2. **–LEN**  No length enforcement; keep raw boundaries.

---

## 6  Phrase Enrichment via YAKE (PHR)

**Research Question**  Do automatically mined multi-word phrases contribute meaningfully to shift detection and segment coherence?

**Hypothesis**  Excluding phrases will decrease both consensus and difference metrics because important multi-word terms are lost.

**Method**
1. **FM**  Author keywords + YAKE phrases (up to 10 per paper).
2. **–PHR**  Author keywords only.

---

## 7  Consensus Metrics Block (C-MET)

**Research Question**  Are intra-segment coherence metrics (C1–C3) necessary in the objective function?

**Hypothesis**  Removing C-metrics will encourage over-segmentation where segments are not internally coherent, lowering \(Q_{\text{cons}}\) and overall \(\mathcal L\).

**Method**
1. **FM**  Full objective.
2. **–C-MET**  Set weights of C1–C3 to zero; renormalise D-metrics.

---

## 8  Difference Metrics Block (D-MET)

**Research Question**  Do inter-segment difference metrics (D1–D3) meaningfully guide the optimiser?

**Hypothesis**  Without D-metrics the optimiser will merge dissimilar eras, lowering separation quality.

**Method**
1. **FM**.
2. **–D-MET**  Weights of D-metrics set to zero; C-metrics only.

---

## 9  Dynamic Normalisation & Feasibility Guard (DYN)

**Research Question**  Does adaptive rescaling plus the 0.9-\(\tau_{\text{cons}}\) guard stabilise optimisation and prevent degenerate solutions?

**Hypothesis**  Static scaling will cause larger variance in objective values, potentially leading to infeasible segmentations with low consensus.

**Method**
1. **FM**  Dynamic normalisation + guard.
2. **–DYN**  Static min-max scaling; guard removed.

---

## 10  Objective Scalarisation: Augmented Tchebycheff vs Linear (AWT)

**Research Question**  Does the max-style Augmented Tchebycheff scalarisation outperform a simple weighted sum?

**Hypothesis**  Linear combination will under-represent whichever criterion has lower magnitude, reducing Pareto balance.

**Method**
1. **FM**  Augmented Tchebycheff (\(\rho=0.1\)).
2. **LIN**  Replace with weighted sum \(w_c\hat Q_{\text{cons}}+w_d\hat Q_{\text{diff}}\).

---

## 11  Segment-Count Penalty (SCP)

**Research Question**  Is the exponential penalty around \(K_{\text{desired}}\) essential to discourage extreme segment counts?

**Hypothesis**  Removing SCP will increase frequency of very high or low segment counts, harming both metrics.

**Method**
1. **FM**.
2. **–SCP**  Penalty term removed from objective.

---

## 12  Optimiser Strategy (GP-BO)

**Research Question**  How much sample-efficiency and solution quality does the Gaussian-Process Bayesian optimiser provide over simpler search strategies?

**Hypothesis**  GP-BO will reach higher \(\mathcal L\) with the same or fewer evaluations compared with random or grid search.

**Method**
1. **FM**  GP-BO, 100 evaluations.
2. **RAND**  Pure random search, 100 evaluations.
3. **GRID**  Coarse grid (5×5×3×5 = 375 configs); report best after **first 100** evaluations for fair budget comparison and after full grid for upper-bound.

---

**Post-Study Analysis**   For each experiment compute ∆\(\mathcal L\) and ∆F1 relative to FM and provide qualitative explanations for observed trends.  A consolidated results table will accompany the camera-ready paper. 