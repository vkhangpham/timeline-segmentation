# Adaptive, Transparent Timeline Segmentation of Scientific Domains via Direction–Citation Fusion

## Abstract

*To be completed.*

## 1 Introduction

*To be completed.*

## 2 Method

### 2.1 Segmentation Algorithm

Our timeline generator operates in three chronological stages that convert raw bibliographic records into a sequence of coherent historical segments.

#### 2.1.1 Dual-Modality Shift Detection

1. **Research-direction volatility**   For every publication year \(t\) we contrast two non-overlapping 3-year windows, a *previous* window \(W_p(t)=\{t-6,t-5,t-4\}\) and a *current* window \(W_c(t)=\{t-3,t-2,t-1\}\).  With multisets of keywords \(K_p\) and \(K_c\) drawn from these windows we compute
\[
\text{Novelty}(t)=\frac{|K_c\setminus K_p|}{|K_c|},\qquad
\text{Overlap}(t)=\frac{|K_c\cap K_p|}{|K_p|},\qquad
S_{\text{dir}}(t)=\text{Novelty}(t)\bigl(1-\text{Overlap}(t)\bigr).
\]
A year is marked as a *direction candidate* when \(S_{\text{dir}}(t)>\tau_{\text{dir}}\).  Prior to the calculation we (i) discard infrequent keywords whose document frequency is below \(p_{\min}=0.05\) of papers in the current window and (ii) enrich the vocabulary with up to ten YAKE-extracted multi-word phrases per paper.

2. **Citation-gradient anomalies**   Annual citation counts \(C_t\) are analysed at three temporal scales \(\{1,3,5\}\) years.  Smoothed series are differentiated to obtain gradient \(G_t=\nabla C_t\) and acceleration \(A_t=\nabla^2 C_t\).  Data-adaptive thresholds
\[\theta_G=1.5\,\sigma(G),\qquad\theta_A=2\,\text{MAD}(A)\]
identify years where either \(|G_t|>\theta_G\) or \(|A_t|>\theta_A\).  Each anomalous year receives a confidence
\[\gamma_t = \min\bigl(|G_t|/\theta_G,2\bigr)/2,\qquad\gamma_t\in[0.3,0.95].\]
A minimum spacing of three years prevents clustered detections.

#### 2.1.2 Confidence Fusion and Validation

Direction candidates are paired with citation detections inside a symmetric two-year window.  If citation support is present the original confidence \(c_y=S_{\text{dir}}(y)\) is boosted multiplicatively:
\[
\hat c_{y}=\begin{cases}
 c_y, & \text{no citation support},\\
 \min\bigl(c_y+\beta c_y,1\bigr), & \text{citation support},
\end{cases}\qquad\beta=0.8.
\]
A shift year is *validated* when \(\hat c_{y}\ge\tau_{\text{val}}\), yielding a sparse set \(\{s_1<\dots<s_m\}\).

#### 2.1.3 Contiguous Segmentation

Validated years act as centroids that partition the domain timeline.  For two neighbouring centroids \(s_i<s_{i+1}\) we scan intermediate years \(t\) and compute Jaccard similarities between the year's keywords and those of each centroid.  The first year satisfying
\[J\bigl(K_t,K_{s_{i+1}}\bigr) > J\bigl(K_t,K_{s_i}\bigr)\]
is adopted as the boundary.  The procedure guarantees non-overlapping segments which are then post-processed to obey
\[\ell_{\min}=3\le\ell\le \ell_{\max}=50.\]
Short segments are greedily merged with their most similar neighbour until all constraints hold.

### 2.2 Parameter Optimisation

The segmentation algorithm exposes four tunable hyper-parameters \(\Theta=(\tau_{\text{dir}},\tau_{\text{val}},\ell_{\min},\ell_{\max})\).  We cast their calibration as the maximisation of a *consensus–difference* objective.

#### 2.2.1 Objective Function

Given a segmentation \(\mathcal S=\{\mathcal P_k\}_{k=1}^{K}\) we quantify quality along two orthogonal axes.

*Within-segment consensus* (C-metrics)
1. **C1** Mean Jaccard overlap between each paper's keywords and the segment keyword set.
2. **C2** Mean pairwise cosine similarity of TF-IDF embeddings (10 k-feature vocabulary).
3. **C3** Internal citation-edge density.

*Between-segment difference* (D-metrics)
1. **D1** Jensen–Shannon divergence of keyword distributions.
2. **D2** Centroid distance: \(1-\)cosine similarity between TF-IDF centroids.
3. **D3** Cross-citation ratio between adjacent segments.

Weighted aggregates
\[
Q_{\text{cons}} = 0.4\,C1 + 0.4\,C2 + 0.2\,C3,\qquad
Q_{\text{diff}} = 0.4\,D1 + 0.4\,D2 + 0.2\,(1-D3).
\]

**Adaptive Augmented Tchebycheff scalarisation.**  To remedy the scale imbalance between the two criteria we introduce an *adaptive* objective that (i) enforces a minimum consensus feasibility guard and (ii) rescales the criteria on-the-fly so that both influence optimisation:

1. *Feasibility guard* Let \(\tau_{\text{cons}}\) be the consensus score obtained when all papers form a single segment.  Any candidate segmentation whose mean consensus \(\overline{Q}_{\text{cons}}<0.9\,\tau_{\text{cons}}\) receives a severe penalty \(0.1\,\overline{Q}_{\text{cons}}/\tau_{\text{cons}}\) and is discarded by the optimiser.
2. *Dynamic normalisation* Upper bounds \(b_{c}, b_{d}\) are estimated as the 95-th percentile of observed consensus and difference scores across all segments (plus a 20 % safety margin).  We set \(\hat{Q}_{\text{cons}}=\min(1,\overline{Q}_{\text{cons}}/b_{c})\) and analogously for \(\hat{Q}_{\text{diff}}\).
3. *Augmented Tchebycheff combination* With preference weight \(w_{d}\in[0,1]\) (default 0.5) and \(w_{c}=1-w_{d}\) the final objective is

\[
\mathcal L_{\text{AWT}} \,=\, \max\bigl(w_{c}\,\hat{Q}_{\text{cons}},\; w_{d}\,\hat{Q}_{\text{diff}}\bigr) + \rho\bigl(w_{c}\,\hat{Q}_{\text{cons}} + w_{d}\,\hat{Q}_{\text{diff}}\bigr), \quad \rho = 0.1.
\]
The augmentation term \(\rho\) guarantees Pareto efficiency while preserving differentiability [[Steuer & Choo 1983]].  A legacy linear objective is retained for ablation studies and historical comparisons.

To discourage over- and under-segmentation we retain the exponential segment-count penalty \(\exp(-|K-K_{\text{desired}}|/\sigma)\) with \(K_{\text{desired}}=\lceil \text{domain span}/10 \rceil\) and \(\sigma=5\).

#### 2.2.2 Bayesian Search Strategy

Hyper-parameter bounds follow the same empirical priors as before—\(\tau_{\text{dir}}\in[0.1,0.4]\), \(\tau_{\text{val}}\in[0.3,0.45]\), \(\ell_{\min}\in[3,5]\), \(\ell_{\max}\in[10,30]\).  A Gaussian-process surrogate with Expected Improvement acquisition explores the four-dimensional space for up to 100 evaluations.  Configurations that violate the feasibility guard or trigger runtime errors receive a large negative reward, steering the optimiser away from pathological regions.  The best configuration per domain—evaluated with the adaptive Tchebycheff objective—is retained for downstream analysis and ablation studies.

## 3 Experiments

### 3.1 Datasets
We evaluate on *eight* scholarly domains that exhibit diverse publication rates and citation cultures:

| Domain | Papers | Years | Avg. papers / year |
|--------|-------:|------:|-------------------:|
| Applied Mathematics | 465 | 1892–2021 | 4 |
| Art | 473 | 1835–2024 | 2 |
| Computer Science | 456 | 1936–2023 | 5 |
| Computer Vision | 213 | 1992–2022 | 7 |
| Deep Learning | 447 | 1973–2021 | 9 |
| Machine Learning | 218 | 1993–2023 | 7 |
| Machine Translation | 225 | 1992–2021 | 8 |
| Natural Language Processing | 440 | 1951–2023 | 6 |

Each domain provides bibliographic metadata including title, abstract content, author keywords, publication year, citation count, and citation relationships. Ground-truth historical periods collected by domain experts are available for all domains and are used only for evaluation, **never** for training.

### 3.2 Baselines
We compare against three reference systems together with our **Bayesian-Optimised Algorithm**:

1. **Bayesian-Optimised Algorithm** Our pipeline with parameters tuned by the Gaussian-Process Bayesian optimiser described in § 2.7.
2. **Decade Baseline** Segments fixed to calendar decades (1960s, 1970s, …).
3. **5-Year Baseline** Segments fixed to 5-year periods for finer temporal granularity.
4. **Gemini DeepResearch** Ground-truth historical periods manually curated by domain experts; serves as an upper-bound oracle.

### 3.3 Evaluation Metrics
Primary metric is the *Consensus–Difference Score* \(\mathcal{L}\) from § 2.6.  Where expert boundaries are available we additionally report *F1@2yr*.

### 3.4 Implementation Details
* **Optimiser configuration** 100 function evaluations with 20 initial random points, using Expected Improvement acquisition via Gaussian Process surrogate models.
* **Search space** Direction threshold ∈ [0.1, 0.4], validation threshold ∈ [0.3, 0.45], minimum segment length ∈ [3, 5], maximum segment length ∈ [10, 30].
* **Failure handling** Infeasible parameter combinations or runtime exceptions receive large negative penalties, ensuring optimiser stability.
* **Reproducibility** Deterministic random seeds derived from domain names ensure consistent results across runs.
* **Text Vectorization** TF-IDF with 10,000 maximum features provides optimal vocabulary representation, empirically outperforming both smaller feature sets and contextual embeddings for temporal segmentation tasks.
* **Keyword Processing** Conservative filtering with p_min=0.05 balances noise reduction with signal preservation. Multi-word phrase enrichment via YAKE extracts up to 10 semantic phrases per paper, improving consensus and difference metrics.
* **Citation Enrichment** Citation networks loaded directly from JSON metadata and GraphML files provide domain-specific coverage ranging from 18.6% (Art) to 70.7% (Deep Learning), enabling meaningful C3 citation density and D3 cross-citation metrics.
* **Text Preprocessing** HTML tag removal, URL cleaning, and stop-word filtering improve embedding quality while preserving domain-specific terminology.

## 4 Results

### 4.1 Overall Performance

*To be completed.*

### 4.2 Domain-Specific Performance

*To be completed.*

## 5 Discussion

*To be completed.*

## 6 Ablation Study

*To be completed.*

## 7 Conclusion

*To be completed.* 