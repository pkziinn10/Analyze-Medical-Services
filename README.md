# Analysis and Prediction of Medical Service Utilization Among Adults Aged 50 and Older

**Authors:** Pedro Kauan Silveira Silva, Mikeyas Brito dos Santos, Wesley Barbosa Silva, Bruno Riccelli dos Santos Silva, Wellington Franco, Paulo Cesar Cortez, and Andressa G. Moreira  
**Affiliation:** Federal University of Ceará (UFC), Brazil

## Executive summary

This project presents a reproducible comparison of supervised machine-learning classifiers for a binary formulation of healthcare service utilization among adults aged 50 years and older. The study uses 714 observations from the National Poll on Healthy Aging (NPHA), preserves the observed class distribution, and evaluates baseline, fold-specific feature-selection, cost-sensitive, ensemble, and threshold-selection scenarios. Nested stratified cross-validation and out-of-fold predictions provide the basis for performance estimates.

The final results indicate limited discrimination between the two utilization classes. Cost-sensitive weighting and internal threshold selection recover part of the minority class in selected configurations, but with trade-offs. These findings are methodological and preliminary; they do not support clinical decision-making, diagnosis, triage, or deployment.

## Research objective

The objective is to assess how different supervised learning paradigms behave when predicting a binary utilization outcome from the demographic and health variables available in the NPHA dataset. The protocol emphasizes fair comparison, preservation of prevalence, leakage-resistant preprocessing, fold-specific feature selection, and class-sensitive evaluation rather than accuracy alone.

## Contributions

- Comparative evaluation of linear, probabilistic, instance-based, tree-based, neural, margin-based, and graph-based classifiers under one protocol.
- Nested stratified cross-validation with explicit inner-loop model selection and held-out outer-fold evaluation.
- Evaluation without SMOTE or other synthetic resampling, preserving the observed epidemiological class distribution.
- Fold-specific Mean Decrease Impurity (MDI) feature selection and cost-sensitive learning as separate scenarios.
- Internal ensemble-member, calibration, and threshold selection using training-side out-of-fold information.
- Exploratory paired Wilcoxon signed-rank comparisons with Holm correction.

## Dataset and target

The project uses the National Poll on Healthy Aging (NPHA) dataset.

- **Sample:** 714 adults aged 50 years and older.
- **Predictors:** 14 sociodemographic and health-related variables.
- **Target:** binary recoding of the original number-of-doctors-visited variable:
  - `0`: zero or one distinct doctor consulted — 131 observations;
  - `1`: two or more distinct doctors consulted — 583 observations.

Class `1` is the positive and majority class (81.65%); class `0` represents 18.35%. The number of distinct doctors consulted is treated as an operational utilization proxy, not as a validated clinical definition of high utilization.

## Evaluation protocol

All preprocessing is contained in a `ColumnTransformer` pipeline and fitted separately within each training partition:

- Numerical variables: training-partition median imputation followed by `StandardScaler`.
- Categorical variables: training-partition most-frequent imputation followed by one-hot encoding with unknown-category handling.
- Validation and outer-test data: transformed with fitted training components, without refitting.

The evaluation uses nested `StratifiedKFold` with 10 shuffled outer folds and 5 shuffled inner folds, using seed `42`. Hyperparameter combinations are generated with `ParameterGrid` and evaluated explicitly by inner-fold macro F1. Each selected configuration is then fitted on the outer training partition and evaluated once on its held-out outer test partition. Out-of-fold predictions aggregate one prediction for every observation.

In the MDI-80% scenario, an auxiliary 100-tree random forest ranks transformed features by Mean Decrease Impurity within each training fold. The smallest subset reaching at least 80% cumulative importance is retained, and selection is recomputed independently by fold. No global feature subset is defined.

No synthetic resampling is applied. Cost-sensitive weighting is evaluated for Decision Tree, Random Forest, Logistic Regression, SVC, and XGBoost using class information from each training partition. SVC scores are sigmoid-calibrated with `CalibratedClassifierCV` using internal five-fold calibration. Ensemble members, equal one-third weights, and decision thresholds are selected without access to the corresponding outer test fold. Candidate thresholds range from 0.10 to 0.90 in increments of 0.05; macro F1 is the primary criterion, with balanced accuracy and MCC used to resolve ties.

## Models and scenarios

The study evaluates 11 supervised classifiers and a majority-class dummy baseline:

1. K-Nearest Neighbors (KNN)
2. Decision Tree (DT)
3. Random Forest (RF)
4. Support Vector Classifier (SVC)
5. Multi-Layer Perceptron (MLP)
6. Logistic Regression (LR)
7. XGBoost (XGB)
8. Gaussian Naive Bayes (GNB)
9. Nearest Centroid (NC)
10. Optimum-Path Forest (OPF)
11. AdaBoost (AdaB)
12. Majority-class Dummy

Scenarios are:

- **BL:** all-feature baseline;
- **MDI:** fold-specific MDI-80% feature selection;
- **COST:** cost-sensitive weighting for DT, LR, RF, SVC, and XGB.

Ensemble analyses include majority voting, probability averaging, a fixed threshold of 0.50, and the internally selected threshold of 0.80.

## Final results

Values are mean ± standard deviation across the 10 outer folds. `Rec0` and `Rec1` denote recall for classes 0 and 1. Dashes indicate metrics not included in this concise summary.

| Configuration | Scenario | Rec0 | Rec1 | F1 macro | BA | MCC | AUROC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | MDI | — | — | **0.498 ± 0.044** | — | — | — | — |
| LR | COST | **0.596 ± 0.088** | — | — | **0.553 ± 0.062** | **0.082 ± 0.096** | **0.574 ± 0.075** | **0.868 ± 0.037** |
| Ensemble, selected threshold 0.80 | THR | **0.129 ± 0.108** | **0.879 ± 0.114** | **0.493 ± 0.058** | — | — | — | — |

Across scenarios, several classifiers largely reproduce the majority-class prediction profile. AUROC values remain near 0.5 for most models, while PR-AUC remains close to the positive-class prevalence, supporting the conclusion of limited discriminative capacity.

OPF does not provide a continuous score in the implemented pipeline. Because AUROC and PR-AUC require a continuous class-1 score rather than discrete predictions, both metrics are unavailable for OPF. Nearest Centroid uses its distance-based decision function for discrimination metrics.

## Statistical evidence

The exploratory statistical analysis compares each classifier in the MDI-80% scenario with the majority-class dummy using paired outer-fold macro F1 values. Eleven contrasts are corrected with Holm’s step-down procedure. No contrast has corrected `p < 0.05`. Since cross-validation folds from one dataset are dependent and only 10 outer folds are available, these results are descriptive rather than conclusive evidence of model superiority.

## Reproducibility

The public final run is identified by seed `42` and stored in `revisao_experimental/20260914T204617Z/`. The directory includes the manifest, fold assignments, per-fold metrics, out-of-fold predictions, MDI outputs, ensemble outputs, threshold results, statistical results, and ROC/PR artifacts. Key files include:

- `manifest.json`
- `folds.csv`
- `metrics_per_fold.csv`
- `mdi_frequency.csv`
- `oof_*.csv`
- `ensemble_members.json`
- `ensemble_profiles.json`
- `thresholds.json`
- `wilcoxon_holm.json`

From the project root, launch the analysis notebook with:

```bash
jupyter notebook src/AUSMI.ipynb
```

The input data are available at `src/NPHA-doctor-visits.csv`, and project dependencies are documented in `requirements.txt`.

## Public project structure

```text
analyze-medical-services/
├── README.md
├── requirements.txt
├── src/
│   ├── AUSMI.ipynb
│   ├── NPHA-doctor-visits.csv
│   └── opfython.log*
├── revisao_experimental/
│   └── 20260914T204617Z/
└── artifacts/
    └── AUSMI_corrigido/20260912_214440/
```

The final results in this document refer to `revisao_experimental/20260914T204617Z/`. The older `artifacts/AUSMI_corrigido/20260912_214440/` directory remains available for repository traceability but is not used as evidence for the final results.

## Limitations

This is a cross-sectional analysis of one dataset with an imbalanced target. The recorded variables and operational outcome form a difficult prediction task, and model behavior and feature importance vary across folds. External validation was not available for a compatible dataset, so performance cannot be assumed to transfer to other populations, settings, or data-collection procedures.

## Citation

Final bibliographic information is pending. Use the following placeholder until publication details are confirmed:

> Silva, P. K. S., et al. *Analysis and Prediction of Medical Service Utilization Among Adults Aged 50 and Older: A Machine Learning Approach*. [Publication details to be added].
