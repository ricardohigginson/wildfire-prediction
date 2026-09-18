# Wildfire Ignition Risk Prediction

## Overview
This project develops an end-to-end machine learning pipeline to estimate the probability of wildfire ignition across the continental United States based on recent environmental and weather conditions. 

The project follows a structured machine learning workflow:
* **Exploratory Data Analysis (EDA)** & geographic visualization
* **Data Preparation & Cleaning** (sentinel missing value handling, unit conversions)
* **Feature Engineering & Redundancy Reduction** (historical time windows & collinearity filtering)
* **Spatial Grouped Train/Test Splitting** (preventing spatial data leakage)
* **Random Forest Baseline Modeling**
* **Permutation Importance & Feature Selection** (reducing 40 features to 14)
* **Methodological Validation** (Out-of-Fold permutation importance check)
* **Multi-Stage XGBoost Hyperparameter Optimization** (Randomized Search, Grid Search, and Manual Fine-Tuning)
* **Classification Threshold Analysis** (Balanced vs. High-Sensitivity operational modes)
* **Model Explainability with SHAP** (quantifying feature contributions)
* **Artifact Export & Interactive Demo Suite** (Terminal CLI and Jupyter `ipywidgets` GUI)

---

## Dataset
The project utilizes the **US Wildfire Dataset (2014–2025)** created by FireCastRL and hosted on Kaggle:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)

### Key Characteristics:
* **126,800 labeled samples**: 50,720 positive wildfire ignition events (IRWIN) and 76,080 negative samples (synthesized using far, near, and yearly offsets).
* **Coverage**: Continental United States (2014–2025).
* **15 raw environmental variables** sourced from GRIDMET (precipitation, humidity, solar radiation, temperature, wind speed, vapor pressure deficit, fuel moisture, ERC, BI, evapotranspiration).
* **75-day temporal sequences**: Each observation consists of 60 historical days before the reference day, the reference day itself (Today), and 14 days post-reference day (>9.5 million sequence rows in raw data).

---

## Data Preparation & Feature Engineering

### 1. Data Cleaning
* **Sentinel Missing Value Handling**: Raw invalid missing data flags (value `32767`) in GRIDMET weather metrics were identified and replaced with `NaN` before further processing.
* **Temperature Unit Conversion**: Daily minimum (`Min_Daily_Temperature`) and maximum (`Max_Daily_Temperature`) readings were converted from Kelvin to Celsius ($^\circ\text{C} = \text{K} - 273.15$).

### 2. Feature Aggregations
To predict ignition risk without data leakage from future days, features were constructed using only observations up to the reference day (**Today**):
* **Historical Windows**: Aggregations over **7, 14, 30, and 60 days** using average, median, minimum, and maximum statistics.
* **Reference Day Features**: Same-day environmental conditions retained to capture immediate fire-conducive weather.

### 3. Collinearity Filtering & Initial Feature Set
An initial temporal redundancy analysis revealed near-perfect collinearity ($r \approx 0.99$) between `Actual_Evapotranspiration` and `Potential_Evapotranspiration` across all time horizons. `Actual_Evapotranspiration` was explicitly removed to eliminate severe feature redundancy, establishing an initial dataset of **40 predictive features**.

---

## Spatial Grouped Train/Test Split
Wildfire observations are geographically correlated; standard random splitting causes severe spatial data leakage between nearby training and test coordinates.

* **Grouping Variable**: Unique spatial locations constructed from `Latitude` and `Longitude` coordinates (coordinates were used strictly for grouping and excluded as model features).
* **Splitting Method**: `StratifiedGroupKFold` (5 folds, 80/20 train/test split).
* **Final Split**:
  * **Training Set**: 47,645 samples
  * **Test Set**: 11,911 samples
  * **Spatial Overlap**: Exactly **0 overlapping locations** between training and test sets.

---

## Models & Methodology

### 1. Random Forest Baseline
A `RandomForestClassifier` was trained on all 40 features to establish a performance baseline:
* **Hyperparameters**: 300 estimators, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features="sqrt"`, `class_weight="balanced"`.
* **Baseline Performance**: 66.86% Test Accuracy, 0.77 Wildfire Recall, 0.71 Wildfire F1.

### 2. Feature Selection (Permutation Importance)
To simplify the model while preserving predictive performance:
1. **Permutation Importance** was computed on the baseline Random Forest model to rank feature contributions.
2. **Feature Family Strategy**: Features were grouped into 10 environmental families (Precipitation, Specific Humidity, Min/Max Relative Humidity, Solar Radiation, Min/Max Temperature, Wind Speed, VPD, Potential Evapotranspiration). The single most informative historical window was selected for each family.
3. **Current-Day Context**: The 4 most impactful reference-day features (`Solar_Radiation_Today`, `Min_Relative_Humidity_Today`, `Vapor_Pressure_Deficit_Today`, `Wind_Speed_Today`) were added.
4. **Final Reduced Set**: **14 features** (a 65% reduction from the original 40 features).

#### Selected 14 Features:
1. `Precipitation_Average_30_Days`
2. `Specific_Humidity_Average_30_Days`
3. `Min_Relative_Humidity_Min_30_Days`
4. `Max_Relative_Humidity_Max_7_Days`
5. `Solar_Radiation_Average_7_Days`
6. `Min_Daily_Temperature_Min_14_Days`
7. `Max_Daily_Temperature_Max_30_Days`
8. `Wind_Speed_Average_30_Days`
9. `Vapor_Pressure_Deficit_Average_30_Days`
10. `Potential_Evapotranspiration_Average_30_Days`
11. `Solar_Radiation_Today`
12. `Min_Relative_Humidity_Today`
13. `Vapor_Pressure_Deficit_Today`
14. `Wind_Speed_Today`

#### Methodological Validation: Out-of-Fold (OOF) Check
* **Context**: Permutation Importance was initially evaluated on the test set. To address this methodological limitation and avoid indirect use of the test set during feature selection, an Out-of-Fold (OOF) permutation importance check was performed strictly within the training set.
* **Findings**:
  * **Virtually Identical Performance**: Retraining the model with the OOF-selected features yielded virtually unchanged results:
    * **Test Accuracy**: Changed marginally from 69.40% to **69.31%**.
    * **Wildfire Recall**: Remained constant at **79%**.
    * **Train-Test Gap**: Remained virtually identical (**13.42 pp** vs. **13.47 pp**).
  * **Minor Feature Window Shifts**: The OOF procedure modified only two temporal windows out of the 14 selected features:
    * `Solar Radiation`: 7-day median $\rightarrow$ 30-day median
    * `Minimum Daily Temperature`: 14-day minimum $\rightarrow$ 30-day minimum
* **Conclusion**: This additional check indicates that the original test-set-based permutation importance had a negligible impact on final model performance. The original 14-feature model and results are therefore retained as the main project benchmark, while the OOF analysis is documented as a methodological validation.

---

### 3. Multi-Stage XGBoost Optimization
XGBoost models were developed and systematically optimized through a three-stage tuning workflow using spatial cross-validation on the training set:

1. **Randomized Search CV**: Broad initial exploration across hyperparameter distributions.
2. **Grid Search CV**: Focused search around promising hyperparameter regions.
3. **Manual Fine-Tuning**: Systematic step-by-step experimentation isolating key hyperparameter interactions:
   * **Tree Complexity**: Tuning `max_depth` and `min_child_weight`.
   * **Regularization**: Evaluating L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties.
   * **Learning Rate & Estimators**: Balancing `learning_rate` and `n_estimators`.
   * **Subsampling**: Testing row (`subsample`) and column (`colsample_bytree`) ratios.
   * **Depth Re-evaluation**: Re-evaluating tree depth to control a ~27% train-CV overfitting gap, selecting `max_depth=8` / `n_estimators=500` for a balanced generalizability trade-off.

#### Final Model Comparison (Untouched Test Set):
| Metric | Baseline RF (40 Features) | Tuned XGBoost (40 Features Benchmark) | Final Tuned XGBoost (14 Features) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 66.86% | **71.31%** | 69.40% |
| **Wildfire Precision** | 0.67 | **0.70** | 0.69 |
| **Wildfire Recall** | 0.77 | **0.81** | 0.79 |
| **Wildfire F1-Score** | 0.71 | **0.75** | 0.74 |
| **Macro F1-Score** | 0.66 | **0.71** | 0.69 |
| **Train-Test Gap** | ~5.8% | 13.55% | **13.47%** |

*Note: The 14-feature XGBoost model was selected as the final operational model due to its simplicity (65% fewer inputs) with minimal loss in wildfire detection performance.*

---

## Classification Threshold Analysis
Out-of-fold predicted probabilities on the training set were analyzed across decision thresholds from 0.30 to 0.60. Two operational deployment modes were selected and evaluated on the untouched test set:

| Metric | Balanced Mode (Threshold = 0.50) | High-Sensitivity Mode (Threshold = 0.35) |
| :--- | :---: | :---: |
| **Test Accuracy** | **69.40%** | 66.80% |
| **Wildfire Precision** | **0.69** | 0.63 |
| **Wildfire Recall** | 78.82% | **92.21%** |
| **Wildfire F1-Score** | 0.74 | **0.75** |
| **No-Wildfire Recall** | **58.37%** | 37.07% |
| **Missed Wildfires (FN)** | 1,360 | **500 (63% reduction)** |

---

## Model Explainability (SHAP Analysis)
TreeSHAP (`TreeExplainer`) was used to quantify feature contributions for the final 14-feature XGBoost model on the test set.

*Note on Feature Labeling: During feature engineering, several temporal features were named `Average` in code despite using median aggregation (except `Precipitation`, which correctly uses mean). Display labels were corrected for SHAP reporting.*

### Top Feature Importance (% Relative Mean Absolute SHAP):
1. **`Precipitation_Average_30_Days`**: **17.25%** (strongest predictor; higher rainfall reduces risk)
2. **`Potential_Evapotranspiration_Median_30_Days`**: **11.96%** (atmospheric water demand)
3. **`Max_Daily_Temperature_Max_30_Days`**: **8.89%** (30-day peak heat)
4. **`Min_Relative_Humidity_Today`**: **7.42%** (immediate air dryness)
5. **`Min_Relative_Humidity_Min_30_Days`**: **6.98%** (extended background dryness)
6. **`Solar_Radiation_Median_7_Days`**: **6.39%**
7. **`Min_Daily_Temperature_Min_14_Days`**: **6.18%**
8. **`Vapor_Pressure_Deficit_Today`**: **6.03%**

---

## Model Deployment & Interactive Demo Suite

### Serialized Artifacts
* `xgb_reduced_final.pkl`: Serialized XGBoost 14-feature model.
* `feature_metadata.json`: Feature ordering, display metadata, and operational threshold definitions (`0.50` and `0.35`).

### Unified Demo Application (`wildfire_prediction_demo.py`)
The pipeline provides a standalone demonstration module supporting two interaction modes:

1. **Jupyter Notebook GUI (`ipywidgets`)**:
   * Visual input cards for all 14 environmental variables with descriptions, units, and valid ranges.
   * Preset scenario autofill buttons (**"Low Wildfire Risk Example"** vs. **"High Wildfire Risk Example"**).
   * Mode toggle switch between **Balanced Mode (0.50)** and **High-Sensitivity Mode (0.35)**.
   * Real-time wildfire probability gauges and formatted **Top 5 SHAP Contributor** tables showing exact directional impact ($\uparrow$ increases risk, $\downarrow$ decreases risk).
2. **Interactive Terminal Application**:
   * CLI interface prompting for environmental metrics with input validation.
   * On-the-fly threshold switching and printed SHAP log-odds explanations.

---

## Limitations
* **Geographic Scope**: Trained exclusively on continental US data; not validated for international ecosystems.
* **Negative Sample Synthesis**: Non-wildfire cases rely on synthesized spatial/temporal offsets from the dataset authors.
* **Unobserved Factors**: Local vegetation fuel load, ignition sources (human activity/lightning strikes), and real-time firefighting interventions are not captured in GRIDMET weather metrics.
* **Methodological Limitation**: Permutation importance was initially evaluated on the test set, introducing indirect test-set influence during feature selection. A subsequent OOF validation within the training set produced virtually identical model performance (69.31% vs. 69.40% test accuracy), suggesting that this limitation had negligible impact on the reported results.
* **Experimental Nature**: Designed as an experimental machine learning risk estimator rather than an operational early-warning emergency system.

---

## Technologies Used
* **Language & Environment**: Python 3.12, Jupyter Notebook
* **Data Processing & Analytics**: `pandas`, `NumPy`, `GeoPandas`
* **Machine Learning**: `scikit-learn`, `XGBoost`
* **Explainability**: `SHAP`
* **UI & Visualization**: `ipywidgets`, `Matplotlib`, `Seaborn`

---

## References & Credits
* **Dataset**: FireCastRL, *US Wildfire Dataset (2014–2025)*, Kaggle.
* **Associated Research**: Mathur, S., Manjunath, S. B., Kulkarni, N., & Vereshchaka, A. (2025). *Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression.* ICMLA 2025.
