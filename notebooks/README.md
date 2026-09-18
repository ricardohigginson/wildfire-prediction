# Wildfire Ignition Risk Prediction

## Overview

This project develops a machine learning model to estimate the probability of wildfire ignition based on recent environmental and weather conditions.

The project follows an end-to-end machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data preparation and feature engineering
- Spatially grouped train/test splitting
- Random Forest modelling
- XGBoost modelling
- Feature selection using Permutation Importance
- Classification threshold analysis
- Model explainability with SHAP
- Final model export for prediction

The main objective is to investigate whether recent environmental conditions can provide useful signals for predicting wildfire ignition risk.

## Dataset

The project uses the **US Wildfire Dataset (2014–2025)** created by FireCastRL and available on Kaggle.

https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset

According to the dataset documentation:

- 126,800 labelled samples
- 50,720 positive wildfire ignition events
- 76,080 negative samples
- Coverage of the continental United States
- Period: 2014–2025
- 15 environmental variables
- Environmental variables from GRIDMET
- Wildfire ignition labels from IRWIN

Each sample is represented as a 75-day temporal window:

- 60 days before the reference day
- The reference day
- 14 days after the reference day

This results in more than 9.5 million sequence rows in the original dataset.

The original dataset contains real wildfire ignition events and synthesized negative samples generated using far, near, and yearly offsets.

### Environmental Variables

The dataset contains environmental variables including:

- Precipitation
- Relative humidity
- Specific humidity
- Solar radiation
- Temperature
- Wind speed
- Vapor pressure deficit (VPD)
- Fuel moisture
- Energy Release Component (ERC)
- Burning Index (BI)
- Evapotranspiration

For this project, the predictive features are constructed using information available up to the reference day, avoiding the use of future observations as predictors.

## Methodology

### Exploratory Data Analysis

The EDA examined:

- Dataset structure
- Missing values
- Variable distributions
- Class balance
- Temporal characteristics
- Geographic distribution
- Environmental differences between wildfire and non-wildfire observations

### Feature Engineering

Daily environmental variables were transformed into aggregated features over historical windows including:

- 7 days
- 14 days
- 30 days
- 60 days

Aggregation statistics included:

- Average
- Median
- Minimum
- Maximum

Features representing conditions on the reference day (`Today`) were also retained.

### Spatial Train/Test Split

Because wildfire observations can be geographically correlated, a spatially grouped train/test split was used.

Locations were grouped using latitude and longitude, and `StratifiedGroupKFold` was used to create the split.

Latitude and longitude were used only for grouping and were not used as model features.

The final split produced:

- Training set: 47,645 samples
- Test set: 11,911 samples
- 40 initial predictive features
- No spatial overlap between training and test locations

## Models

### Random Forest

A Random Forest classifier was used as the initial baseline model.

The model used:

- 300 trees
- Maximum depth: 10
- Minimum samples per split: 10
- Minimum samples per leaf: 4
- Square-root feature selection
- Balanced class weights

Random Forest was also used as the reference model for the **Permutation Importance** analysis used during feature selection.

### XGBoost

XGBoost was subsequently used to develop the final models.

The final reduced model used:

- 500 estimators
- Learning rate: 0.05
- Maximum depth: 8
- Minimum child weight: 2
- Subsample: 0.8
- Column subsampling: 0.9
- L1 regularisation (`reg_alpha`): 2
- L2 regularisation (`reg_lambda`): 15

## Model Performance

### Random Forest — 40 Features

| Metric | Score |
|---|---:|
| Accuracy | 66.86% |
| Wildfire Precision | 0.67 |
| Wildfire Recall | 0.77 |
| Wildfire F1 | 0.71 |
| Macro F1 | 0.66 |

### XGBoost — 40 Features

| Metric | Score |
|---|---:|
| Accuracy | 71.31% |
| Wildfire Precision | 0.70 |
| Wildfire Recall | 0.81 |
| Wildfire F1 | 0.75 |
| Macro F1 | 0.71 |

### XGBoost — 14 Features

After feature selection, the model was retrained using 14 selected features.

| Metric | Score |
|---|---:|
| Accuracy | 69.40% |
| Wildfire Precision | 0.69 |
| Wildfire Recall | 0.79 |
| Wildfire F1 | 0.74 |
| Macro F1 | 0.69 |

The 40-feature model is retained as a benchmark, while the 14-feature model is used for the final prediction workflow.

## Feature Selection

Feature selection was performed using **Permutation Importance** with a Random Forest model.

Permutation Importance measures the importance of each feature by randomly shuffling its values and measuring how much the model's performance decreases. If shuffling a feature causes a large performance drop, the feature is considered more important to the model's predictions.

The 40 initial features were ranked according to their permutation importance. Since several features were derived from the same underlying environmental variable using different time windows and aggregation methods, the selection process also considered **feature families**.

### Selection Strategy

The selection was performed in two stages:

1. **One representative feature from each environmental family**

   The Permutation Importance ranking was used to identify the most informative representation within each feature family. This prevented families containing many derived variables from dominating the final feature set and ensured that different types of environmental information were represented.

2. **Four additional `Today` features**

   Four features representing the environmental conditions on the reference day were then added. These features complement the historical aggregated variables by providing information about the conditions on the prediction day itself.

This resulted in a final set of **14 features**, combining historical environmental patterns with current-day conditions.

The reduced feature set was then used to retrain the XGBoost model and evaluate the impact of feature selection on predictive performance.

The final 14 selected features were:

1. Precipitation — 30-day average
2. Specific Humidity — 30-day median
3. Minimum Relative Humidity — 30-day minimum
4. Maximum Relative Humidity — 7-day maximum
5. Solar Radiation — 7-day median
6. Minimum Daily Temperature — 14-day minimum
7. Maximum Daily Temperature — 30-day maximum
8. Wind Speed — 30-day median
9. Vapor Pressure Deficit — 30-day median
10. Potential Evapotranspiration — 30-day median
11. Solar Radiation — Today
12. Minimum Relative Humidity — Today
13. Vapor Pressure Deficit — Today
14. Wind Speed — Today

The resulting feature set provides a more compact representation of wildfire-related environmental conditions while maintaining coverage across the main environmental families and preserving information about conditions on the prediction day.

## Classification Threshold Analysis

The XGBoost model produces a probability between 0 and 1. Two classification thresholds were investigated:

- **0.50** — balanced threshold
- **0.35** — high-sensitivity threshold

The 0.35 threshold was selected using out-of-fold predictions from the training data. It was then evaluated on the untouched test set.

### Threshold = 0.50

| Metric | Score |
|---|---:|
| Accuracy | 69.40% |
| Wildfire Precision | 0.688989 |
| Wildfire Recall | 0.788228 |
| Wildfire F1 | 0.735275 |
| No-Wildfire Recall | 0.583713 |

Confusion matrix:

| | Predicted No Wildfire | Predicted Wildfire |
|---|---:|---:|
| Actual No Wildfire | 3,204 | 2,285 |
| Actual Wildfire | 1,360 | 5,062 |

### Threshold = 0.35

| Metric | Score |
|---|---:|
| Accuracy | 66.8038% |
| Wildfire Precision | 0.631613 |
| Wildfire Recall | 0.922143 |
| Wildfire F1 | 0.749715 |
| No-Wildfire Recall | 0.370741 |

Confusion matrix:

| | Predicted No Wildfire | Predicted Wildfire |
|---|---:|---:|
| Actual No Wildfire | 2,035 | 3,454 |
| Actual Wildfire | 500 | 5,922 |

The lower threshold increases wildfire recall while also increasing the number of false positives.

## Model Explainability

SHAP was used to investigate which features contributed most strongly to the XGBoost model predictions.

The top features based on mean absolute SHAP values were:

| Rank | Feature | Relative SHAP Importance |
|---|---|---:|
| 1 | Precipitation — 30-day average | 7.838% |
| 2 | Potential Evapotranspiration — 30-day median | 4.424% |
| 3 | Solar Radiation — 7-day median | 3.085% |
| 4 | Maximum Daily Temperature — 30-day maximum | 2.838% |
| 5 | Minimum Relative Humidity — 30-day minimum | 2.698% |
| 6 | Minimum Daily Temperature — 14-day minimum | 2.187% |
| 7 | Specific Humidity — 30-day median | 1.927% |
| 8 | VPD — 30-day median | 1.787% |
| 9 | Solar Radiation — Today | 1.540% |
| 10 | Minimum Relative Humidity — Today | 1.476% |
| 11 | VPD — Today | 1.476% |
| 12 | Maximum Relative Humidity — 7-day maximum | 1.338% |
| 13 | Wind Speed — 30-day median | 1.288% |
| 14 | Wind Speed — Today | 0.746% |

These values represent relative model importance based on mean absolute SHAP values. They should not be interpreted as causal effects or as direct percentage-point contributions to the predicted probability.

## Final Model

The final prediction workflow uses the 14-feature XGBoost model.

The exported model artifacts are:

- `xgb_reduced_final.pkl`
- `feature_metadata.json`
- `wildfire_prediction_demo.py`

The metadata stores:

- Required feature names
- Balanced classification threshold (`0.50`)
- High-sensitivity threshold (`0.35`)

The prediction script loads the trained model and metadata and generates a wildfire probability from the required environmental inputs.

## Repository Structure

The current repository keeps the CSV files inside `notebooks/` because the existing notebooks use those paths.

```text
wildfire-prediction/
│
├── notebooks/
│   ├── eda_ml_project.ipynb
│   ├── ml_training.ipynb
│   ├── wildfire_prediction_demo.ipynb
│   ├── wildfire_prediction_demo.py
│   ├── feature_metadata.json
│   ├── xgb_reduced_final.pkl
│   ├── header.jpg
│   ├── Wildfire_Dataset.csv
│   ├── Wildfire_Dataset_cleaned.csv
│   └── Wildfire_Dataset_model_ready.csv
│
├── .gitignore
└── README.md
```

The large CSV files are kept locally and excluded from GitHub using `.gitignore`.

## Reproducibility

The project was developed using Python and Jupyter Notebook.

The notebooks contain the exploratory analysis, feature engineering, model training and evaluation process.

The final trained model is provided as a serialized XGBoost model together with its feature metadata.

Because the dataset files are large, they are not included in the Git repository. They must be available in the `notebooks/` directory when running the notebooks that reference them.

The dataset can be downloaded from Kaggle:

https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset

## Limitations

- The dataset covers the continental United States and the model has not been validated for other geographic regions.
- The negative samples in the original dataset are synthesized using the methodology described by the dataset authors.
- Wildfire ignition is influenced by factors that are not represented by the environmental variables used in this project.
- The model identifies statistical patterns in the available data and does not establish causal relationships.
- Model performance depends on the characteristics and distribution of the underlying dataset.
- A lower classification threshold increases wildfire recall but also produces more false positives.
- The model should be interpreted as an experimental wildfire-risk prediction model rather than an operational wildfire warning system.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Jupyter Notebook
- Git
- GitHub

## References

### Dataset

FireCastRL. *US Wildfire Dataset (2014–2025).* Kaggle.

https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset

### Associated Research

Mathur, S., Manjunath, S. B., Kulkarni, N., & Vereshchaka, A. (2025).

*Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression.*

2025 International Conference on Machine Learning and Applications (ICMLA).
