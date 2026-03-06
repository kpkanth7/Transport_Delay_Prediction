# Public Transport Delay Prediction Using Weather and City Events

## Objective

The objective of this project is to explore whether public transport delays can be predicted using external and operational factors such as weather conditions, traffic congestion, city events, and scheduled trip information.

The project investigates two machine learning formulations:

1. **Regression** — predict the exact arrival delay in minutes  
2. **Classification** — predict whether a trip will be delayed or not  

After evaluating both tasks, the project is framed primarily as a **classification problem**, with **regression included as a secondary analysis**.

---

## Project Overview

Public transport systems are influenced by many real-world conditions such as rain, snow, rush-hour traffic, large city events, and route-level operational context. This project uses a structured tabular dataset to examine how these factors relate to delays.

The dataset contains trip-level records with schedule information, weather attributes, traffic indicators, and event-related features. Using this data, the project builds an end-to-end machine learning workflow that includes:

- data loading and inspection
- missing-value handling
- leakage-aware feature selection
- feature engineering
- exploratory data analysis
- regression and classification modeling
- baseline comparison
- model evaluation
- saved model artifacts for inference

---

## Dataset

**File used:** `public_transport_delays.csv`
- Rows: **2,000**
- Columns: **24**
- Missing values: mainly in `event_type` (missing usually means no event)

### Target variables
- `actual_arrival_delay_min` → used for regression
- `delayed` → used for classification

### Key feature groups
- route and stop information
- scheduled departure and arrival times
- date-based features
- weather condition and precipitation
- wind speed, humidity, and temperature
- traffic congestion level
- event type and event size indicators

---

## Project Flow

### 1. Data Inspection and Cleaning
The dataset was first inspected to understand:
- column structure
- data types
- missing values
- target distributions

Special attention was given to `event_type`, where missing values were treated as likely indicating **no event**.

### 2. Leakage-Aware Feature Selection
Columns that directly contained outcome information were excluded from model inputs.

Examples:
- `actual_departure_delay_min`
- `actual_arrival_delay_min`
- `delayed`

These columns were removed appropriately depending on the prediction task to avoid data leakage and keep the modeling realistic.

### 3. Feature Engineering
Additional features were created from raw operational fields to improve model signal.

Engineered features included:
- `month`, `day`, `day_of_year`, `is_weekend`
- `scheduled_departure_hour`, `scheduled_arrival_hour`
- `scheduled_trip_duration_min`
- `is_morning_peak`, `is_evening_peak`, `is_commute_peak`
- `has_event`, `event_size_bucket`
- `is_rain`, `is_snow`, `is_storm`, `is_fog`, `is_bad_weather`
- `is_heavy_precip`, `is_high_wind`, `is_high_humidity`
- `is_extreme_temp`, `is_high_congestion`

### 4. Exploratory Data Analysis
EDA was performed to study:
- distribution of arrival delay
- delay distribution across weather conditions
- class balance for the `delayed` target
- average arrival delay by weather and event context

### 5. Modeling
Two machine learning tracks were explored.

#### Regression models
- Dummy Regressor
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

#### Classification models
- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### 6. Evaluation
Regression was evaluated using:
- MAE
- RMSE
- R²

Classification was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

All models were compared against dummy baselines to ensure honest interpretation.

### 7. Inference
The final pipeline saves trained models using `joblib` and supports inference through `predict.py`.

Saved artifacts:
- `best_delay_regressor.joblib`
- `best_delay_classifier.joblib`

Saved metric files:
- `regression_metrics.json`
- `classification_metrics.json`

---

## Why These Models Were Chosen

This project compares both simple baselines and stronger nonlinear models.

### Regression
- **Dummy Regressor** establishes a mean-baseline benchmark
- **Linear Regression** provides a simple interpretable starting point
- **Random Forest / Gradient Boosting / XGBoost** are suitable for structured tabular data and can capture nonlinear relationships between weather, schedule, congestion, and events

### Classification
- **Dummy Classifier** establishes a majority-class baseline
- **Logistic Regression** provides an interpretable benchmark
- **Random Forest / Gradient Boosting / XGBoost** are strong choices for learning nonlinear feature interactions in tabular data

---

## Key Results

### Classification Metrics
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|------:|------:|------:|------:|------:|
| DummyClassifier | 0.7675 | 0.7675 | 1.0000 | 0.8685 | 0.5000 |
| RandomForestClassifier | 0.7675 | 0.7675 | 1.0000 | 0.8685 | 0.4827 |
| LogisticRegression | 0.7650 | 0.7669 | 0.9967 | 0.8669 | 0.5069 |
| GradientBoostingClassifier | 0.7575 | 0.7665 | 0.9837 | 0.8616 | 0.4841 |
| XGBClassifier | 0.7400 | 0.7609 | 0.9642 | 0.8506 | 0.4551 |

### Regression Metrics
| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| GradientBoostingRegressor | 8.1355 | 9.5059 | -0.0141 |
| DummyRegressor | 8.1546 | 9.4589 | -0.0041 |
| XGBRegressor | 8.1693 | 9.6697 | -0.0494 |
| RandomForestRegressor | 8.1995 | 9.6070 | -0.0358 |
| LinearRegression | 8.9846 | 10.8546 | -0.3223 |

---

## Achievements

- Built a complete machine learning workflow for a real-world transport-delay problem
- Performed leakage-aware feature selection to keep modeling realistic
- Engineered time-based, weather-based, event-based, and traffic-based predictors
- Evaluated both regression and classification formulations of the problem
- Compared trained models against dummy baselines for honest benchmarking
- Saved model artifacts and created an inference pipeline using `predict.py`
- Documented both the strengths and limitations of the dataset clearly

---

## Inference Example

The saved models were successfully loaded and used for prediction through `predict.py`.

Example outputs:
- **Classification:** predicted `delayed = 1`
- **Class probabilities:** `[0.0, 1.0]`
- **Regression:** predicted arrival delay = `13.13` minutes

This confirms that the pipeline works end-to-end from training to inference.

---

## Project Difficulties and Limitations

This project produced a working machine learning pipeline, but the predictive results also revealed important limitations.

### Regression difficulty
Predicting the exact delay in minutes turned out to be difficult. The regression models stayed close to the dummy baseline, and the negative R² values suggested limited predictive structure for the minute-level target.

### Classification difficulty
The classification models also performed close to the dummy majority baseline. This likely reflects:
- class imbalance
- limited feature signal
- unobserved operational factors not included in the dataset

### Why this is still a strong project
Even when performance is limited, this project still demonstrates:
- structured data cleaning
- leakage-aware modeling
- feature engineering
- baseline comparison
- honest model interpretation
- saved artifacts and inference
- end-to-end reproducibility

---

## Final Conclusion

This project explored two machine learning formulations for public transport delay prediction using schedule, route, weather, traffic, and city-event features.

- The **regression analysis** attempted to predict exact arrival delay in minutes, but the models remained close to a dummy mean baseline.
- The **classification analysis** attempted to predict whether a trip would be delayed, but those models also performed close to a majority-class baseline.

These results suggest that the dataset contains limited predictive signal for both precise delay-duration estimation and delayed/not-delayed classification, at least with the current features and sample size.

Even though predictive performance was limited, the project successfully demonstrates a complete machine learning workflow:
- data inspection and cleaning
- leakage-aware preprocessing
- feature engineering
- exploratory analysis
- regression and classification modeling
- baseline comparison
- model evaluation
- artifact saving with `joblib`
- end-to-end inference with a prediction script

