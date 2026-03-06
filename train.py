import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path("public_transport_delays.csv")
CLASSIFICATION_MODEL_PATH = Path("best_delay_classifier.joblib")
REGRESSION_MODEL_PATH = Path("best_delay_regressor.joblib")
CLASSIFICATION_METRICS_PATH = Path("classification_metrics.json")
REGRESSION_METRICS_PATH = Path("regression_metrics.json")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path.resolve()}")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Basic cleaning(Sanity Check)
    data["event_type"] = data["event_type"].fillna("NoEvent")
    data["event_attendance_est"] = data["event_attendance_est"].fillna(0)

    # Datetime fields
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["time_dt"] = pd.to_datetime(data["time"], format="%H:%M:%S", errors="coerce")
    data["scheduled_departure_dt"] = pd.to_datetime(
        data["scheduled_departure"], format="%H:%M:%S", errors="coerce"
    )
    data["scheduled_arrival_dt"] = pd.to_datetime(
        data["scheduled_arrival"], format="%H:%M:%S", errors="coerce"
    )

    # Calendar features
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_year"] = data["date"].dt.dayofyear
    data["is_weekend"] = data["weekday"].isin([5, 6]).astype(int)

    # Time features from scheduled values
    data["scheduled_departure_hour"] = data["scheduled_departure_dt"].dt.hour
    data["scheduled_departure_minute"] = data["scheduled_departure_dt"].dt.minute
    data["scheduled_arrival_hour"] = data["scheduled_arrival_dt"].dt.hour
    data["scheduled_arrival_minute"] = data["scheduled_arrival_dt"].dt.minute

    data["scheduled_departure_minute_of_day"] = (
        data["scheduled_departure_hour"] * 60 + data["scheduled_departure_minute"]
    )
    data["scheduled_arrival_minute_of_day"] = (
        data["scheduled_arrival_hour"] * 60 + data["scheduled_arrival_minute"]
    )

    trip_duration = (
        data["scheduled_arrival_minute_of_day"] - data["scheduled_departure_minute_of_day"]
    ) % (24 * 60)
    data["scheduled_trip_duration_min"] = trip_duration

    # Commute peak indicators differentiating morning peak and evening peak 
    data["is_morning_peak"] = data["scheduled_departure_hour"].between(7, 10).astype(int)
    data["is_evening_peak"] = data["scheduled_departure_hour"].between(16, 19).astype(int)
    data["is_commute_peak"] = (
        (data["is_morning_peak"] == 1) | (data["is_evening_peak"] == 1)
    ).astype(int)

    
    data["has_event"] = (data["event_type"] != "NoEvent").astype(int)

    def bucket_event_size(x):
        if pd.isna(x) or x == 0:
            return "NoEvent"
        if x < 1000:
            return "Small"
        if x < 10000:
            return "Medium"
        return "Large"

    data["event_size_bucket"] = data["event_attendance_est"].apply(bucket_event_size)

    # Weather indicators grouping together
    weather = data["weather_condition"].astype(str).str.lower()
    data["is_rain"] = weather.str.contains("rain").astype(int)
    data["is_snow"] = weather.str.contains("snow").astype(int)
    data["is_storm"] = weather.str.contains("storm").astype(int)
    data["is_fog"] = weather.str.contains("fog").astype(int)
    data["is_bad_weather"] = (
        (data["is_rain"] == 1)
        | (data["is_snow"] == 1)
        | (data["is_storm"] == 1)
        | (data["is_fog"] == 1)
    ).astype(int)

    # Numeric threshold features
    data["is_heavy_precip"] = (data["precipitation_mm"] >= 10).astype(int)
    data["is_high_wind"] = (data["wind_speed_kmh"] >= 30).astype(int)
    data["is_high_humidity"] = (data["humidity_percent"] >= 80).astype(int)
    data["is_extreme_temp"] = (
        (data["temperature_C"] <= 0) | (data["temperature_C"] >= 35)
    ).astype(int)
    data["is_high_congestion"] = (data["traffic_congestion_index"] >= 70).astype(int)

    
    data = data.drop(
        columns=[
            "date",
            "time_dt",
            "scheduled_departure_dt",
            "scheduled_arrival_dt",
        ],
        errors="ignore",
    )
    return data


def time_ordered_split(df: pd.DataFrame, test_size: float = 0.2):
    if "date" not in df.columns:
        raise ValueError("The original dataframe must contain a 'date' column for time split.")
    ordered = df.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.sort_values(["date", "time"]).reset_index(drop=True)

    split_idx = int(len(ordered) * (1 - test_size))
    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()
    return train_df, test_df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(rmse, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_classification(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["roc_auc"] = None
    return metrics


def get_top_feature_names(preprocessor, X_columns):
    feature_names = []
    # numeric names
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(list(num_cols))
    # categorical one hot names
    cat_cols = preprocessor.transformers_[1][2]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = onehot.get_feature_names_out(cat_cols)
    feature_names.extend(cat_names.tolist())
    return feature_names


def print_feature_importance(best_pipeline, X_train, task_name: str):
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        feature_names = get_top_feature_names(preprocessor, X_train.columns)
        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        print(f"\nTop 10 feature importances for best {task_name} model:")
        print(importance_df.head(10).to_string(index=False))
    else:
        print(f"\nBest {task_name} model does not expose tree-based feature importances.")


def run_regression(train_feat: pd.DataFrame, test_feat: pd.DataFrame):
    target = "actual_arrival_delay_min"
    leakage_cols = ["actual_departure_delay_min", "delayed"]
    feature_df_train = train_feat.drop(columns=[target] + leakage_cols, errors="ignore")
    feature_df_test = test_feat.drop(columns=[target] + leakage_cols, errors="ignore")

    y_train = train_feat[target]
    y_test = test_feat[target]

    preprocessor = make_preprocessor(feature_df_train)

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    results = []
    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        pipeline.fit(feature_df_train, y_train)
        preds = pipeline.predict(feature_df_test)
        metrics = evaluate_regression(y_test, preds)
        results.append({"model": model_name, **metrics})

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = model_name
            best_pipeline = pipeline

    results = sorted(results, key=lambda x: x["mae"])
    with open(REGRESSION_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(best_pipeline, REGRESSION_MODEL_PATH)

    print("\nRegression results:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nBest regression model saved: {best_name} -> {REGRESSION_MODEL_PATH}")
    print_feature_importance(best_pipeline, feature_df_train, "regression")


def run_classification(train_feat: pd.DataFrame, test_feat: pd.DataFrame):
    target = "delayed"
    leakage_cols = ["actual_departure_delay_min", "actual_arrival_delay_min"]
    feature_df_train = train_feat.drop(columns=[target] + leakage_cols, errors="ignore")
    feature_df_test = test_feat.drop(columns=[target] + leakage_cols, errors="ignore")

    y_train = train_feat[target].astype(int)
    y_test = test_feat[target].astype(int)

    preprocessor = make_preprocessor(feature_df_train)

    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    results = []
    best_name = None
    best_pipeline = None
    best_f1 = -1.0

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        pipeline.fit(feature_df_train, y_train)
        preds = pipeline.predict(feature_df_test)

        y_prob = None
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob = pipeline.predict_proba(feature_df_test)[:, 1]
        elif hasattr(pipeline.named_steps["model"], "decision_function"):
            scores = pipeline.decision_function(feature_df_test)
            y_prob = 1 / (1 + np.exp(-scores))

        metrics = evaluate_classification(y_test, preds, y_prob=y_prob)
        results.append({"model": model_name, **metrics})

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_pipeline = pipeline

    results = sorted(results, key=lambda x: (x["f1"], x["roc_auc"] or -1), reverse=True)
    with open(CLASSIFICATION_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(best_pipeline, CLASSIFICATION_MODEL_PATH)

    print("\nClassification results:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nBest classification model saved: {best_name} -> {CLASSIFICATION_MODEL_PATH}")
    print_feature_importance(best_pipeline, feature_df_train, "classification")


def main():
    print("Running unified training for regression + classification...")
    raw_df = load_data(DATA_PATH)

    # time splitting before feature building, so split respects chronology
    train_raw, test_raw = time_ordered_split(raw_df, test_size=0.2)
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    run_regression(train_feat, test_feat)
    run_classification(train_feat, test_feat)

    print("\nDone. Generated files:")
    print(f"- {REGRESSION_METRICS_PATH}")
    print(f"- {CLASSIFICATION_METRICS_PATH}")
    print(f"- {REGRESSION_MODEL_PATH}")
    print(f"- {CLASSIFICATION_MODEL_PATH}")


if __name__ == "__main__":
    main()
