import argparse
import joblib
import pandas as pd


def load_data(path="public_transport_delays.csv"):
    return pd.read_csv(path)


def add_features(df):
    df = df.copy()

    # event_type
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].fillna("NoEvent").replace("", "NoEvent")
    else:
        df["event_type"] = "NoEvent"

    # for date-derived features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_year"] = df["date"].dt.dayofyear
        df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    else:
        df["month"] = 1
        df["day"] = 1
        df["day_of_year"] = 1
        df["is_weekend"] = 0

    # parsing HH:MM time columns
    def split_time(colname, prefix):
        if colname in df.columns:
            t = df[colname].astype(str).str.split(":", expand=True)
            df[f"{prefix}_hour"] = pd.to_numeric(t[0], errors="coerce").fillna(0).astype(int)
            df[f"{prefix}_minute"] = pd.to_numeric(t[1], errors="coerce").fillna(0).astype(int)
        else:
            df[f"{prefix}_hour"] = 0
            df[f"{prefix}_minute"] = 0

    split_time("scheduled_departure", "scheduled_departure")
    split_time("scheduled_arrival", "scheduled_arrival")

    df["scheduled_departure_minute_of_day"] = (
        df["scheduled_departure_hour"] * 60 + df["scheduled_departure_minute"]
    )
    df["scheduled_arrival_minute_of_day"] = (
        df["scheduled_arrival_hour"] * 60 + df["scheduled_arrival_minute"]
    )
    df["scheduled_trip_duration_min"] = (
        df["scheduled_arrival_minute_of_day"] - df["scheduled_departure_minute_of_day"]
    ).clip(lower=0)

    df["is_morning_peak"] = df["scheduled_departure_hour"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["scheduled_departure_hour"].between(16, 19).astype(int)
    df["is_commute_peak"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)

    # helper for numeric columns with different possible name variations
    def get_numeric_col(possible_names, default=0):
        for name in possible_names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series([default] * len(df), index=df.index)

    df["has_event"] = (df["event_type"].astype(str).str.lower() != "noevent").astype(int)

    event_size = get_numeric_col(["event_attendance", "event_size", "attendance"], default=0)
    df["event_size_bucket"] = pd.cut(
        event_size,
        bins=[-1, 0, 1000, 10000, float("inf")],
        labels=["NoEvent", "Small", "Medium", "Large"]
    ).astype(str)

    weather_col = "weather_condition" if "weather_condition" in df.columns else None
    weather_text = df[weather_col].astype(str).str.lower() if weather_col else pd.Series([""] * len(df), index=df.index)

    df["is_rain"] = weather_text.str.contains("rain", na=False).astype(int)
    df["is_snow"] = weather_text.str.contains("snow", na=False).astype(int)
    df["is_storm"] = weather_text.str.contains("storm|thunder", na=False).astype(int)
    df["is_fog"] = weather_text.str.contains("fog|mist|haze", na=False).astype(int)
    df["is_bad_weather"] = (
        (df["is_rain"] == 1) | (df["is_snow"] == 1) | (df["is_storm"] == 1) | (df["is_fog"] == 1)
    ).astype(int)

    precip = get_numeric_col(["precipitation_mm", "precipitation_MM"])
    wind = get_numeric_col(["wind_speed_kmh", "wind_speed_KMH"])
    humidity = get_numeric_col(["humidity_percent", "humidity_%"])
    temp = get_numeric_col(["temperature_C", "temperature_c", "temperature"])
    traffic = get_numeric_col(["traffic_congestion_level"])

    df["is_heavy_precip"] = (precip >= 5).astype(int)
    df["is_high_wind"] = (wind >= 25).astype(int)
    df["is_high_humidity"] = (humidity >= 80).astype(int)
    df["is_extreme_temp"] = ((temp <= 0) | (temp >= 35)).astype(int)
    df["is_high_congestion"] = (traffic >= 7).astype(int)

    return df


def align_to_model_columns(input_df, model):
    """
    Ensuring the prediction dataframe contains the exact column names expected by the saved pipeline.
    also handles case mismatches like temperature_C vs temperature_c.
    """
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)
    else:
        return input_df

    existing_map = {col.lower(): col for col in input_df.columns}

    for expected in expected_cols:
        if expected not in input_df.columns:
            lowered = expected.lower()
            if lowered in existing_map:
                input_df[expected] = input_df[existing_map[lowered]]

    return input_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    args = parser.parse_args()

    if args.task == "classification":
        model = joblib.load("best_delay_classifier.joblib")
    else:
        model = joblib.load("best_delay_regressor.joblib")

    df = load_data()
    df = add_features(df)

    # using first row as demo inference input
    input_df = df.iloc[[0]].copy()
    input_df = align_to_model_columns(input_df, model)

    if args.task == "classification":
        pred = model.predict(input_df)[0]
        print("Prediction task: classification")
        print("Predicted delayed:", pred)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            print("Class probabilities:", proba)
    else:
        pred = model.predict(input_df)[0]
        print("Prediction task: regression")
        print("Predicted arrival delay (minutes):", round(float(pred), 2))


if __name__ == "__main__":
    main()