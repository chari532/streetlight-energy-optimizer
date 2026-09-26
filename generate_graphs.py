import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




os.makedirs("visualizations", exist_ok=True)


folder = "/content"

files = glob.glob(
    os.path.join(
        folder,
        "*_ENERGYMETER.csv"
    )
)

df_list = []

for file in files:
    temp = pd.read_csv(file)
    df_list.append(temp)

df = pd.concat(
    df_list,
    ignore_index=True
)

print("Dataset shape:", df.shape)



df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    ["device_id", "timestamp"]
)

df["value_change"] = (
    df.groupby("device_id")["value"].diff()
)

df["energy_consumption"] = (
    df["value_change"]
    .where(df["value_change"] >= 0)
)

df_clean = df.dropna(
    subset=["energy_consumption"]
).copy()



Q1 = df_clean[
    "energy_consumption"
].quantile(0.25)

Q3 = df_clean[
    "energy_consumption"
].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df_clean[
    (df_clean["energy_consumption"] >= lower)
    &
    (df_clean["energy_consumption"] <= upper)
].copy()




df_clean["hour"] = (
    df_clean["timestamp"].dt.hour
)

df_clean["day_of_week"] = (
    df_clean["timestamp"].dt.dayofweek
)


hourly_energy = (
    df_clean
    .groupby("hour")["energy_consumption"]
    .mean()
    .reindex(range(24))
)

plt.figure(figsize=(12, 6))

plt.plot(
    hourly_energy.index,
    hourly_energy.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Streetlight Energy Consumption by Hour")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/energy_by_hour.png",
    dpi=300
)

plt.close()

print(
    "Saved: visualizations/energy_by_hour.png"
)




day_energy = (
    df_clean
    .groupby("day_of_week")[
        "energy_consumption"
    ]
    .mean()
    .reindex(range(7))
)

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(10, 6))

plt.bar(
    days,
    day_energy.values
)

plt.xlabel("Day of Week")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title(
    "Streetlight Energy Consumption by Day of Week"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "visualizations/energy_by_day.png",
    dpi=300
)

plt.close()

print(
    "Saved: visualizations/energy_by_day.png"
)




model = joblib.load(
    "streetlight_energy_model.pkl"
)

model_features = joblib.load(
    "model_features.pkl"
)




df_clean["day"] = (
    df_clean["timestamp"].dt.day
)

df_clean["month"] = (
    df_clean["timestamp"].dt.month
)

df_clean["is_weekend"] = (
    df_clean["day_of_week"] >= 5
).astype(int)

df_clean["hour_sin"] = np.sin(
    2 * np.pi * df_clean["hour"] / 24
)

df_clean["hour_cos"] = np.cos(
    2 * np.pi * df_clean["hour"] / 24
)

df_clean["month_sin"] = np.sin(
    2 * np.pi * df_clean["month"] / 12
)

df_clean["month_cos"] = np.cos(
    2 * np.pi * df_clean["month"] / 12
)




prediction_data = df_clean[
    [
        "device_id",
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
    ]
].copy()


encoded = pd.get_dummies(
    prediction_data,
    columns=["device_id"],
    dtype=int
)

encoded = encoded.reindex(
    columns=model_features,
    fill_value=0
)

predictions = model.predict(
    encoded
)



actual = df_clean[
    "energy_consumption"
].values


actual_plot = actual[:100]
predicted_plot = predictions[:100]

plt.figure(figsize=(12, 6))

plt.plot(
    actual_plot,
    label="Actual"
)

plt.plot(
    predicted_plot,
    label="Predicted"
)

plt.xlabel("Time Period")
plt.ylabel("Energy Consumption (kWh)")
plt.title(
    "Actual vs Predicted Streetlight Energy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print(
    "Saved: visualizations/actual_vs_predicted.png"
)



importance = pd.DataFrame({
    "Feature": model_features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png",
    dpi=300
)

plt.close()

print(
    "Saved: visualizations/feature_importance.png"
)



threshold = df_clean[
    "energy_consumption"
].quantile(0.75)

recommendations = np.where(
    predictions > threshold,
    "Reduce/DIM lighting",
    "Normal lighting"
)

recommendation_counts = (
    pd.Series(recommendations)
    .value_counts()
)

plt.figure(figsize=(8, 6))

plt.bar(
    recommendation_counts.index,
    recommendation_counts.values
)

plt.xlabel("Recommendation")
plt.ylabel("Number of Time Periods")
plt.title(
    "Streetlight Lighting Recommendations"
)

plt.tight_layout()

plt.savefig(
    "visualizations/lighting_recommendations.png",
    dpi=300
)

plt.close()

print(
    "Saved: visualizations/lighting_recommendations.png"
)



print("\nAll PNG graphs generated successfully!")

print("\nFiles:")
print("1. energy_by_hour.png")
print("2. energy_by_day.png")
print("3. actual_vs_predicted.png")
print("4. feature_importance.png")
print("5. lighting_recommendations.png")