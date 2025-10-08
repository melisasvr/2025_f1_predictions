import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Enable the FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# MODIFIED: Load the 2024 Spanish session data (Round 10)
session_2024 = fastf1.get_session(2024, 10, "R") 
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"] 
)

# Clean air race pace data (this is generally driver-specific, not track-specific)
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128, "GAS": 94.8, "TSU": 95.1, "ALB": 95.5
}

# MODIFIED: Qualifying data from the 2024 Spanish GP to predict 2025
# Times are in seconds (e.g., 1:11.383 -> 71.383)
qualifying_2025 = pd.DataFrame({
    "Driver": ["NOR", "VER", "HAM", "RUS", "LEC", "SAI", "GAS", "OCO", "PIA", "ALO", "STR", "HUL", "ALB"],
    "QualifyingTime (s)": [
        71.383,  # NOR
        71.403,  # VER
        71.701,  # HAM
        71.703,  # RUS
        71.731,  # LEC
        71.736,  # SAI
        71.857,  # GAS
        72.125,  # OCO
        72.011,  # PIA (Using Q2 time as Q3 lap was deleted)
        72.128,  # ALO
        None,    # STR (Did not make Q3)
        None,    # HUL (Did not make Q3)
        None     # ALB (Did not make Q3)
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Weather data (assuming dry conditions for prediction)
rain_probability = 0.05
temperature = 28.0

qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor's data
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Haas", "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# MODIFIED: Average position change at Spain (estimated values)
# Positive means gaining positions, negative means losing. Spain allows more overtaking than Monaco.
average_position_change_spain = {
    "VER": 0.5, "NOR": 0.2, "PIA": 0.0, "RUS": -0.3, "SAI": 0.1, "ALB": -0.2,
    "LEC": -0.5, "OCO": 0.4, "HAM": 0.8, "STR": -0.8, "GAS": 0.2, "ALO": 1.0, "HUL": -0.5
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_spain)

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Filter for drivers present in both datasets
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers].copy() # Use .copy() to avoid SettingWithCopyWarning

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]].copy()
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values for features and target
imputer_X = SimpleImputer(strategy="median")
X_imputed = imputer_X.fit_transform(X)

# Impute missing values in y if any (though less likely)
y = y.fillna(y.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Spanish GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Sort final results and print the podium
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='coral')
plt.xlabel("Importance")
plt.title("Feature Importance in Spanish GP Race Time Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show() # Commented out to prevent the plot window from blocking script execution