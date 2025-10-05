import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

# Load the 2024 Singapore GP race session (Round 18)
session_2024 = fastf1.get_session(2024, 18, "R")
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

# --- 2025 Prediction Data and Feature Assumptions ---

# 2025 Qualifying Data (Converted from 1:XX.XXX to seconds)
qualifying_2025 = pd.DataFrame({
    "Driver": ["RUS", "VER", "PIA", "ANT", "NOR"],
    "QualifyingTime (s)": [  
        89.158,  # G. Russell (1:29.158)
        89.340,  # M. Verstappen (1:29.340)
        89.524,  # O. Piastri (1:29.524)
        89.537,  # A.K. Antonelli (1:29.537)
        89.586,  # L. Norris (1:29.586)
    ]
})

# Estimated Clean Air Race Pace for Singapore (Fictional, based on Qualifying)
clean_air_race_pace = {
    "RUS": 95.500, "VER": 95.400, "PIA": 95.600, "ANT": 95.900, "NOR": 95.550
}
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# --- Weather Data (Hardcoded to fix API error) ---
rain_probability = 0.1  # Assuming dry track
temperature = 30.0      # Assuming typical Singapore heat
# --------------------------

# Set QualifyingTime feature for model input (assuming dry track)
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor's data (Kept general for 2025 season performance)
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Ferrari": 114, "Williams": 51
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Updated driver-to-team map (Assuming Antonelli replaces Hamilton)
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "RUS": "Mercedes",
    "ANT": "Mercedes" 
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Average Position Change at Singapore (Fictional data for the provided drivers)
# Positive means losing positions (due to high risk)
average_position_change_singapore = {
    "RUS": 0.5, "VER": -0.5, "PIA": 0.2, "ANT": 0.0, "NOR": 1.0
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_singapore)

# --- Model Training and Prediction ---

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Filter to include only drivers present in both prediction input and 2024 race data
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
# Target is the average lap time from the 2024 race for the merged drivers
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values for features (e.g. if RUS had DNF in quali, or ANT data is missing)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split (using data from all laps in 2024 Singapore race)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)

# Predict 2025 race times using the imputed 2025 feature set
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Singapore GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# --- Output and Plotting ---

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.savefig("singapore_feature_importance.png")
plt.close()

# sort results and get top 3
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted in the Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")