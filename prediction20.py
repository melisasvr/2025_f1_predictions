import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# This will create the 'f1_cache' folder automatically if it's missing
fastf1.Cache.enable_cache("f1_cache")

# --- 2024 Training Data (United States GP) ---
# We use the 2024 US GP (Round 19) to train the model
print("Loading 2024 US GP (Round 19) data for training...")
session_2024 = fastf1.get_session(2024, 19, "R")
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
print("Training data loaded.")

# --- 2025 Prediction Data and Feature Assumptions (United States GP) ---

# 2025 Qualifying Data (Converted from 1:XX.XXX to seconds)
# These are the official results from today's (Oct 18, 2025) session
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "LEC", "RUS", "HAM", "PIA", "ANT"],
    "QualifyingTime (s)": [  
        92.510,  # 1. M. Verstappen (1:32.510)
        92.801,  # 2. L. Norris (1:32.801)
        92.807,  # 3. C. Leclerc (1:32.807)
        92.826,  # 4. G. Russell (1:32.826)
        92.912,  # 5. L. Hamilton (1:32.912)
        93.084,  # 6. O. Piastri (1:33.084)
        93.114,  # 7. A.K. Antonelli (1:33.114)
    ]
})

# **************************************************************************
# * !!! YOU MUST EDIT THIS SECTION !!!                     *
# * These are your model's assumptions. Update them for COTA (Austin).  *
# **************************************************************************

# 1. Estimated Clean Air Race Pace for COTA (Fictional - EDIT THESE VALUES)
clean_air_race_pace = {
    "VER": 97.500,  # Your estimate for VER race pace
    "NOR": 97.600,  # Your estimate for NOR race pace
    "LEC": 97.700,  # Your estimate for LEC race pace
    "RUS": 97.800,  # Your estimate for RUS race pace
    "HAM": 97.850,  # Your estimate for HAM race pace
    "PIA": 97.750,  # Your estimate for PIA race pace
    "ANT": 98.000   # Your estimate for ANT race pace
}
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# 2. Average Position Change at COTA (Fictional - EDIT THESE VALUES)
# COTA has a big run to T1. Negative = gains positions, Positive = loses positions
average_position_change_COTA = {
    "VER": -0.2, # Your estimate for VER start
    "NOR": 0.1,  # Your estimate for NOR start
    "LEC": 0.0,  # Your estimate for LEC start
    "RUS": 0.3,  # Your estimate for RUS start
    "HAM": 0.2,  # Your estimate for HAM start
    "PIA": 0.4,  # Your estimate for PIA start
    "ANT": 0.5   # Your estimate for ANT start
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_COTA)

# **************************************************************************
# * (End of data for you to edit)                      *
# **************************************************************************


# --- Weather Data (Updated for 2025 US GP Race Day) ---
rain_probability = 0.2  # 20% chance of rain
temperature = 32.0      # 32°C forecast for race start
# --------------------------

# Set QualifyingTime feature for model input
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor's data (Kept general for 2025 season performance)
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Ferrari": 114, "Williams": 51
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Updated driver-to-team map (Now includes Ferrari drivers)
driver_to_team = {
    "VER": "Red Bull", 
    "NOR": "McLaren", 
    "PIA": "McLaren", 
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "LEC": "Ferrari", # Added
    "HAM": "Ferrari"  # Added
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)


# --- Model Training and Prediction ---

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Filter to include only drivers present in both prediction input and 2024 race data
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Handle potential missing drivers (e.g., ANT was not in 2024 race)
if "ANT" in merged_data["Driver"].values:
    print("Note: 'ANT' (Antonelli) was not in the 2024 race. His 2024-based features (TotalSectorTime) will be 'NaN' and imputed.")

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
# Target is the average lap time from the 2024 race for the merged drivers
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values for features AND target
# Impute X (e.g., TotalSectorTime for ANT)
feature_imputer = SimpleImputer(strategy="median")
X_imputed = feature_imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Impute y (e.g., if a driver had no laps in 2024, their target 'y' would be NaN)
target_imputer = SimpleImputer(strategy="median")
y_imputed = target_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()


# Train-test split (using data from all laps in 2024 US race)
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y_imputed, test_size=0.3, random_state=37)

# Train Gradient Boosting Model
print("Training model...")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)
print("Model trained.")

# Predict 2025 race times using the imputed 2025 feature set
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed_df)

# sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 United States GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# Calculate and print Model Error
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# --- Output and Plotting ---

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in US GP Race Time Prediction")
plt.tight_layout()
plt.savefig("cota_feature_importance.png")
plt.close()

# sort results and get top 3
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted Top 3 for US GP 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['PredictedRaceTime (s)']:.3f}s)")
print(f"🥈 P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['PredictedRaceTime (s)']:.3f}s)")
print(f"🥉 P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['PredictedRaceTime (s)']:.3f}s)")