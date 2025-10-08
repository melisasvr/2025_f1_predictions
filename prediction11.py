import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Enable the FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# MODIFIED: Load the 2024 Canadian session data (Round 9)
try:
    session_2024 = fastf1.get_session(2024, 9, "R")
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
except Exception as e:
    print(f"Could not load 2024 Canadian GP data from fastf1. Using placeholder data. Error: {e}")
    # Create placeholder data if API fails
    drivers_list = ["VER", "NOR", "PIA", "LEC", "SAI", "HAM", "RUS", "ALO", "STR", "OCO", "GAS", "TSU", "ALB", "HUL"]
    placeholder_laps = {
        'Driver': np.random.choice(drivers_list, 100),
        'LapTime (s)': np.random.uniform(75, 85, 100)
    }
    laps_2024 = pd.DataFrame(placeholder_laps)
    sector_times_2024 = pd.DataFrame({
        'Driver': drivers_list,
        'TotalSectorTime (s)': np.random.uniform(74, 84, len(drivers_list))
    })


# Clean air race pace data
clean_air_race_pace = {
    "VER": 93.1, "HAM": 94.0, "LEC": 93.4, "NOR": 93.4, "ALO": 94.7, "PIA": 93.2,
    "RUS": 93.8, "SAI": 94.4, "STR": 95.3, "HUL": 95.3, "OCO": 95.6, "GAS": 94.8,
    "TSU": 95.1, "ALB": 95.5
}

# MODIFIED: Qualifying data from the 2024 Canadian GP to predict 2025
# Times are in seconds (e.g., 1:12.000 -> 72.000)
qualifying_2025 = pd.DataFrame({
    "Driver": ["RUS", "VER", "NOR", "PIA", "ALO", "HAM", "STR", "ALB", "LEC", "SAI", "GAS", "OCO", "HUL", "TSU"],
    "QualifyingTime (s)": [
        72.000,  # RUS
        72.000,  # VER
        72.021,  # NOR
        72.103,  # PIA
        72.133,  # ALO
        72.280,  # HAM
        72.389,  # STR
        72.498,  # ALB
        72.697,  # LEC (Q2 Time)
        72.711,  # SAI (Q2 Time)
        None,    # GAS
        None,    # OCO
        None,    # HUL
        None     # TSU
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Weather data (Canada can be variable, assuming partially damp conditions)
rain_probability = 0.25
temperature = 18.0

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

# MODIFIED: Average position change at Canada (estimated values)
# The "Wall of Champions" can cause incidents, making for unpredictable results.
average_position_change_canada = {
    "VER": 0.3, "NOR": -0.1, "PIA": 0.1, "RUS": -0.5, "SAI": -0.2, "ALB": 0.5,
    "LEC": 0.8, "OCO": -0.3, "HAM": 1.2, "STR": 0.0, "GAS": -0.4, "ALO": 0.6, "HUL": -0.6, "TSU": 0.2
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_canada)

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Filter for drivers present in both datasets
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers].copy()

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)", "AveragePositionChange"
]].copy()
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values
imputer_X = SimpleImputer(strategy="median")
X_imputed = imputer_X.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Ensure y aligns with X after filtering/imputing and handle potential NaNs
y = y.reindex(X.index).fillna(y.median())
X, y = X.align(y, axis=0, join='inner') # Ensure perfect alignment

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
prediction_data_imputed = imputer_X.transform(merged_data[X.columns])
merged_data["PredictedRaceTime (s)"] = model.predict(prediction_data_imputed)


# Sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Canadian GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Sort final results and print the podium
podium = final_results.head(3)

print("\n🏆 Predicted Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='darkred')
plt.xlabel("Importance")
plt.title("Feature Importance in Canadian GP Race Time Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show() # Commented out to prevent the plot window from blocking script execution