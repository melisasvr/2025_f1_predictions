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

# MODIFIED: Load the 2024 Belgian session data (Round 14)
try:
    session_2024 = fastf1.get_session(2024, 14, "R")
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
    print(f"Could not load 2024 Belgian GP data from fastf1. Using placeholder data. Error: {e}")
    drivers_list = ["VER", "NOR", "PIA", "LEC", "SAI", "HAM", "RUS", "ALO", "STR", "OCO", "GAS", "TSU", "ALB", "HUL", "ANT"]
    placeholder_laps = {'Driver': np.random.choice(drivers_list, 100), 'LapTime (s)': np.random.uniform(105, 115, 100)}
    laps_2024 = pd.DataFrame(placeholder_laps)
    sector_times_2024 = pd.DataFrame({'Driver': drivers_list, 'TotalSectorTime (s)': np.random.uniform(104, 114, len(drivers_list))})


# Clean air race pace data
clean_air_race_pace = {
    "VER": 93.1, "HAM": 94.0, "LEC": 93.4, "NOR": 93.4, "ALO": 94.7, "PIA": 93.2,
    "RUS": 93.8, "SAI": 94.4, "STR": 95.3, "HUL": 95.3, "OCO": 95.6, "GAS": 94.8,
    "TSU": 95.1, "ALB": 95.5, "ANT": 93.9
}

# MODIFIED: Qualifying data from the 2024 Belgian GP to predict 2025
# Times are in seconds (e.g., 1:58.694 -> 118.694)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "LEC", "HAM", "SAI", "PIA", "NOR", "RUS", "ALO", "STR", "OCO", "GAS", "TSU", "ALB", "HUL", "ANT"],
    "QualifyingTime (s)": [
        103.298, # VER (Adjusted from wet quali time for dry estimate)
        103.834, # LEC
        104.211, # HAM
        104.337, # SAI
        104.421, # PIA
        104.588, # NOR
        104.601, # RUS
        104.998, # ALO
        105.121, # STR
        None,    # OCO
        None,    # GAS
        None,    # TSU
        None,    # ALB
        None,    # HUL
        104.750  # ANT (Estimated)
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Weather data (It's Spa - rain is always a possibility)
rain_probability = 0.40
temperature = 17.0

qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# UPDATED: Constructor's data after British GP
team_points = {
    "McLaren": 366, "Mercedes": 220, "Red Bull": 190, "Ferrari": 210, "Kick Sauber": 21,
    "Williams": 51, "Haas": 20, "Aston Martin": 44, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Driver to team mapping
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams",
    "ANT": "Mercedes"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# MODIFIED: Average position change at Spa (estimated values)
# Famous for overtaking, so expect lots of movement.
average_position_change_belgian = {
    "VER": 0.8, "NOR": 0.5, "PIA": 0.3, "RUS": -0.1, "SAI": -0.5, "ALB": -0.4,
    "LEC": 1.2, "OCO": 0.2, "HAM": 2.5, "STR": -0.6, "GAS": 0.0, "ALO": 0.7, "HUL": -0.5, "TSU": -0.2,
    "ANT": 0.8
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_belgian)

# Merge data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers].copy()

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)", "AveragePositionChange"
]].copy()
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute, align, and split data
imputer_X = SimpleImputer(strategy="median")
X_imputed = imputer_X.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
y = y.reindex(X.index).fillna(y.median())
X, y = X.align(y, axis=0, join='inner')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
prediction_data_imputed = imputer_X.transform(merged_data[X.columns])
merged_data["PredictedRaceTime (s)"] = model.predict(prediction_data_imputed)

# Sort results and print podium
final_results = merged_data.sort_values("PredictedRaceTime (s)")
podium = final_results.head(3)
print("\n🏆 Predicted Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

# Evaluate the model
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot feature importances and save to file
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_, color='purple')
plt.xlabel("Importance")
plt.title("Feature Importance in Belgian GP Race Time Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("belgian_gp_feature_importance.png")
print("\nFeature importance plot saved to belgian_gp_feature_importance.png")
# plt.show() # Commented out to prevent the plot window from blocking script execution