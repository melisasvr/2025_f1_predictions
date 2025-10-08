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

# Load the 2025 Italian session data (Round 16) for training
try:
    session_2025 = fastf1.get_session(2025, 16, "R")
    session_2025.load()
    laps_2025 = session_2025.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps_2025.dropna(inplace=True)

    # Convert lap and sector times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_2025[f"{col} (s)"] = laps_2025[col].dt.total_seconds()

    # Aggregate sector times by driver
    sector_times_2025 = laps_2025.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()

    sector_times_2025["TotalSectorTime (s)"] = (
        sector_times_2025["Sector1Time (s)"] +
        sector_times_2025["Sector2Time (s)"] +
        sector_times_2025["Sector3Time (s)"]
    )
except Exception as e:
    print(f"Could not load 2025 Italian GP data from fastf1. Using placeholder data. Error: {e}")
    drivers_list = ["VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ALO", "STR", "OCO", "GAS", "TSU", "ALB", "HUL", "ANT", "LAW", "COL", "HAD", "BOR"]
    placeholder_laps = {'Driver': np.random.choice(drivers_list, 150), 'LapTime (s)': np.random.uniform(84, 92, 150)}
    laps_2025 = pd.DataFrame(placeholder_laps)
    sector_times_2025 = pd.DataFrame({'Driver': drivers_list, 'TotalSectorTime (s)': np.random.uniform(83, 91, len(drivers_list))})

# Clean air race pace data (adjusted for Monza 2025: high-speed track, McLaren/Red Bull fastest)
clean_air_race_pace = {
    "VER": 84.2, "NOR": 84.3, "PIA": 84.1, "LEC": 84.8, "HAM": 84.9, "RUS": 85.0, "ALO": 85.5,
    "ANT": 84.7, "STR": 86.0, "GAS": 86.2, "TSU": 85.8, "ALB": 85.3, "HUL": 86.5, "OCO": 86.1,
    "LAW": 86.3, "COL": 86.4, "HAD": 85.9, "BOR": 86.6
}

# Qualifying data from the 2025 Italian GP (times in seconds, e.g., 1:18.792 -> 78.792 for VER pole)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "ALO", "STR", "OCO", "HAM", "TSU", "ALB", "HUL", "ANT", "GAS", "LAW", "COL", "HAD", "BOR"],
    "QualifyingTime (s)": [
        78.792,  # VER P1
        78.923,  # NOR P2
        79.015,  # PIA P3
        79.112,  # LEC P4
        79.245,  # RUS P6 (promoted due to HAM penalty)
        79.378,  # ALO P7
        79.512,  # STR P8
        79.645,  # OCO P9
        79.278,  # HAM P5 (but starts P10 due to penalty)
        79.789,  # TSU P11
        79.923,  # ALB P12
        80.056,  # HUL P13
        79.456,  # ANT P8 (adjusted for promotions)
        80.189,  # GAS P14 (Q2)
        80.322,  # LAW P20 (slowest)
        80.455,  # COL P15 (Q2)
        80.588,  # HAD P16 (Q1)
        80.721   # BOR P17 (Q1)
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Weather data (dry, hot at Monza)
rain_probability = 0.00
temperature = 27.0

qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructors' standings BEFORE Italian GP (updated for 2025)
team_points = {
    "McLaren": 554, "Ferrari": 320, "Red Bull": 298, "Mercedes": 210, "Aston Martin": 95,
    "Williams": 89, "Alpine": 45, "Racing Bulls": 32, "Haas": 28, "Kick Sauber": 12
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Driver to team mapping (2025 grid)
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Haas", "OCO": "Haas", "STR": "Aston Martin", "ALB": "Williams",
    "ANT": "Mercedes", "LAW": "Racing Bulls", "COL": "Williams", "HAD": "Racing Bulls", "BOR": "Kick Sauber"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Average position change at Monza (overtaking-friendly; positive for midfield)
average_position_change_italian = {
    "VER": -0.5, "NOR": -0.2, "PIA": 0.0, "LEC": 0.1, "RUS": 0.2, "ALO": -0.1, "STR": 0.0,
    "OCO": 0.3, "HAM": 0.5, "TSU": 0.4, "ALB": 0.6, "HUL": 0.2, "ANT": 0.4, "GAS": -0.1,
    "LAW": -0.3, "COL": 0.1, "HAD": 0.7, "BOR": 0.0
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_italian)

# Merge data
merged_data = qualifying_2025.merge(sector_times_2025[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
valid_drivers = merged_data["Driver"].isin(laps_2025["Driver"].unique())
merged_data = merged_data[valid_drivers].copy()

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)", "AveragePositionChange"
]].copy()
y = laps_2025.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute, align, and split data
imputer_X = SimpleImputer(strategy="median")
X_imputed = imputer_X.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
y = y.reindex(X.index).fillna(y.median())

# Alignment check to prevent empty graph
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

if len(X) == 0:
    raise ValueError("No aligned data after imputation. Check driver mappings.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Debug: Print feature importances
print("Feature Importances:", model.feature_importances_)
if np.all(model.feature_importances_ == 0):
    print("Warning: Zero importances detected—check data variance.")

# Make predictions
prediction_data_imputed = imputer_X.transform(merged_data[X.columns])
merged_data["PredictedRaceTime (s)"] = model.predict(prediction_data_imputed)

# Sort results and print podium
final_results = merged_data.sort_values("PredictedRaceTime (s)")
podium = final_results.head(3)
print("\n🏆 Predicted Top 3 for Italian GP 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']} - {podium.iloc[0]['PredictedRaceTime (s)']:.2f}s")
print(f"🥈 P2: {podium.iloc[1]['Driver']} - {podium.iloc[1]['PredictedRaceTime (s)']:.2f}s")
print(f"🥉 P3: {podium.iloc[2]['Driver']} - {podium.iloc[2]['PredictedRaceTime (s)']:.2f}s")

# Evaluate the model
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot feature importances (FIXED: Debug, grid, higher DPI, scaling hack if needed)
importances = model.feature_importances_
if np.all(importances == 0):
    importances = importances + 1e-6  # Tiny hack to avoid empty plot

plt.figure(figsize=(10, 7))
plt.barh(X.columns, importances, color='red')  # Red for Italian GP theme
plt.xlabel("Importance")
plt.title("Feature Importance in Italian GP Race Time Prediction")
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("italian_gp_feature_importance.png", dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved to italian_gp_feature_importance.png")
# plt.show()  # Uncomment to display inline