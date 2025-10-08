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

# Load the 2025 Azerbaijan session data (Round 17) for training
try:
    session_2025 = fastf1.get_session(2025, 17, "R")
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
    print(f"Could not load 2025 Azerbaijan GP data from fastf1. Using placeholder data. Error: {e}")
    drivers_list = ["VER", "SAI", "LAW", "NOR", "PIA", "RUS", "ANT", "LEC", "HAM", "TSU", "ALO", "STR", "OCO", "GAS", "ALB", "HUL", "HAD", "BOR", "PER", "BOT"]
    placeholder_laps = {'Driver': np.random.choice(drivers_list, 200), 'LapTime (s)': np.random.uniform(88, 95, 200)}
    laps_2025 = pd.DataFrame(placeholder_laps)
    sector_times_2025 = pd.DataFrame({'Driver': drivers_list, 'TotalSectorTime (s)': np.random.uniform(87, 94, len(drivers_list))})

# Clean air race pace data (adjusted for Baku 2025: long straights, low downforce; ~88-92s laps)
clean_air_race_pace = {
    "VER": 88.2, "SAI": 88.5, "LAW": 88.8, "NOR": 88.4, "PIA": 88.3, "RUS": 89.0,
    "ANT": 89.1, "LEC": 88.9, "HAM": 89.2, "TSU": 89.5, "ALO": 89.3, "STR": 89.7,
    "OCO": 90.0, "GAS": 89.8, "ALB": 89.4, "HUL": 90.2, "HAD": 89.6, "BOR": 90.1,
    "PER": 88.7, "BOT": 90.3
}

# Qualifying data from the 2025 Azerbaijan GP (adjusted post-Ocon DQ; times in seconds, approx. based on pole ~1:28.8)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "SAI", "LAW", "NOR", "RUS", "ANT", "LEC", "HAM", "TSU", "ALO", "STR", "PIA", "GAS", "OCO", "ALB", "HUL", "HAD", "BOR", "PER", "BOT"],
    "QualifyingTime (s)": [
        88.792,  # VER P1
        88.923,  # SAI P2
        89.015,  # LAW P3
        89.112,  # NOR P4? (Q3, but scruffy lap)
        89.245,  # RUS P5
        89.378,  # ANT P6
        None,    # LEC P10 (no time in crash)
        89.512,  # HAM P12 (Q2)
        89.645,  # TSU P9
        89.789,  # ALO P11
        89.923,  # STR P8
        90.056,  # PIA P9 (crash, no final time)
        90.200,  # GAS P13
        None,    # OCO DQ from P18
        90.322,  # ALB P14
        90.455,  # HUL P15
        90.588,  # HAD P16
        90.721,  # BOR P17
        90.854,  # PER P19 (est.)
        90.987   # BOT P20 (est.)
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Weather data (cool, windy, low rain chance)
rain_probability = 0.20
temperature = 21.0

qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructors' standings BEFORE Azerbaijan GP (McLaren dominant)
team_points = {
    "McLaren": 617, "Ferrari": 280, "Mercedes": 260, "Red Bull": 250, "Williams": 150,
    "Aston Martin": 120, "Alpine": 80, "Racing Bulls": 70, "Haas": 50, "Kick Sauber": 30
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Driver to team mapping (2025 grid)
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "SAI": "Williams", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "OCO": "Haas", "HUL": "Haas", "STR": "Aston Martin", "ALB": "Williams",
    "ANT": "Mercedes", "LAW": "Racing Bulls", "GAS": "Alpine", "HAD": "Racing Bulls", "BOR": "Kick Sauber",
    "PER": "Red Bull", "BOT": "Kick Sauber"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Average position change at Baku (overtaking heaven in straights; positive for midfield chaos-lovers)
average_position_change_azerbaijan = {
    "VER": -0.6, "SAI": -0.3, "LAW": 0.2, "NOR": -0.1, "RUS": 0.1, "ANT": 0.3,
    "LEC": 0.4, "HAM": 0.5, "TSU": 0.6, "ALO": 0.0, "STR": -0.2, "PIA": -0.4,  # Penalize crash
    "GAS": 0.2, "OCO": -0.1, "ALB": 0.7, "HUL": 0.1, "HAD": 0.5, "BOR": 0.0,
    "PER": -0.5, "BOT": -0.3
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_azerbaijan)

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
model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=3, random_state=42)
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
print("\n🏆 Predicted Top 3 for Azerbaijan GP 🏆")
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
plt.barh(X.columns, importances, color='blue')  # Blue for Azerbaijan theme
plt.xlabel("Importance")
plt.title("Feature Importance in Azerbaijan GP Race Time Prediction")
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("azerbaijan_gp_feature_importance.png", dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved to azerbaijan_gp_feature_importance.png")
# plt.show()  # Uncomment to display inline