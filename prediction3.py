import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Japanese GP race session, lap and sector times
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Manually input qualifying data for 2025 Japanese GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.836, 88.570, 88.696, 89.271]
})

# Add wet performance factor from script
driver_wet_performance = {
    "VER": 0.975196, 
    "HAM": 0.976464,  
    "LEC": 0.975862,  
    "NOR": 0.978179,  
    "ALO": 0.972655,  
    "RUS": 0.968678,  
    "SAI": 0.978754,  
    "TSU": 0.996338,  
    "OCO": 0.981810,  
    "GAS": 0.978832,  
    "STR": 0.979857   
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Weather Data
API_KEY = "YOURAPIKEY"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
print(weather_data)

# Extract the relevant weather data for the race (Sunday at 2pm local time)
forecast_time = "2025-04-05 14:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

if forecast_data:
    rain_probability = forecast_data["pop"]
    temperature = forecast_data["main"]["temp"]  
else:
    rain_probability = 0 
    temperature = 20 

# Merge qualifying data with sector times data 
merged_data = qualifying_2025.merge(sector_times_2024, left_on="Driver", right_on="Driver", how="left")

# Create weather features for the model
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define feature set (Qualifying + Sector Times + Weather + Wet Performance Factor)
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor", "RainProbability", "Temperature"]].fillna(0)

# Use average lap time as the target
y = merged_data.merge(laps_2024.groupby("Driver")["LapTime (s)"].mean(), left_on="Driver", right_index=True)["LapTime (s)"]

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Japanese GP Winner🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate 
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
