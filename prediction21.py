import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("2025 MEXICO GP RACE PREDICTION")
print("=" * 60)

# --- 2025 Prediction Data (Mexico GP) ---
qualifying_2025 = pd.DataFrame({
    "Driver": ["NOR", "LEC", "HAM", "RUS", "VER", "ANT", "SAI", "PIA"],
    "QualifyingPosition": [1, 2, 3, 4, 5, 6, 7, 8],
    "QualifyingTime (s)": [  
        75.586,  # 1. L. Norris (1:15.586)
        75.848,  # 2. C. Leclerc (1:15.848)
        75.938,  # 3. L. Hamilton (1:15.938)
        76.034,  # 4. G. Russell (1:16.034)
        76.070,  # 5. M. Verstappen (1:16.070)
        76.118,  # 6. A.K. Antonelli (1:16.118)
        76.172,  # 7. C. Sainz (1:16.172)
        76.174   # 8. O. Piastri (1:16.174)
    ]
})

# ===============================================
# EDITABLE SECTION: Adjust these values to tune predictions
# ===============================================

# 1. Expected race pace (lap time in seconds)
# Based on: quali time + race fuel + tire deg + track evolution
race_pace = {
    "NOR": 80.8,   # P1 quali, strong McLaren
    "LEC": 80.4,   # P2 quali + US P3 form = BEST race pace
    "HAM": 81.0,   # P3 quali, solid Ferrari pace
    "RUS": 81.2,   # P4 quali, competitive Mercedes
    "VER": 80.5,   # P5 quali BUT US win + aggressive = excellent pace
    "ANT": 81.8,   # P6 quali, rookie = conservative
    "SAI": 81.5,   # P7 quali, Williams = midfield
    "PIA": 81.3    # P8 quali, slightly off Norris
}

# 2. Overtaking potential (negative = likely to gain positions)
# Mexico has long straight = slipstream overtakes possible
overtaking_ability = {
    "NOR": 0.2,    # P1 vulnerable but strong defense
    "LEC": -0.8,   # AGGRESSIVE: P2 + Ferrari top speed + recent form
    "HAM": -0.3,   # P3, decent gain potential
    "RUS": 0.0,    # P4, neutral
    "VER": -1.0,   # MAX FACTOR: P5 with best overtaking + US momentum
    "ANT": 0.4,    # P6, cautious rookie
    "SAI": 0.3,    # P7, hard to gain from there
    "PIA": 0.2     # P8, tough to move up
}

# 3. Recent form multiplier (1.0 = neutral, <1.0 = boost, >1.0 = penalty)
recent_form = {
    "NOR": 1.00,   # Consistent but no recent win
    "LEC": 0.97,   # 3% BOOST for US P3
    "HAM": 1.00,   # Steady
    "RUS": 1.00,   # Steady
    "VER": 0.95,   # 5% BOOST for US win dominance
    "ANT": 1.02,   # 2% penalty for rookie nerves
    "SAI": 1.00,   # Neutral
    "PIA": 1.00    # Neutral
}

# 4. Team reliability factor (1.0 = reliable, >1.0 = risk of issues)
reliability = {
    "NOR": 1.00,   # McLaren reliable
    "LEC": 1.00,   # Ferrari solid
    "HAM": 1.00,   # Ferrari solid
    "RUS": 1.00,   # Mercedes solid
    "VER": 1.00,   # Red Bull reliable
    "ANT": 1.01,   # Rookie 1% higher risk
    "SAI": 1.00,   # Williams okay
    "PIA": 1.00    # McLaren reliable
}

# ===============================================
# END EDITABLE SECTION
# ===============================================

# Apply factors to calculate predicted average race lap time
qualifying_2025["RacePace"] = qualifying_2025["Driver"].map(race_pace)
qualifying_2025["OvertakingFactor"] = qualifying_2025["Driver"].map(overtaking_ability)
qualifying_2025["FormMultiplier"] = qualifying_2025["Driver"].map(recent_form)
qualifying_2025["ReliabilityFactor"] = qualifying_2025["Driver"].map(reliability)

# Calculate predicted race performance score
# Lower score = better predicted finish
qualifying_2025["RaceScore"] = (
    qualifying_2025["RacePace"] * 
    qualifying_2025["FormMultiplier"] * 
    qualifying_2025["ReliabilityFactor"]
)

# Adjust for overtaking (subtract because negative = gains positions)
qualifying_2025["RaceScore"] = (
    qualifying_2025["RaceScore"] - 
    (qualifying_2025["OvertakingFactor"] * 0.3)  # 0.3s per position change
)

# Sort by race score to get predicted finishing order
final_results = qualifying_2025.sort_values("RaceScore").reset_index(drop=True)
final_results["PredictedPosition"] = range(1, len(final_results) + 1)

# Display results
print("\n🏁 PREDICTED RACE RESULTS 🏁\n")
print(final_results[[
    "PredictedPosition", 
    "Driver", 
    "QualifyingPosition",
    "RaceScore",
    "RacePace"
]].to_string(index=False))

# Podium
print("\n" + "=" * 60)
print("🏆 PREDICTED PODIUM 🏆")
print("=" * 60)
podium = final_results.head(3)
print(f"🥇 P1: {podium.iloc[0]['Driver']} (Quali P{podium.iloc[0]['QualifyingPosition']})")
print(f"🥈 P2: {podium.iloc[1]['Driver']} (Quali P{podium.iloc[1]['QualifyingPosition']})")
print(f"🥉 P3: {podium.iloc[2]['Driver']} (Quali P{podium.iloc[2]['QualifyingPosition']})")

# Analysis
print("\n" + "=" * 60)
print("📊 RACE ANALYSIS")
print("=" * 60)

# Biggest movers
final_results["PositionChange"] = (
    final_results["QualifyingPosition"] - 
    final_results["PredictedPosition"]
)
movers = final_results.sort_values("PositionChange", ascending=False)

print("\n🚀 Biggest Gainers:")
for idx in range(min(3, len(movers))):
    driver = movers.iloc[idx]
    if driver["PositionChange"] > 0:
        print(f"   {driver['Driver']}: P{int(driver['QualifyingPosition'])} → "
              f"P{int(driver['PredictedPosition'])} "
              f"(+{int(driver['PositionChange'])} positions)")

print("\n📉 Biggest Losers:")
for idx in range(len(movers) - 1, max(len(movers) - 4, -1), -1):
    driver = movers.iloc[idx]
    if driver["PositionChange"] < 0:
        print(f"   {driver['Driver']}: P{int(driver['QualifyingPosition'])} → "
              f"P{int(driver['PredictedPosition'])} "
              f"({int(driver['PositionChange'])} positions)")

# Key factors
print("\n🔑 Key Factors:")
print(f"   • Best Race Pace: {final_results.iloc[0]['Driver']} "
      f"({final_results.iloc[0]['RacePace']:.2f}s)")
print(f"   • Recent Form Boost: VER (US win), LEC (US P3)")
print(f"   • Overtaking Potential: Mexico's long straight favors slipstream")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Race Pace Comparison
ax1 = axes[0, 0]
colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'steelblue' 
          for i in range(len(final_results))]
ax1.barh(final_results["Driver"], final_results["RacePace"], color=colors)
ax1.set_xlabel("Predicted Race Pace (seconds)", fontsize=11)
ax1.set_title("Predicted Average Lap Times", fontsize=12, fontweight='bold')
ax1.invert_xaxis()
ax1.invert_yaxis()

# 2. Qualifying vs Predicted Finish
ax2 = axes[0, 1]
x = range(len(final_results))
ax2.plot(x, final_results["QualifyingPosition"], 'o-', label='Qualifying', linewidth=2, markersize=8)
ax2.plot(x, final_results["PredictedPosition"], 's-', label='Predicted Finish', linewidth=2, markersize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(final_results["Driver"])
ax2.set_ylabel("Position", fontsize=11)
ax2.set_title("Qualifying vs Predicted Race Position", fontsize=12, fontweight='bold')
ax2.legend()
ax2.invert_yaxis()
ax2.grid(alpha=0.3)

# 3. Position Changes
ax3 = axes[1, 0]
change_colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                 for x in final_results["PositionChange"]]
ax3.barh(final_results["Driver"], final_results["PositionChange"], color=change_colors)
ax3.set_xlabel("Position Change", fontsize=11)
ax3.set_title("Predicted Position Changes from Qualifying", fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.invert_yaxis()

# 4. Factor Contributions
ax4 = axes[1, 1]
top3 = final_results.head(3)
factors = ['Race Pace', 'Form Boost', 'Overtaking']
x_pos = np.arange(len(top3))
width = 0.25

pace_normalized = (82 - top3["RacePace"]) * 5  # Normalize for visualization
form_normalized = (1.05 - top3["FormMultiplier"]) * 100
overtake_normalized = -top3["OvertakingFactor"] * 10

ax4.bar(x_pos - width, pace_normalized, width, label='Race Pace', color='steelblue')
ax4.bar(x_pos, form_normalized, width, label='Recent Form', color='orange')
ax4.bar(x_pos + width, overtake_normalized, width, label='Overtaking', color='green')
ax4.set_ylabel('Factor Contribution (normalized)', fontsize=11)
ax4.set_title('Top 3 Drivers - Factor Breakdown', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(top3["Driver"])
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("mexico_2025_race_prediction.png", dpi=300, bbox_inches='tight')
print(f"\n📈 Visualization saved as 'mexico_2025_race_prediction.png'")
print("=" * 60)