import pandas as pd
import matplotlib.pyplot as plt
import os

############# CHANGE HERE the paths for thresholds_csv and output_png  #############
# Configuration (set the model name and path to input/output files)
model_name = "OpenCV"
thresholds_csv = rf"D:\BiometricsProject\318172467\Results\thresholds_{model_name}.csv"
output_png = rf"D:\BiometricsProject\318172467\Results\roc_{model_name}.png"

# Load the threshold evaluation CSV
df = pd.read_csv(thresholds_csv)

# Compute True Positive Rate (TPR = 1 - FNMR)
df["TPR"] = 1 - df["FNMR"]  

# Plot ROC curve: FMR (x-axis) vs. TPR (y-axis)
plt.figure(figsize=(8, 6))
plt.plot(df["FMR"], df["TPR"], marker='o', label=f'ROC: {model_name}')
plt.plot([0, 1], [0, 1], '--', color='gray')  # random guess line
plt.xlabel("False Match Rate (FMR)")
plt.ylabel("True Match Rate (1 - FNMR)")
plt.title(f"ROC Curve - {model_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as PNG
plt.savefig(output_png)
plt.show()

print(f"\n ROC plot saved to: {output_png}")
