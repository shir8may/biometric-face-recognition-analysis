import pandas as pd
import matplotlib.pyplot as plt



# Configuration section: set path and model names/colors for plotting
results_path = r"D:\BiometricsProject\318172467\Results" ############# CHANGE HERE #############
models = ["Facenet", "VGG-Face", "OpenCV"]
colors = ["blue", "green", "red"]

plt.figure(figsize=(8, 6))

# Loop through all models and plot their ROC curves
for model_name, color in zip(models, colors):
    file_path = fr"{results_path}\thresholds_{model_name}.csv"
    df = pd.read_csv(file_path)

    # Plot ROC: X = FMR, Y = 1 - FNMR (True Match Rate)
    plt.plot(df["FMR"], 1 - df["FNMR"], marker='o', label=model_name, color=color)

# Add reference line for random guess
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
plt.xlabel("False Match Rate (FMR)")
plt.ylabel("True Match Rate (1 - FNMR)")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save combined ROC plot to file
output_path = fr"{results_path}\roc_multi_curve.png"
plt.savefig(output_path)
print(f" Saved to: {output_path}")
plt.show()
