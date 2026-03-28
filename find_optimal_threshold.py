import pandas as pd
import os
import csv

# Define the list of model names to process
model_names = ["Facenet", "VGG-Face", "OpenCV"]
input_dir = "Results"
output_file = os.path.join(input_dir, "optimal_thresholds.csv")

# List to store the final optimal thresholds for each model
rows = []

# For each model, load its threshold CSV and find the row with the minimal error sum (FMR + FNMR)
for model in model_names:
    file_path = os.path.join(input_dir, f"thresholds_{model}.csv")
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df["error_sum"] = df["FMR"] + df["FNMR"] # Compute total error for each threshold
    best_row = df.loc[df["error_sum"].idxmin()] # Get the threshold with lowest combined error
    
    # Extract optimal values
    threshold = round(best_row["threshold"], 3)
    fmr = round(best_row["FMR"], 3)
    fnmr = round(best_row["FNMR"], 3)

    rows.append([model, threshold, fmr, fnmr])
    print(f"{model}: threshold={threshold}, FMR={fmr}, FNMR={fnmr}")

# Save all optimal thresholds to a single CSV file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Tool", "Optimal Threshold", "FMR", "FNMR"])
    writer.writerows(rows)

print(f"\nSaved to {output_file}")
