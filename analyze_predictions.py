import pandas as pd

############# CHANGE HERE: update the paths for thresholds_path and scores_path #############
# Analyze predictions for a specific model using its optimal threshold
def analyze_predictions(model_name):
    thresholds_path = fr"D:\BiometricsProject\318172467\Results\thresholds_{model_name}.csv"
    scores_path = fr"D:\BiometricsProject\318172467\Results\scores_{model_name}.csv"
    output_csv = fr"D:\BiometricsProject\318172467\Results\prediction_analysis_{model_name}.csv"

    # Load threshold and score data
    thresholds_df = pd.read_csv(thresholds_path)
    scores_df = pd.read_csv(scores_path)

    # Use optimal threshold (minimizing FMR + FNMR) 
    thresholds_df["error_sum"] = thresholds_df["FMR"] + thresholds_df["FNMR"]
    best_row = thresholds_df.loc[thresholds_df["error_sum"].idxmin()]
    threshold = best_row["threshold"]

    # Initialize confusion matrix counters
    tp = fp = tn = fn = 0

    # Iterate through all score rows and classify based on threshold
    for _, row in scores_df.iterrows():
        label = int(row["label"])
        score = float(row["score"])

        if label == 1:
            if score < threshold:
                tp += 1
            else:
                fn += 1
        else:  # label == 0
            if score < threshold:
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn

    # Calculate accuracy and error rate
    accuracy = round((tp + tn) / total * 100, 2)
    error_rate = round((fp + fn) / total * 100, 2)

    # Save evaluation results to CSV
    results = {
        "category": ["True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)", "Accuracy (%)", "Error Rate (%)"],
        "count": [
            f"{tp}/1000",
            f"{fp}/1000",
            f"{tn}/1000",
            f"{fn}/1000",
            accuracy,
            error_rate
        ],
        "percent":  [
            round(tp / total * 100, 2),
            round(fp / total * 100, 2),
            round(tn / total * 100, 2),
            round(fn / total * 100, 2),
            "-", "-"
        ]
    }

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f" Prediction analysis saved to {output_csv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_predictions.py <ModelName>")
        print("Example: python analyze_predictions.py Facenet")
    else:
        model_name = sys.argv[1]
        analyze_predictions(model_name)