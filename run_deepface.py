from deepface import DeepFace
import pandas as pd
import os
import csv

############# CHANGE HERE the paths for pairs_csv and output_csv #############
# Configuration section: set model name and input/output paths
model_name = "Facenet"  #  runing again with- VGG-Face 
pairs_csv = r"D:\BiometricsProject\318172467\Results\pairs.csv"
output_csv = rf"D:\BiometricsProject\318172467\Results\scores_{model_name}.csv"

# Force cache path to D: (to avoid filling C:, since in my computer this drive is full)
os.environ["DEEPFACE_HOME"] = r"D:\BiometricsProject"

# Load the list of image pairs to compare
df = pd.read_csv(pairs_csv)

# Run DeepFace for each pair
results = []

# Iterate over all pairs and compute similarity using DeepFace
for idx, row in df.iterrows():
    img1 = row["img1"]
    img2 = row["img2"]
    label = row["label"]

    try:
        # Run DeepFace verification on the image pair

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=False
        )
        score = result["distance"]  # Cosine distance: lower means more similar
    except Exception as e:
        print(f"[!] Error comparing:\n  {img1}\n  {img2}\n  {e}")
        score = -1 # Mark comparison failure

    results.append([img1, img2, label, score])

    if idx % 50 == 0:
        print(f"Compared {idx} pairs...")

# Save scores to CSV file
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label", "score"])
    writer.writerows(results)

print(f"\n Done! Scores saved to {output_csv}")


