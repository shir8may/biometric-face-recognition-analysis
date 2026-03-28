import os
import random
import csv

# Script to generate pairs of images: genuine (same person) and imposter (different people)

############# CHANGE HERE the paths for data_dir and output_csv  #############
# CONFIGURATION
# Define paths and parameters for sampling
data_dir = r"D:\BiometricsProject\318172467\Data\CASIA-FaceV5(BMP)"
output_csv = r"D:\BiometricsProject\318172467\Results\pairs.csv"
num_subjects = 50
images_per_subject = 5
seed = 42  # for reproducibility

random.seed(seed)

# STEP 1: Load 50 random subject from the dataset
subject_ids = sorted(os.listdir(data_dir))
chosen_ids = random.sample(subject_ids, num_subjects)

# STEP 2: Build image paths for each subject
subjects = {}  # key: subject_id, value: list of 5 image paths
for sid in chosen_ids:
    subject_path = os.path.join(data_dir, sid)
    images = [os.path.join(subject_path, f) for f in sorted(os.listdir(subject_path)) if f.endswith(".bmp")]
    if len(images) >= images_per_subject:
        subjects[sid] = images[:images_per_subject]

# STEP 3: Generate genuine pairs (same subject)
genuine_pairs = []
for sid, imgs in subjects.items():
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            genuine_pairs.append((imgs[i], imgs[j], 1))

# STEP 4: Generate random imposter pairs (random different subjects)
subject_keys = list(subjects.keys())
imposter_pairs = []

while len(imposter_pairs) < len(genuine_pairs):
    sid1, sid2 = random.sample(subject_keys, 2)
    img1 = random.choice(subjects[sid1])
    img2 = random.choice(subjects[sid2])
    imposter_pairs.append((img1, img2, 0))

# STEP 5: Save all pairs to CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label"])  # header
    writer.writerows(genuine_pairs + imposter_pairs)

print(f" Generated {len(genuine_pairs)} genuine and {len(imposter_pairs)} imposter pairs.")
print(f" Saved to: {output_csv}")
