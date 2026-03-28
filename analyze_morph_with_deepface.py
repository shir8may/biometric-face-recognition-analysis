from pathlib import Path
import json
from deepface import DeepFace

############# CHANGE HERE: update the paths for img_A and img_B to match your local Data folder #############

# Paths
base_path = Path("D:/BiometricsProject/318172467/Results/frames_W_T_C")
img_A = Path("D:/BiometricsProject/318172467/Data/CASIA-FaceV5(BMP)/318/318_1.bmp")
img_B = Path("D:/BiometricsProject/318172467/Data/CASIA-FaceV5(BMP)/208/208_1.bmp")
img_morph = Path("D:/BiometricsProject/318172467/Results/frames_W_T_C/frame005.png")

# Run comparisons
print(" Comparing Morph vs. Person A...")
res_A = DeepFace.verify(str(img_morph), str(img_A), model_name='VGG-Face')

print(" Comparing Morph vs. Person B...")
res_B = DeepFace.verify(str(img_morph), str(img_B), model_name='VGG-Face')

# Output both
results = {
    "morph_vs_A": res_A,
    "morph_vs_B": res_B
}

print("\n=== VGG-Face Morph Attack Results ===")
print(json.dumps(results, indent=2))

# Save result to file
output_file = base_path / "vggface_morph_attack_results_W_T_C.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n Results saved to: {output_file}")
