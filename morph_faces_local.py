
import sys
import os
from PIL import Image
import numpy as np
import cv2

############# CHANGE HERE the path for the #############
# Add path to face_morpher repository
sys.path.append(r"D:\BiometricsProject\318172467\face_morpher-dlib")

from facemorpher.morpher import morph
from facemorpher import videoer, locator, aligner

# frame_M_T_W : 079_0 + 207_0
# frame_M_T_M : 001_1 + 149_0
# frame_W_T_W : 136_1 + 063_1
# frame_W_T_C : 318_1 + 208_1
# frame_M_T_C : 247_0 + 208_1

############# CHANGE HERE the paths for img1_path and img2_path to your true path of the Data #############
# Input image paths
img1_path = r"D:\BiometricsProject\318172467\Data\CASIA-FaceV5(BMP)\318\318_1.bmp"
img2_path = r"D:\BiometricsProject\318172467\Data\CASIA-FaceV5(BMP)\208\208_1.bmp"


# Output folder and file
output_dir = r"D:\BiometricsProject\318172467\Results" ############# CHANGE HERE #############
os.makedirs(output_dir, exist_ok=True)
output_img_path = os.path.join(output_dir, "morph_result.png")

# Load images using OpenCV
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)


if img1 is None or img2 is None:
    print("[ERROR] One of the images failed to load.")
    exit()

# Detect landmarks
points1 = locator.face_points(img1)
points2 = locator.face_points(img2)


if len(points1) == 0 or len(points2) == 0:
    print("[ERROR] No face detected in one of the images.")
    exit()

# Align faces
aligned1, points1 = aligner.resize_align(img1, points1, size=(600, 500))
aligned2, points2 = aligner.resize_align(img2, points2, size=(600, 500))

# Dummy video object
video = videoer.Video(None, 1, 500, 600)  # args: filename, fps, width, height


# Run morph
morph(
    aligned1, points1,
    aligned2, points2,
    video,
    width=500,
    height=600,
    num_frames=11,
    fps=1,
    out_frames=os.path.join(output_dir, "frame_W_T_C"),
    out_video=None,
    plot=False,
    background='black'
)

print(f"\n Morphing succeeded! Resulting frames saved to:\n{os.path.join(output_dir, 'frame###.png')}")
