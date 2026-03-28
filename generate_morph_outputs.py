import os
from PIL import Image
import imageio.v2 as imageio

############# CHANGE HERE the path for frames_folder #############
# Configuration 
frames_folder = r"D:\BiometricsProject\318172467\Results\frames_W_T_C"
output_grid_path = os.path.join(frames_folder, "morph_grid_3x3.png")
output_gif_path = os.path.join(frames_folder, "morph_animation_9frames.gif")

# Load frame images (frame001.png to frame009.png)
frame_filenames = [f"frame00{i}.png" for i in range(1, 10)]  # 1 to 9 inclusive
frame_paths = [os.path.join(frames_folder, fname) for fname in frame_filenames]

# Create a 3x3 grid image
images = [Image.open(path) for path in frame_paths]
w, h = images[0].size
grid = Image.new("RGB", (w * 3, h * 3))  # 3 columns × 3 rows

for idx, img in enumerate(images):
    row = idx // 3
    col = idx % 3
    grid.paste(img, (col * w, row * h))

grid.save(output_grid_path)
print(f" Grid image saved to: {output_grid_path}")

# Create animated GIF
gif_frames = [imageio.imread(path) for path in frame_paths]
imageio.mimsave(output_gif_path, gif_frames, duration=0.5)
print(f" GIF animation saved to: {output_gif_path}")
