from PIL import Image
import random
import os

folder = "/home/tepk2924/tepk2924Works/myGraduationProject/texture_dataset/valid_texture"

width, height = 500, 500
for i in range(25):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    image = Image.new("RGB", (width, height), color)
    image.save(os.path.join(folder, f"single_color_{i:03d}.jpg"))