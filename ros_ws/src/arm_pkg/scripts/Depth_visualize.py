import os
import numpy as np
from PIL import Image

depth:np.ndarray = np.load(os.path.join(os.path.dirname(__file__), "Depth.npy"))

normal = (depth - np.mean(depth))/np.std(depth)

brightness = 127 - 128*normal

brightness = np.where(brightness >= 255, 255, np.where(brightness < 0, 0, brightness)).astype(np.uint8)

Image.fromarray(brightness).save(os.path.join(os.path.dirname(__file__), "Depth.png"))