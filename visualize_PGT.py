import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
depth = Image.open("data/ADEChallengeData2016/images/training_processed_depth_medium/7001.png").convert('L')
print(np.unique(depth))