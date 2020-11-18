import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img_file = '../data/test/images/17MAR16163727-S3XS-502717381070_01_P004_0_chunk1_14.npy'

# use np and plt function to show the input data
arr = np.load(img_file)
fig = plt.imshow(arr[:, :, 0])
fig = plt.imshow(arr[:, :, 1])
fig = plt.imshow(arr[:, :, 2])
fig = plt.imshow(arr[:, :, 3])
plt.show()

mask_file = '0.npy'

arr = np.load(mask_file)
fig2 = plt.imshow(arr.squeeze())
plt.show()
