from PIL import Image
import numpy as np


def png_to_arr(path):
    Img = Image.open(path)
    arr = np.asarray(Img, dtype='uint8')
    return arr


train_data = []
for i in range(10):
    train_data.append(png_to_arr("data_signatures/Sig%s.png" % (i + 1)))
