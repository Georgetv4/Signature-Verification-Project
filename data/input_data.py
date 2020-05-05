from PIL import Image
import numpy as np


def png_to_arr(path):
    Img = Image.open(path)
    arr = np.asarray(Img, dtype='uint8')
    return arr


img1 = png_to_arr('data_signatures/Sig1.png')
img2 = png_to_arr('data_signatures/Sig2.png')
img3 = png_to_arr('data_signatures/Sig3.png')
img4 = png_to_arr('data_signatures/Sig4.png')
img5 = png_to_arr('data_signatures/Sig5.png')
img6 = png_to_arr('data_signatures/Sig6.png')
img7 = png_to_arr('data_signatures/Sig7.png')
img8 = png_to_arr('data_signatures/Sig8.png')
img9 = png_to_arr('data_signatures/Sig9.png')
img10 = png_to_arr('data_signatures/Sig10.png')

train_data = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
