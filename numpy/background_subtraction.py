import numpy as np
import matplotlib.pyplot as plt
import cv2


def compute_difference(image1, image2):
    difference_rgb = np.abs(image1 - image2)
    difference_compressed = np.sum(difference_rgb, axis = 2) / 3.0
    difference_compressed = difference_compressed.astype('uint8')
    return difference_compressed

def compute_binary_mask(matrix, threshold = 50):

    difference_binary = np.where(matrix >= threshold, 255, 0)
    binary_mask = np.stack((difference_binary,)*3, axis=-1)
    return binary_mask

def replace_background(image_bg1, image_bg2, image_obj):
    difference = compute_difference(image_bg1, image_obj)
    binary_mask = compute_binary_mask(difference)
    output = np.where(binary_mask == 255, image_obj, image_bg2)
    return output

def main():
    image_bg1 = cv2.imread("./AIO/numpy/assets/background1.png")
    image_bg2 = cv2.imread("./AIO/numpy/assets/background2.png")
    image_obj = cv2.imread("./AIO/numpy/assets/object.png")

    image_bg1 = cv2.resize(image_bg1, (640,480))
    image_bg2 = cv2.resize(image_bg2, (640,480))
    image_obj = cv2.resize(image_obj, (640,480))

    # image_bg1 = cv2.cvtColor(image_bg1, cv2.COLOR_BGR2RGB)
    # image_bg2 = cv2.cvtColor(image_bg2, cv2.COLOR_BGR2RGB)
    # image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)

    diff_matrix = compute_binary_mask(compute_difference(image_bg1, image_obj))

    
    output = replace_background(image_bg1, image_bg2, image_obj)
    cv2.imshow("Remove background shit", output)
    cv2.waitKey(0)
main()