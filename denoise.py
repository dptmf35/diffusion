import cv2
import numpy as np
import os

def remove_noise(image_path, output_path, kernel_size=(5,5), sigmaX=0):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'The image at {image_path} could not be found')
    
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
    cv2.imwrite(output_path, blurred_image)

image_path = '/data/hyebin/mai_dataset/ffhq64_0.1/00000.png'
output_path = 'result.png'
remove_noise(image_path, output_path)
# output_path = '/data/hye'
# noise_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')])

# for file in noise_files:
#     remove_noise(file, )