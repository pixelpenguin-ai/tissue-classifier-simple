import cv2
import os
import glob

input_folder = "dataset/input/connective"
output_folder = "dataset/output/connective"
target_size = (256, 256)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_filenames = glob.glob(
    input_folder + "/*.jpg"
)  # Adjust the pattern to match your image file format

for file in image_filenames:
    img = cv2.imread(file)
    resized_img = cv2.resize(img, target_size)
    output_filename = os.path.join(output_folder, os.path.basename(file))
    cv2.imwrite(output_filename, resized_img)
