import shutil
import random
import os
import glob

random.seed(42)  # Set a seed for reproducibility

data_dirs = [
    "dataset/output/connective",
    "dataset/output/muscle",
    "dataset/output/epithelial",
    "dataset/output/nervous",
]  # The list of directories with tissue images

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

for folder in data_dirs:
    subfolder_name = os.path.basename(folder)

    train_folder = os.path.join("dataset/training", subfolder_name)
    validation_folder = os.path.join("dataset/validation", subfolder_name)
    test_folder = os.path.join("dataset/testing", subfolder_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = glob.glob(folder + "/*.jpg")
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    validation_size = int(len(images) * validation_ratio)

    train_images = images[:train_size]
    validation_images = images[train_size : train_size + validation_size]
    test_images = images[train_size + validation_size :]

    for img_file in train_images:
        shutil.copy(img_file, train_folder)

    for img_file in validation_images:
        shutil.copy(img_file, validation_folder)

    for img_file in test_images:
        shutil.copy(img_file, test_folder)
