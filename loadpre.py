import os
import tensorflow as tf
from PIL import Image
import glob


def load_images(image_paths, labels_path, target_size=(256, 256)):
    images = []
    labels = []

    label_map = {}
    with open(labels_path, "r") as f:
        for line in f:
            (tissue_type, index) = line.strip().split()
            label_map[tissue_type] = int(index)

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        images.append(image_array)

        label = os.path.dirname(image_path).split(os.sep)[-1]
        labels.append(
            tf.keras.utils.to_categorical(label_map[label], num_classes=len(label_map))
        )

    return tf.data.Dataset.from_tensor_slices((images, labels))


def get_image_paths(root_folder):
    all_filetypes = ["jpg", "jpeg", "png"]
    image_paths = []
    for filetype in all_filetypes:
        image_paths.extend(glob.glob(f"{root_folder}/*/*.{filetype}"))
    return image_paths


train_image_paths = get_image_paths("dataset/training")
val_image_paths = get_image_paths("dataset/validation")
test_image_paths = get_image_paths("dataset/testing")

labels_path = "dataset/output/labels.txt"

train_dataset = load_images(train_image_paths, labels_path)
val_dataset = load_images(val_image_paths, labels_path)
test_dataset = load_images(test_image_paths, labels_path)
