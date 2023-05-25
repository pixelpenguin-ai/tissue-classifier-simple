import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_data(data_directory, num_classes):
    x_data = []
    y_data = []

    for tissue_type_index in range(num_classes):
        tissue_type_folder = os.path.join(data_directory, str(tissue_type_index))
        for file in os.listdir(tissue_type_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(tissue_type_folder, file)
                image = Image.open(image_path).resize((256, 256))
                x_data.append(np.array(image))
                y_data.append(tissue_type_index)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


data_directory = "dataset/output/"
num_classes = 4

x_data, y_data = load_data(data_directory, num_classes)

# Normalize image data
x_data = x_data / 255.0

# One-hot encode labels
y_data = to_categorical(y_data, num_classes)

# Split dataset into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42
)  # 70% train, 30% temp
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)  # 15% val, 15% test


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


input_shape = (
    256,
    256,
    3,
)  # Assuming that your images are resized to 256x256 pixels and have 3 color channels
num_classes = 4  # Replace this with the number of tissue types in your dataset

model = create_model(input_shape, num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs = 10  # You can start with 10 epochs and adjust this number according to your model's performance
batch_size = 32

history = model.fit(
    train_dataset.batch(batch_size),
    epochs=epochs,
    validation_data=val_dataset.batch(batch_size),
)

model.save("tissue_classifier.h5")

test_loss, test_acc = model.evaluate(test_dataset.batch(batch_size))
print("Test accuracy:", test_acc)
