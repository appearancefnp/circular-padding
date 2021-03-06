import tensorflow as tf

from unet import unet
import os

import random

BATCH_SIZE = 6
PANORAMA_WIDTH = 512
PANORAMA_HEIGHT = 256


def preprocess_data(img, mask, train=False):
    img = img / 255.
    mask = mask / 255.
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    if train == True:
        roll = random.randint(1, PANORAMA_WIDTH - 1)
        img = tf.roll(img, roll, axis=2)
        mask = tf.roll(mask, roll, axis=2)

    return (img, mask)


def train_generator(path):
    """Create training example generator flow from directory
    Args:
      flags_obj: absl flag object.
      data_aug_args: Arguments for tf.keras.preprocessing.image ImageDataGenerator
    Yields:
      tf.data.Dataset of (img, mask)
    """

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        directory=path,
        classes=["image"],
        class_mode=None,
        color_mode="rgb",
        target_size=(PANORAMA_HEIGHT, PANORAMA_WIDTH),
        batch_size=BATCH_SIZE,
        save_to_dir=None,
        save_prefix='image',
        seed=1337  # Same seed for image_datagen and mask_datagen.
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=path,
        classes=["label"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(PANORAMA_HEIGHT, PANORAMA_WIDTH),
        batch_size=BATCH_SIZE,
        save_to_dir=None,
        save_prefix='label',
        seed=1337)

    train_gene = (preprocess_data(img, mask, train=True)
                  for img, mask in zip(image_generator, mask_generator))

    return train_gene


def test_generator(path):
    """Create training example generator flow from directory
    Args:
      flags_obj: absl flag object.
      data_aug_args: Arguments for tf.keras.preprocessing.image ImageDataGenerator
    Yields:
      tf.data.Dataset of (img, mask)
    """

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        directory=path,
        classes=["image"],
        class_mode=None,
        color_mode="rgb",
        target_size=(PANORAMA_HEIGHT, PANORAMA_WIDTH),
        batch_size=BATCH_SIZE,
        save_to_dir=None,
        save_prefix='image',
        seed=1337  # Same seed for image_datagen and mask_datagen.
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=path,
        classes=["label"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(PANORAMA_HEIGHT, PANORAMA_WIDTH),
        batch_size=BATCH_SIZE,
        save_to_dir=None,
        save_prefix='label',
        seed=1337)

    test_gene = (preprocess_data(img, mask)
                 for img, mask in zip(image_generator, mask_generator))

    return test_gene


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.BinaryAccuracy()

train_gene = train_generator("/home/karlis/Desktop/dataset/train")
samples = 4648
test_gene = test_generator("/home/karlis/Desktop/dataset/test")

# LOGS1 - no augment, no circular
LOG_DIR = "logs_1_2"
# Logs1_2 augment, no circular

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, update_freq='epoch'),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(LOG_DIR, "best_loss.h5"), monitor='val_loss', save_best_only=True,
                                       save_weights_only=False, mode='auto', save_freq='epoch')
]


model = unet((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3))

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.summary()


history = model.fit_generator(
    train_gene,
    epochs=100,
    steps_per_epoch=(samples // BATCH_SIZE),
    validation_data=test_gene,
    validation_steps=(500 // BATCH_SIZE),
    callbacks=[callbacks])

# https://github.com/minoring/unet/blob/643b7b91aba7ec8d251b42ac79cfb07a1add0205/data.py#L16
