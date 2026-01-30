import os
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU,
    MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset paths
DATA_TRAIN = r"C:\Users\KyawNyi\Desktop\ML\datasets\train"
DATA_VALID = r"C:\Users\KyawNyi\Desktop\ML\datasets\valid"
DATA_TEST  = r"C:\Users\KyawNyi\Desktop\ML\datasets\test"


# Image generators
def build_data_loaders(train_folder, valid_folder):
    aug_train = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    aug_valid = ImageDataGenerator(rescale=1.0 / 255.0)

    train_loader = aug_train.flow_from_directory(
        train_folder,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    valid_loader = aug_valid.flow_from_directory(
        valid_folder,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    return train_loader, valid_loader


# CNN
def create_cnn_model(input_dim=(128, 128, 3), total_classes=2):
    weight_decay = regularizers.l2(1e-4)
    net = Sequential()

    #Block 1
    net.add(Conv2D(32, (3, 3), padding="same",
                   kernel_regularizer=weight_decay,
                   input_shape=input_dim))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Conv2D(32, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Dropout(0.3))

    #Block 2
    net.add(Conv2D(64, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Conv2D(64, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Dropout(0.3))

    #Block 3
    net.add(Conv2D(128, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Conv2D(128, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Dropout(0.4))

    #Block 4
    net.add(Conv2D(256, (3, 3), padding="same",
                   kernel_regularizer=weight_decay))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Dropout(0.4))

    #Classifier
    net.add(GlobalAveragePooling2D())
    net.add(Dense(256, activation="relu",
                  kernel_regularizer=weight_decay))
    net.add(Dropout(0.4))
    net.add(Dense(total_classes, activation="softmax"))

    return net


#Data preparation
train_loader, valid_loader = build_data_loaders(DATA_TRAIN, DATA_VALID)

label_ids = np.unique(train_loader.classes)
bal_weights = compute_class_weight(
    class_weight="balanced",
    classes=label_ids,
    y=train_loader.classes
)
weight_map = dict(zip(label_ids, bal_weights))

print("Computed Class Weights:", weight_map)
print("Class Mapping:", train_loader.class_indices)


#Model compilation
cnn_model = create_cnn_model()

cnn_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)


#Training configuration
stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.6,
    patience=2
)

saver = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)


#Model training
with tf.device("/GPU:0"):
    training_log = cnn_model.fit(
        train_loader,
        validation_data=valid_loader,
        epochs=30,
        class_weight=weight_map,
        callbacks=[stopper, lr_scheduler, saver],
        verbose=1
    )


cnn_model.save("final_model.keras")

#Test evaluation
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATA_TEST,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = cnn_model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")


#Training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_log.history["accuracy"], label="train")
plt.plot(training_log.history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(training_log.history["loss"], label="train")
plt.plot(training_log.history["val_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#Confusion matrix
val_output = cnn_model.predict(valid_loader, verbose=0)
predicted_ids = np.argmax(val_output, axis=1)
true_ids = valid_loader.classes
labels = list(valid_loader.class_indices.keys())

conf_mat = confusion_matrix(true_ids, predicted_ids)


def draw_confusion_matrix(matrix, labels, normalized=True):
    if normalized:
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalized else "d"
    threshold = matrix.max() / 2.0

    for r, c in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(
            c, r,
            format(matrix[r, c], fmt),
            ha="center",
            color="white" if matrix[r, c] > threshold else "black"
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


draw_confusion_matrix(conf_mat, labels)
