
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from model_cnn import cnn_base, cnn_dropout, cnn_l2, cnn_aug




# 1. data loading and preprocessing
def load_mnist_from_local_files():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "MNIST", "raw")

    def read_images(filepath):
        with open(filepath, "rb") as f:
            _magic = struct.unpack(">I", f.read(4))[0]
            num_images = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            images = f.read(num_images * rows * cols)
            return tf.reshape(tf.io.decode_raw(images, tf.uint8), (num_images, rows, cols)).numpy()

    def read_labels(filepath):
        with open(filepath, "rb") as f:
            _magic = struct.unpack(">I", f.read(4))[0]
            num_labels = struct.unpack(">I", f.read(4))[0]
            labels = f.read(num_labels)
            return tf.io.decode_raw(labels, tf.uint8).numpy()

    x_train = read_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
    y_train = read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    x_test = read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    y_test = read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    return (x_train, y_train), (x_test, y_test)


# Load MNIST dataset 
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
except Exception:
    (x_train, y_train), (x_test, y_test) = load_mnist_from_local_files()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension for CNN input (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]



# 2. data augmentation generator 
def get_datagen():
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )


datagen = get_datagen()
datagen.fit(x_train)



# 3. compile and train models
def train_model(model, use_augmentation=False):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Evaluate untrained model on test accuracy before training
    _, untrained_test_acc = model.evaluate(x_test, y_test, verbose=0)   

    if use_augmentation:
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=64),
            epochs=10,
            validation_data=(x_test, y_test),
            verbose=2
        )
    else:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=10,
            batch_size=64,
            verbose=2
        )

    return history, float(untrained_test_acc) 



# 4. define all 4 models 
models = {
    "Base": cnn_base(),
    "Dropout": cnn_dropout(),
    "L2": cnn_l2(),
    "Aug": cnn_aug()
}



# 5. train all models and store accuracies
train_accs = {}
test_accs = {}
untrained_test_accs = {}

for name, model in models.items():
    print(f"\n--- Training {name} model ---")

    history, untrained_test_acc = train_model(
        model,
        use_augmentation=(name == "Aug")
    )

    # Save final accuracy (last epoch)
    train_accs[name] = float(history.history["accuracy"][-1])
    test_accs[name] = float(history.history["val_accuracy"][-1])
    untrained_test_accs[name] = untrained_test_acc



# 6. calculate overfitting (train - test accuracy)
overfitting = {
    name: train_accs[name] - test_accs[name]
    for name in models
}


# 7. visualization: train vs test accuracy 
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(8, 5))

labels = list(models.keys())
x = range(len(labels))
bar_width = 0.35

plt.bar(x, [train_accs[l] for l in labels],
        width=bar_width, label="Train Accuracy")

plt.bar([i + bar_width for i in x],
        [test_accs[l] for l in labels],
        width=bar_width, label="Test Accuracy")

plt.xlabel("Model Variant")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.xticks([i + bar_width / 2 for i in x], labels)
plt.grid(axis="y", alpha=0.25)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
plt.close()



# 8. visualization: overfitting / generalization gap
plt.figure(figsize=(8, 5))

plt.bar(labels, [overfitting[l] for l in labels])
plt.axhline(0.0, color="black", linewidth=1)

plt.xlabel("Model Variant")
plt.ylabel("Train - Test Accuracy")
plt.title("Overfitting / Generalization Gap")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "generalization_gap.png"))
plt.close()


# 9. visualization: untrained vs trained test accuracy
plt.figure(figsize=(8, 5))
plt.bar(x, [untrained_test_accs[l] for l in labels],
        width=bar_width, label="Before Training")
plt.bar([i + bar_width for i in x], [test_accs[l] for l in labels],
        width=bar_width, label="After Training")

plt.xlabel("Model Variant")
plt.ylabel("Test Accuracy")
plt.title("Untrained vs Trained Test Accuracy")
plt.xticks([i + bar_width / 2 for i in x], labels)
plt.grid(axis="y", alpha=0.25)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(results_dir, "before_after_test_accuracy.png"))
plt.close()



# 9. print final results
print("\nFINAL RESULTS")
for name in models:
    print(
        f"{name}: "
        f"Train={train_accs[name]:.4f}, "
        f"Test={test_accs[name]:.4f}, "
        f"Overfitting={overfitting[name]:.4f}, "
        f"Before={untrained_test_accs[name]:.4f}"
    )
