import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load and prepare the MNIST dataset
# ------------------------------------------------------------

# Load dataset (handwritten digits 0–9)
mnist = keras.datasets.mnist

# Split into training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
# This helps the model to train faster and more stable
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
# Original shape: (28, 28)
# New shape: (28, 28, 1)
# CNNs expect a channel dimension (grayscale = 1 channel)
x_train = x_train[..., None]
x_test = x_test[..., None]


# ------------------------------------------------------------
# 2. Function to create CNN model
#    Only the number of convolutional layers changes
# ------------------------------------------------------------
def create_cnn_model(num_conv_layers: int) -> keras.Model:
    """
    Creates a CNN model where only the number of Conv2D layers varies.

    Parameters:
        num_conv_layers (int): number of convolution layers

    Returns:
        keras.Model: compiled CNN model
    """

    # Define input layer
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")

    # Start building the model
    x = inputs

    # Predefined number of filters for each layer
    # (deeper layers usually have more filters)
    filter_list = [32, 64, 128, 128]

    # Loop to add convolutional layers
    for i in range(num_conv_layers):

        # Select number of filters for this layer
        filters = filter_list[i]

        # Add convolutional layer
        # - kernel_size (3x3): small window to scan the image
        # - padding="same": keeps image size constant
        # - activation="relu": introduces non-linearity
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name=f"conv_{i+1}"
        )(x)

        # Add pooling layer to reduce spatial size
        # This helps reduce computation and extract important features
        if i == 0 or i == num_conv_layers - 1:
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f"pool_{i+1}"
            )(x)

    # Flatten feature maps into a vector
    # Needed before feeding into Dense layers
    x = keras.layers.Flatten(name="flatten")(x)

    # Fully connected layer
    # Combines extracted features
    x = keras.layers.Dense(
        64,
        activation="relu",
        name="dense_1"
    )(x)

    # Output layer
    # 10 neurons = digits 0–9
    # Softmax gives probabilities
    outputs = keras.layers.Dense(
        10,
        activation="softmax",
        name="output"
    )(x)

    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"cnn_{num_conv_layers}_conv_layers"
    )

    # Compile the model
    model.compile(
        optimizer="adam",                              # optimization algorithm
        loss="sparse_categorical_crossentropy",        # loss function for classification
        metrics=["accuracy"]                           # performance metric
    )

    return model


# ------------------------------------------------------------
# 3. Function to train and evaluate the model
# ------------------------------------------------------------
def train_and_evaluate(num_conv_layers: int,
                       epochs: int = 10,
                       batch_size: int = 64):
    """
    Trains a CNN model and evaluates it on test data.

    Returns:
        model, history, test_loss, test_accuracy
    """

    print(f"\n--- Training model with {num_conv_layers} conv layer(s) ---")

    # Clear previous models from memory (important when running multiple experiments)
    tf.keras.backend.clear_session()

    # Create model
    model = create_cnn_model(num_conv_layers)

    # Print model structure
    model.summary()

    # Train the model
    # validation_split=0.2 means:
    # 20% of training data is used for validation
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate model on test data (never seen during training)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return model, history, test_loss, test_acc


# ------------------------------------------------------------
# 4. Run experiments (Hypothesis 1)
# ------------------------------------------------------------

# Dictionary to store results
results = {}

# Test different numbers of convolution layers
for num_layers in [1, 2, 3]:

    model, history, test_loss, test_acc = train_and_evaluate(num_layers)

    # Store results for later comparison
    results[num_layers] = {
        "history": history.history,
        "test_loss": test_loss,
        "test_acc": test_acc
    }


# ------------------------------------------------------------
# 5. Print comparison results
# ------------------------------------------------------------
print("\n--- Model Comparison ---")

for num_layers, result in results.items():
    print(
        f"{num_layers} Conv Layers -> "
        f"Test Accuracy: {result['test_acc']:.4f}, "
        f"Test Loss: {result['test_loss']:.4f}"
    )


# ------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------

# Plot accuracy and loss curves for each model
for num_layers, result in results.items():

    history = result["history"]

    # Accuracy plot
    plt.figure()
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Accuracy - {num_layers} Conv Layers")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Loss - {num_layers} Conv Layers")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()