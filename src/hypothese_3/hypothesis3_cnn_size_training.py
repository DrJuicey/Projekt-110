import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────
mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = x_train_full / 255.0
x_test       = x_test       / 255.0

# ─────────────────────────────────────────────
# 2. Helper: build the same CNN every time
# ─────────────────────────────────────────────
def build_model(name="simple_cnn"):
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dense(units=64, activation="relu"),
            layers.Dense(units=10, activation="softmax"),
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# 3. Four experimental conditions
# ─────────────────────────────────────────────

# Condition A – 100 % of training data (baseline)
x_train_100 = x_train_full
y_train_100 = y_train_full

# Condition B – 50 % of training data (random subset, reproducible)
rng = np.random.default_rng(seed=42)
idx_50 = rng.choice(len(x_train_full), size=len(x_train_full) // 2, replace=False)
x_train_50 = x_train_full[idx_50]
y_train_50 = y_train_full[idx_50]

# Condition C – 10 % of training data
idx_10 = rng.choice(len(x_train_full), size=len(x_train_full) // 10, replace=False)
x_train_10 = x_train_full[idx_10]
y_train_10 = y_train_full[idx_10]

# Condition D – 100 % + Gaussian noise (σ = 0.1, clipped to [0, 1])
noise_sigma = 0.1
x_train_noisy = np.clip(
    x_train_full + rng.normal(loc=0.0, scale=noise_sigma, size=x_train_full.shape),
    0.0,
    1.0,
)
y_train_noisy = y_train_full


# ─────────────────────────────────────────────
# 4. Training loop – same hyper-parameters for all
# ─────────────────────────────────────────────
BATCH_SIZE     = 64
EPOCHS         = 10
VALIDATION_SPLIT = 0.2

conditions = [
    ("100_data",       x_train_100,   y_train_100),
    ("50_data",        x_train_50,    y_train_50),
    ("10_data",        x_train_10,    y_train_10),
    ("100_noisy",      x_train_noisy, y_train_noisy),
]

histories = {}
results   = {}

for label, x_tr, y_tr in conditions:
    print(f"\n{'='*60}")
    print(f"  Condition: {label}  |  Training samples: {len(x_tr)}")
    print(f"{'='*60}")

    model = build_model(name=label)
    history = model.fit(
        x_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )
    histories[label] = history.history

    print(f"\nPerformance on test data ({label}):")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    results[label] = {"test_loss": test_loss, "test_acc": test_acc}


# ─────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────
colors = ["steelblue", "darkorange", "seagreen", "tomato"]
labels_pretty = {
    "100_data":  "100  data (baseline)",
    "50_data":   "50  data",
    "10_data":   "10  data",
    "100_noisy": "100  + Gaussian noise",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hypothese 3 – Effect of Training Data on CNN Performance", fontsize=14, fontweight="bold")

for (label, _x, _y), color in zip(conditions, colors):
    h = histories[label]
    ep = range(1, len(h["accuracy"]) + 1)
    axes[0].plot(ep, h["val_accuracy"], label=labels_pretty[label], color=color)
    axes[1].plot(ep, h["val_loss"],     label=labels_pretty[label], color=color)

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()


# ─── Bar chart: Test accuracy per condition ───
fig2, ax = plt.subplots(figsize=(8, 5))
bar_labels = [labels_pretty[l] for l in results]
bar_values = [results[l]["test_acc"] for l in results]

bars = ax.bar(bar_labels, bar_values, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Test Accuracy")
ax.set_title("Hypothese 3 – Test Accuracy per Data Condition")
ax.grid(axis="y", alpha=0.3)
plt.xticks(wrap=True)
plt.tight_layout()
plt.show()
plt.close()


# ─── Summary printout ───────────────────────
print("\n\n" + "="*60)
print("  SUMMARY – Hypothese 3")
print("="*60)
print(f"{'Condition':<28} {'Test Acc':>10} {'Test Loss':>12}")
print("-"*52)
for label in results:
    r = results[label]
    print(f"{labels_pretty[label]:<28} {r['test_acc']:>10.4f} {r['test_loss']:>12.4f}")
print("="*60)
print("\nPlots saved to:")
print("  hypothese3_curves.png  (Validation Accuracy & Loss over Epochs)")
print("  hypothese3_bar.png     (Test Accuracy per Condition)")