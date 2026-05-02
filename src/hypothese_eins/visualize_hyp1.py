import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

def load_results_from_csv(csv_path: Path):
    """
    Load experiment results from CSV file.

    Expected columns:
    conv_layers, test_accuracy, test_loss
    """
    results = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append({
                "conv_layers": int(row["conv_layers"]),
                "test_acc": float(row["test_accuracy"]),
                "test_loss": float(row["test_loss"])
            })

    # Sort by number of convolution layers
    results.sort(key=lambda x: x["conv_layers"])
    return results


def find_latest_plot_folder(base_plot_dir: Path):
    """
    Find the newest timestamp folder in results/plots/hypothese_eins.
    """
    if not base_plot_dir.exists():
        return None

    subdirs = [p for p in base_plot_dir.iterdir() if p.is_dir()]

    if not subdirs:
        return None

    # Sort folders by name (timestamp format works well for sorting)
    subdirs.sort()
    return subdirs[-1]


# ------------------------------------------------------------
# 2. Plot: test accuracy comparison
# ------------------------------------------------------------

def plot_test_accuracy(results, save_path: Path):
    conv_layers = [r["conv_layers"] for r in results]
    test_acc = [r["test_acc"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(conv_layers, test_acc, marker="o", linewidth=2)
    plt.xticks(conv_layers)
    plt.xlabel("Number of Convolution Layers")
    plt.ylabel("Test Accuracy")
    plt.title("Hypothesis 1: Test Accuracy vs. CNN Depth")
    plt.grid(True)

    for x, y in zip(conv_layers, test_acc):
        plt.text(x, y + 0.0005, f"{y:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 3. Plot: test loss comparison
# ------------------------------------------------------------

def plot_test_loss(results, save_path: Path):
    conv_layers = [r["conv_layers"] for r in results]
    test_loss = [r["test_loss"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(conv_layers, test_loss, marker="o", linewidth=2)
    plt.xticks(conv_layers)
    plt.xlabel("Number of Convolution Layers")
    plt.ylabel("Test Loss")
    plt.title("Hypothesis 1: Test Loss vs. CNN Depth")
    plt.grid(True)

    for x, y in zip(conv_layers, test_loss):
        plt.text(x, y + 0.001, f"{y:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 4. Plot: combined grouped bar chart
#    One chart, two metric groups:
#    - Test Accuracy
#    - Test Loss
#    Colors = number of conv layers
# ------------------------------------------------------------

def plot_combined_bar_chart(results, save_path: Path):
    # Extract values
    acc_values = [r["test_acc"] for r in results]
    loss_values = [r["test_loss"] for r in results]
    layer_labels = [f"{r['conv_layers']} Conv" for r in results]

    x = np.array([0, 1])   # 0 = Accuracy, 1 = Loss
    width = 0.22

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(
        x - width,
        [acc_values[0], loss_values[0]],
        width=width,
        label=layer_labels[0]
    )
    bars2 = plt.bar(
        x,
        [acc_values[1], loss_values[1]],
        width=width,
        label=layer_labels[1]
    )
    bars3 = plt.bar(
        x + width,
        [acc_values[2], loss_values[2]],
        width=width,
        label=layer_labels[2]
    )

    plt.xticks(x, ["Test Accuracy", "Test Loss"])
    plt.ylabel("Value")
    plt.title("Hypothesis 1: Combined Comparison of Test Accuracy and Test Loss")
    plt.legend(title="CNN Depth")
    plt.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        plt.bar_label(bars, fmt="%.4f", padding=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 5. Create overview image from the existing training plots
#    This collects the already saved accuracy/loss plots from
#    the newest timestamp folder.
# ------------------------------------------------------------

def create_training_plot_overview(latest_plot_dir: Path, save_path: Path):
    """
    Combine the existing per-model training plots into one overview image.

    Expected files:
    accuracy_1_conv_layers.png
    accuracy_2_conv_layers.png
    accuracy_3_conv_layers.png
    loss_1_conv_layers.png
    loss_2_conv_layers.png
    loss_3_conv_layers.png
    """
    if latest_plot_dir is None or not latest_plot_dir.exists():
        print("No plot folder found for training plot overview.")
        return

    image_files = [
        latest_plot_dir / "accuracy_1_conv_layers.png",
        latest_plot_dir / "accuracy_2_conv_layers.png",
        latest_plot_dir / "accuracy_3_conv_layers.png",
        latest_plot_dir / "loss_1_conv_layers.png",
        latest_plot_dir / "loss_2_conv_layers.png",
        latest_plot_dir / "loss_3_conv_layers.png",
    ]

    existing_files = [img for img in image_files if img.exists()]

    if len(existing_files) != 6:
        print("Not all expected training plot files were found. Skipping overview image.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Hypothesis 1: Training and Validation Curves Overview", fontsize=16)

    for ax, img_path in zip(axes.flat, image_files):
        image = mpimg.imread(img_path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(img_path.stem.replace("_", " "))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 6. Main run function
# ------------------------------------------------------------

def run():
    """
    Main function for Hypothesis 1 visualization.
    Can be called from main.py.
    """
    print("\n--- Visualizing Hypothesis 1 ---")

    # Project root:
    # visualize_hyp1.py is expected in: src/hypothese_eins/
    project_root = Path(__file__).resolve().parents[2]

    metrics_csv = project_root / "results" / "metrics" / "hyp1_cnn_depth_results.csv"
    plot_base_dir = project_root / "results" / "plots" / "hypothese_eins"
    summary_dir = plot_base_dir / "summary"

    summary_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        print(f"CSV file not found: {metrics_csv}")
        print("Please run hyp1_cnn_depth.py first.")
        return

    # Load result data
    results = load_results_from_csv(metrics_csv)

    # Create summary comparison plots
    plot_test_accuracy(
        results,
        summary_dir / "hyp1_test_accuracy_comparison.png"
    )

    plot_test_loss(
        results,
        summary_dir / "hyp1_test_loss_comparison.png"
    )

    plot_combined_bar_chart(
        results,
        summary_dir / "hyp1_combined_bar_chart.png"
    )

    # Find newest timestamp plot folder and create combined overview
    latest_plot_dir = find_latest_plot_folder(plot_base_dir)

    if latest_plot_dir is not None:
        print(f"Latest training plot folder found: {latest_plot_dir}")
        create_training_plot_overview(
            latest_plot_dir,
            summary_dir / "hyp1_training_plot_overview.png"
        )
    else:
        print("No timestamp plot folder found. Skipping training overview.")

    print("\nHypothesis 1 visualization finished.")
    print(f"All summary plots are stored in: {summary_dir}")


# ------------------------------------------------------------
# 7. Execute directly
# ------------------------------------------------------------

if __name__ == "__main__":
    run()