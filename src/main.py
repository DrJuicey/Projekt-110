import subprocess
import sys
from pathlib import Path


# ------------------------------------------------------------
# Project path helper
# ------------------------------------------------------------
def get_project_root() -> Path:
    """
    Returns the project root folder.

    If this file is located at:
        Projekt-101/src/main.py

    then parents[1] points to:
        Projekt-101/
    """
    return Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Helper function to run existing Python scripts
# ------------------------------------------------------------
def run_script(script_path: Path, timeout: int | None = None) -> bool:
    """
    Runs a Python script as a separate process.

    This is useful for executing scripts from other team members
    without changing their internal code.

    Parameters:
        script_path (Path):
            Path to the Python script that should be executed.

        timeout (int | None):
            Optional maximum runtime in seconds.
            If None, no timeout is used.

    Returns:
        bool:
            True if the script finished successfully.
            False if it failed, timed out, or was interrupted.
    """

    project_root = get_project_root()

    if not script_path.exists():
        print(f"\nScript not found: {script_path}")
        return False

    print("\n" + "-" * 60)
    print(f"Running script: {script_path.name}")
    print("-" * 60)

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True,
            timeout=timeout
        )

        print("\n" + "-" * 60)
        print(f"Finished script: {script_path.name}")
        print("-" * 60)

        return True

    except subprocess.TimeoutExpired:
        print(f"\nScript timed out: {script_path.name}")
        return False

    except subprocess.CalledProcessError:
        print(f"\nScript failed: {script_path.name}")
        return False

    except KeyboardInterrupt:
        print(f"\nScript interrupted manually: {script_path.name}")
        return False


# ------------------------------------------------------------
# Hypothesis 1
# ------------------------------------------------------------
def run_hypothesis_1_training():
    """
    Runs the training script for hypothesis 1.

    Hypothesis 1 compares CNN models with different numbers
    of convolutional layers.
    """

    project_root = get_project_root()
    script = project_root / "src" / "hypothese_eins" / "hyp1_cnn_depth.py"

    run_script(script)


def run_hypothesis_1_visualization():
    """
    Runs the visualization script for hypothesis 1.
    """

    project_root = get_project_root()
    script = project_root / "src" / "hypothese_eins" / "visualize_hyp1.py"

    run_script(script)



# ------------------------------------------------------------
# Hypothesis 2
# ------------------------------------------------------------
def run_hypothesis_2_training():
    """
    Runs the training script for hypothesis 2.

    Hypothesis 2 compares regularization/training strategies:
    Base CNN, Dropout, L2 regularization, and data augmentation.
    """

    project_root = get_project_root()
    script = project_root / "src" / "hypothese_2" / "train.py"

    if not script.exists():
        print(f"\nScript not found: {script}")
        return

    run_script(script)



# ------------------------------------------------------------
# Hypothesis 3
# ------------------------------------------------------------
def run_hypothesis_3_training():
    """
    Runs the training script for hypothesis 3.

    Hypothesis 3 compares CNN performance under different
    training data conditions.
    """

    project_root = get_project_root()
    script = project_root / "src" / "hypothese_3" / "hypothesis3_cnn_size_training.py"

    run_script(script)


# ------------------------------------------------------------
# Main hypothesis: model comparison
# ------------------------------------------------------------
def get_main_hypothesis_scripts() -> dict[str, Path]:
    """
    Returns all script paths for the main hypothesis.

    The main hypothesis compares different model types:
    Logistic Regression, Random Forest, SVM, and CNN.
    """

    project_root = get_project_root()
    scripts_dir = project_root / "src" / "Haupt_Hypothese_Models"

    return {
        "hcnn": scripts_dir / "CNN_train.py",
        "hrf": scripts_dir / "Random_Forest_train.py",
        "hlr": scripts_dir / "logistic_regression_train.py",
        "hsvm": scripts_dir / "SVM_train.py",
    }


def run_main_hypothesis_model(model_key: str):
    """
    Runs one selected training script of the main hypothesis.

    Parameters:
        model_key (str):
            hcnn -> CNN
            hrf  -> Random Forest
            hlr  -> Logistic Regression
            hsvm -> SVM
    """

    project_root = get_project_root()

    # Some scripts from the main hypothesis save to "Results/"
    # with capital R. This folder is created here to avoid path errors.
    (project_root / "Results").mkdir(exist_ok=True)

    scripts = get_main_hypothesis_scripts()

    script = scripts.get(model_key)

    if script is None:
        print(f"\nUnknown model key: {model_key}")
        return

    run_script(script)


def run_main_hypothesis_training():
    """
    Runs all main hypothesis training scripts.

    The order starts with CNN because it usually finishes faster
    than SVM or Logistic Regression with GridSearch.
    """

    project_root = get_project_root()

    # Compatibility folder for scripts that save to "Results/"
    (project_root / "Results").mkdir(exist_ok=True)

    scripts = get_main_hypothesis_scripts()

    # Recommended execution order:
    # CNN first, then Random Forest, Logistic Regression, SVM.
    execution_order = ["hcnn", "hrf", "hlr", "hsvm"]

    for key in execution_order:
        script = scripts[key]

        success = run_script(script)

        if not success:
            print("\nStopping main hypothesis training because one script failed or was interrupted.")
            break


def run_main_hypothesis_visualization():
    """
    Runs the visualization script for the main hypothesis.

    This should usually be executed after the main hypothesis
    model scripts have created their result files.
    """

    project_root = get_project_root()
    script = project_root / "src" / "Haupt_Hypothese_Models" / "visualize.py"

    run_script(script)


# ------------------------------------------------------------
# Menu
# ------------------------------------------------------------
def print_menu():
    """
    Prints the interactive experiment menu.
    """

    print("\n" + "=" * 50)
    print("CNN Project Experiment Menu")
    print("=" * 50)

    print("\nHypothesis 1:")
    print("1      -> Run Hypothesis 1 training")
    print("1v     -> Visualize Hypothesis 1")

    print("\nHypothesis 2:")
    print("2      -> Run Hypothesis 2 training")

    print("\nHypothesis 3:")
    print("3      -> Run Hypothesis 3 training")

    print("\nMain hypothesis:")
    print("hcnn   -> Run main hypothesis CNN")
    print("hrf    -> Run main hypothesis Random Forest")
    print("hlr    -> Run main hypothesis Logistic Regression")
    print("hsvm   -> Run main hypothesis SVM")
    print("h      -> Run all main hypothesis trainings")
    print("hv     -> Visualize main hypothesis")

    print("\nCombined:")
    print("all    -> Run all currently available trainings")
    print("allv   -> Run all currently available visualizations")

    print("\nExit:")
    print("q      -> Quit")

    print("=" * 50)


def main():
    """
    Interactive main function.

    This function is the central entry point for the project.
    It does not contain the experiment logic itself.
    It only starts the corresponding scripts.
    """

    while True:
        print_menu()

        choice = input("Your choice: ").strip().lower()

        # ------------------------------------------------------------
        # Quit
        # ------------------------------------------------------------
        if choice == "q":
            print("\nProgram stopped.")
            break

        # ------------------------------------------------------------
        # Hypothesis 1
        # ------------------------------------------------------------
        elif choice == "1":
            run_hypothesis_1_training()

        elif choice == "1v":
            run_hypothesis_1_visualization()

        # ------------------------------------------------------------
        # Hypothesis 2
        # ------------------------------------------------------------

        elif choice == "2":
            run_hypothesis_2_training()

        # ------------------------------------------------------------
        # Hypothesis 3
        # ------------------------------------------------------------
        elif choice == "3":
            run_hypothesis_3_training()

        # ------------------------------------------------------------
        # Main hypothesis individual models
        # ------------------------------------------------------------
        elif choice in ["hcnn", "hrf", "hlr", "hsvm"]:
            run_main_hypothesis_model(choice)

        # ------------------------------------------------------------
        # Main hypothesis all trainings
        # ------------------------------------------------------------
        elif choice == "h":
            run_main_hypothesis_training()

        # ------------------------------------------------------------
        # Main hypothesis visualization
        # ------------------------------------------------------------
        elif choice == "hv":
            run_main_hypothesis_visualization()

        # ------------------------------------------------------------
        # Combined training runs
        # ------------------------------------------------------------
        elif choice == "all":
            run_hypothesis_1_training()
            run_hypothesis_3_training()
            run_main_hypothesis_training()

        # ------------------------------------------------------------
        # Combined visualizations
        # ------------------------------------------------------------
        elif choice == "allv":
            run_hypothesis_1_visualization()
            run_main_hypothesis_visualization()

        # ------------------------------------------------------------
        # Invalid input
        # ------------------------------------------------------------
        else:
            print("\nInvalid input. Please choose one of the menu options.")

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()