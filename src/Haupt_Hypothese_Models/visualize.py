import tensorflow as tf
from sklearn.linear_model import  LogisticRegression                    # Logistic Regression
from sklearn.ensemble import RandomForestClassifier                     # random forests
from sklearn.model_selection import GridSearchCV                        # Gridsearch
from sklearn.svm import SVC                                             # Support Vector Machines
import joblib                                                           # Loading models + results
import matplotlib.pyplot as plt                                         # Plotting results
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

#Loading Dataset
mnist= tf.keras.datasets.mnist

#Train-Test-Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing Data
x_train = x_train / 255
x_test = x_test / 255

#reshape for SVM
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)                     
x_test_reshaped  = x_test.reshape(x_test.shape[0], -1)                  # classical machine learning models only take 2-dimensional vectors

#Loading models + results
model_LR = joblib.load("src/Haupt_Hypothese_Models/Results/logistic_regression_model.pkl")
results_LR = joblib.load("src/Haupt_Hypothese_Models/Results/logistic_regression_results.pkl")

model_SVM = joblib.load("src/Haupt_Hypothese_Models/Results/svm_model.pkl")
results_SVM = joblib.load("src/Haupt_Hypothese_Models/Results/svm_results.pkl")

model_RF = joblib.load("src/Haupt_Hypothese_Models/Results/random_forest_model.pkl")
results_RF = joblib.load("src/Haupt_Hypothese_Models/Results/random_forest_results.pkl")

model_CNN = joblib.load("src/Haupt_Hypothese_Models/Results/cnn_model.pkl")
results_CNN = joblib.load("src/Haupt_Hypothese_Models/Results/cnn_results.pkl")

LR_train_score = results_LR["train_score"]
LR_test_score = results_LR["test_score"]
LR_val_acc = results_LR["val_acc"]

SVM_train_score = results_SVM["train_score"]
SVM_test_score = results_SVM["test_score"]
SVM_val_acc = results_SVM["val_acc"]

RF_train_score = results_RF["train_score"]
RF_test_score = results_RF["test_score"]
RF_val_acc = results_RF["val_acc"]

CNN_train_score = results_CNN["train_score"]
CNN_test_score = results_CNN["test_score"]
CNN_val_acc = results_CNN["val_acc"]

#evaluate performance on train data
print("\nPerformance on train data:")
print("\nLogistic Regression:", LR_train_score)
print("\nRandom Forest:", RF_train_score)
print("\nSupport Vector Machine:", SVM_train_score)
print("\nConvolutional Neural Network:", CNN_train_score)

#evaluate performance on test data
print("\nPerformance on test data:")
print("\nLogistic Regression:", LR_test_score)
print("\nRandom Forest:", RF_test_score)
print("\nSupport Vector Machine:", SVM_test_score)
print("\nConvolutional Neural Network:", CNN_test_score)

#evaluate validation accuracy
print("\nValidierungsgenauigkeit:")
print("\nLogistic Regression:", LR_val_acc)
print("\nRandom Forest:", RF_val_acc)
print("\nSupport Vector Machine:", SVM_val_acc)
print("\nConvolutional Neural Network:", CNN_val_acc)

###Plotting Model Accuracies
models = ["Logistic Regression", "Random Forest", "SVM", "CNN"]

train_performance = [
    LR_train_score,
    RF_train_score,
    SVM_train_score,
    CNN_train_score
]

test_performance = [
    LR_test_score,
    RF_test_score,
    SVM_test_score,
    CNN_test_score
]

val_accuracies = [
    LR_val_acc,
    RF_val_acc,
    SVM_val_acc,
    CNN_val_acc
]

fig, axs = plt.subplots(3, 2, figsize=(12, 6))                          # 3 rows for the 3 metrics, 2 cols for normal visualisation and zoomed in variant

fig.suptitle("Comparison of Classification Accuracy Across Models")

fig.set_layout_engine("tight")

bar_colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

# plotting train performance
bars_1 = axs[0, 0].bar(models, train_performance, color=bar_colors)
axs[0, 0].set_title("Accuracy for train data")
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].bar_label(bars_1, fmt="%.3f")
axs[0, 0].set_ylim(0, 1.2)

bars_2 = axs[0, 1].bar(models, train_performance, color=bar_colors)
axs[0, 1].set_title("Accuracy for train data (zoomed: 0.90-1.02)")
axs[0, 1].bar_label(bars_2, fmt="%.3f")
axs[0, 1].set_ylim(0.90, 1.02)

# plotting test performance
bars_3 = axs[1, 0].bar(models, test_performance, color=bar_colors)
axs[1, 0].set_title("Accuracy for test data")
axs[1, 0].set_ylabel("Accuracy")
axs[1, 0].bar_label(bars_3, fmt="%.3f")
axs[1, 0].set_ylim(0, 1.2)

bars_4 = axs[1, 1].bar(models, test_performance, color=bar_colors)
axs[1, 1].set_title("Accuracy for test data (zoomed: 0.90-1.02)")
axs[1, 1].bar_label(bars_4, fmt="%.3f")
axs[1, 1].set_ylim(0.9, 1.02)

# plotting validation accuracy
bars_5 = axs[2, 0].bar(models, val_accuracies, color=bar_colors)
axs[2, 0].set_title("validation accuracy")
axs[2, 0].set_xlabel("Models")
axs[2, 0].set_ylabel("Accuracy")
axs[2, 0].bar_label(bars_5, fmt="%.3f")
axs[2, 0].set_ylim(0, 1.2)

bars_6 = axs[2, 1].bar(models, val_accuracies, color=bar_colors)
axs[2, 1].set_title("validation accuracy (zoomed: 0.90-1.02)")
axs[2, 1].set_xlabel("Models")
axs[2, 1].bar_label(bars_6, fmt="%.3f")
axs[2, 1].set_ylim(0.9, 1.02)

fig.savefig("Results/barcharts.png", bbox_inches="tight")

plt.show()

###Plotting falsely categorized numbers 
pred_LR_test = model_LR.predict(x_test_reshaped)
pred_RF_test = model_RF.predict(x_test_reshaped)
pred_SVM_test = model_SVM.predict(x_test_reshaped)
pred_CNN_test = model_CNN.predict(x_test)
pred_CNN_test = pred_CNN_test.argmax(axis=1)

# computing error rate per digit per model
def compute_errors_per_digit(y_true, pred):
    errors = []
    for digit in range(10):
        mask = (y_true == digit)
        errors.append(np.mean(pred[mask] != y_true[mask]))
    return errors 

errors_LR_test = compute_errors_per_digit(y_test, pred_LR_test)
errors_RF_test = compute_errors_per_digit(y_test, pred_RF_test)
errors_SVM_test = compute_errors_per_digit(y_test, pred_SVM_test)
errors_CNN_test = compute_errors_per_digit(y_test, pred_CNN_test)

fig, axs = plt.subplots(2, 2, figsize=(12, 6))                          # 4 plots, 1 for every model

fig.suptitle("Comparison of falsely classified Digits accross Models")

fig.set_layout_engine("tight")

all_errors = (
    errors_LR_test +
    errors_RF_test +
    errors_SVM_test +
    errors_CNN_test
)

max_error = max(all_errors)

bars_1 = axs[0, 0].bar(range(10), errors_LR_test)
axs[0, 0].bar_label(bars_1, fmt="%.3f")
axs[0, 0].set_title("Errors Logistic Regression on test data")
axs[0, 0].set_ylabel("Error Rate per Digit")

bars_2 = axs[0, 1].bar(range(10), errors_RF_test)
axs[0, 1].bar_label(bars_2, fmt="%.3f")
axs[0, 1].set_title("Errors Random Forest on test data")

bars_3 = axs[1, 0].bar(range(10), errors_SVM_test)
axs[1, 0].bar_label(bars_3, fmt="%.3f")
axs[1, 0].set_title("Errors SVM on test data")
axs[1, 0].set_ylabel("Error Rate per Digit")
axs[1, 0].set_xlabel("Digits")

bars_4 = axs[1, 1].bar(range(10), errors_CNN_test)
axs[1, 1].bar_label(bars_4, fmt="%.3f")
axs[1, 1].set_title("Errors CNN on test data")
axs[1, 1].set_xlabel("Digits")

# scaling y axis and making both axes more readable by adding ticks
for i in range(2):
    axs[i, 0].set_ylim(0, max_error)
    axs[i, 0].set_xticks(range(10))
    axs[i, 0].set_yticks(np.linspace(0, max_error * 1.1, 5))
    axs[i, 1].set_ylim(0, max_error)
    axs[i,1].set_xticks(range(10))
    axs[i, 1].set_yticks(np.linspace(0, max_error * 1.1, 5))

fig.savefig("Results/falsely_classified_numbers.png", bbox_inches="tight")

plt.show()

###Confusion matrices
ConfusionMatrixDisplay.from_predictions(y_test, pred_LR_test)
plt.savefig("Results/Confusionmatrix_LR.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred_RF_test)
plt.savefig("Results/Confusionmatrix_RF.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred_SVM_test)
plt.savefig("Results/Confusionmatrix_SVM.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred_CNN_test)
plt.savefig("Results/Confusionmatrix_CNN.png")
plt.show()