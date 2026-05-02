import joblib
import tensorflow as tf
from sklearn.svm import SVC                                         # Support Vector Machines
from sklearn.model_selection import GridSearchCV                    # Gridsearch

#Loading Dataset
mnist= tf.keras.datasets.mnist

#Train-Test-Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing Data
x_train = x_train / 255
x_test = x_test / 255

#reshape for SVM
x_train = x_train.reshape(x_train.shape[0], -1)                     
x_test  = x_test.reshape(x_test.shape[0], -1)                       #SVM only takes 2-dimensional vectors

#evaluation of best hyperparams
grid = GridSearchCV(                                                
    estimator= SVC(),                                               #SVM is used
    param_grid= {
        "C" : [0.1, 1, 10],                                         #different Values for the regularization parameter are tested
        "kernel": ["linear", "rbf"],                                #different kernels are tested
        "gamma" : ["scale", 0.01, 0.001]                            #different kernel coefficients are tested
    },
    scoring="accuracy",                                             #performance is measured by accuracy
    n_jobs = -1,                                                    #all processors are used for faster evaluation
    cv = 5,                                                         #Cross validation so its less random and gives a more stable result
)

grid.fit(x_train, y_train)

model = grid.best_estimator_

results =  {
    "train_score" : model.score(x_train, y_train),
    "test_score" : model.score(x_test, y_test),
    "val_acc" : grid.best_score_
}

#best model and results get saved for reuse in visualisation file
joblib.dump(model, "Results/svm_model.pkl")
joblib.dump(results, "Results/svm_results.pkl")