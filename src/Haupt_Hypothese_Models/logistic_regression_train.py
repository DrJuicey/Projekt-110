import tensorflow as tf
from sklearn.linear_model import  LogisticRegression                    # Logistic Regression
from sklearn.model_selection import GridSearchCV                        # Gridsearch
import joblib

#Loading Dataset
mnist= tf.keras.datasets.mnist

#Train-Test-Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing Data
x_train = x_train / 255
x_test = x_test / 255

#reshape for logistic regression
x_train = x_train.reshape(x_train.shape[0], -1)                     
x_test  = x_test.reshape(x_test.shape[0], -1)                       #Logistic Regression model only takes 2-dimensional vectors

#evaluation of best hyperparams
grid = GridSearchCV(                                                
    estimator= LogisticRegression(max_iter = 3000),                 #Logistic regression is used
    param_grid= {
        "C": [0.01, 0.1, 1, 10],                                    #different values for c are tested
        "solver": ["lbfgs", "saga"],                                #different solvers are tested
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
joblib.dump(model, "Results/logistic_regression_model.pkl")     
joblib.dump(results, "Results/logistic_regression_results.pkl")