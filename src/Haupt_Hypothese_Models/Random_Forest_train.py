import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier                 # random forests
from sklearn.model_selection import GridSearchCV                    # Gridsearch
import joblib

#Loading Dataset
mnist= tf.keras.datasets.mnist

#Train-Test-Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing Data
x_train = x_train / 255
x_test = x_test / 255

#reshape for random forest
x_train = x_train.reshape(x_train.shape[0], -1)                     
x_test  = x_test.reshape(x_test.shape[0], -1)                       #random forest only takes 2-dimensional vectors

#evaluation of best hyperparams
grid = GridSearchCV(                                                
    estimator= RandomForestClassifier(),                            #Randomforestclassifier is used
    param_grid= {
        "n_estimators" : [10, 100, 200],                            #number of trees in the forest 
        "max_depth" : [None, 10, 20],                               #max tree depth; None -> tree grows to full
        "max_features" : ["sqrt", "log2"]                           #number of features to consider when looking for the best split
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
joblib.dump(model, "Results/random_forest_model.pkl")
joblib.dump(results, "Results/random_forest_results.pkl")