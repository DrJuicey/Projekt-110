
import tensorflow as tf                          
from tensorflow import keras
from keras import layers                                               

#Loading Dataset
mnist= tf.keras.datasets.mnist

#Train-Test-Split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing Data
x_train = x_train / 255
x_test = x_test / 255

#Create model
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),                                    #expects 28x28 pixel images with 1 colorchannel (different levels of grey)
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",),  #convolutional layer with 32 filters sized 3x3 and ReLU 
        layers.MaxPooling2D(pool_size=(3, 3)),                              #reduces dimensions of Feature-Maps
        layers.Flatten(),                                                   #converts 2D-Feature-Maps into 1D-vectors for Dense Layers
        layers.Dense(units=64, activation="relu"),                          #Fully connected Layer with ReLU activation
        layers.Dense(units=10, activation="softmax")                        #Output Layer 10 neurons for 10 numerals
    ], 
    name="simple_cnn"
)

model.compile(
    optimizer="adam",                                                       #Algorithm for Optimising wieghts
    loss="sparse_categorical_crossentropy",                                 #loss function
    metrics=["accuracy"]                                                    #evaluation metric: proportion of correctly classified samples
    )

model.summary()

#Fit model
model.fit(x_train, y_train, batch_size= 64, epochs = 10)                    #Optimal batch_size either 32 or 64 as evaluated in lecture

#Evaluate model
print("\nPerformance on test data:")
model.evaluate(x_test, y_test)