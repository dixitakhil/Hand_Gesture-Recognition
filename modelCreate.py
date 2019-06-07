import numpy as np
import matplotlib.pyplot as plt
from keras.layers import  Dense, Flatten, Conv2D
from keras.layers import  MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd


def keras_model(im_x, im_y):
    number_of_classes_available = 15    #Here input the number of classes that you want it is from 0 to class-1
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(im_x, im_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(number_of_classes_available, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelPath = "model.h5"
    checkpoint1 = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def run():   #run this to make model
    data = pd.read_csv("data.csv")   #Read CSV data
    dataset = np.array(data)      #Taking in array
    np.random.shuffle(dataset)    #Shuffling the data as to get enough data for training and testing an the first coloumn represents class
    X = dataset
    Y = dataset
    X = X[:, 1:2501] # As we have a 50*50 image so 50*50=2500
    Y = Y[:, 0]    #Here we take the first coloumn of all the rows that shows class

    X_train = X[0:17000, :] #Here in row part mention how many samples to take here we are taking 17000 samples
    X_train = X_train / 255.
    X_test = X[17000:19601, :]   #As we have 14 datasets  so 14*1400=19600 Taking a 80% to 20% train and test data
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:17000, :]
    Y_train = Y_train.T
    Y_test = Y[17000:19601, :]
    Y_test = Y_test.T

    image_x = 50
    image_y = 50

    train_y = np_utils.to_categorical(Y_train)
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)
    history=model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=20, batch_size=32,
              callbacks=callbacks_list)
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("Error: %.2f%%" % (100 - scores[1] * 100))
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('model.h5')  #Save the model formed


run()

