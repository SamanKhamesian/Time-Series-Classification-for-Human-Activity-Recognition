import os

import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential

from Source.config import Model
from Source.driver import Driver

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run():
    dataset = Driver()
    n_output = dataset.y_train.shape[1]
    n_features = dataset.x_train.shape[2]

    # CNN model will read subsequences of the main sequence in as blocks, so we should split each window
    x_train = dataset.x_train.reshape(dataset.x_train.shape[0], Model.N_STEPS, Model.N_LENGTH, n_features)
    x_test = dataset.x_test.reshape(dataset.x_test.shape[0], Model.N_STEPS, Model.N_LENGTH, n_features)

    model = CNNLSTMModel(n_output, n_features)
    model.fit(x_train=x_train, y_train=dataset.y_train)

    accuracy = model.evaluate(x_test=x_test, y_test=dataset.y_test)
    print('Accuracy = %{0:.4f}'.format(accuracy * 100))
    print(model.model.summary())


class CNNLSTMModel:
    def __init__(self, n_output, n_features):
        self.model = Sequential()
        # Add 3 CNN models with 32 filters and window size of 4
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=4, activation='relu'), input_shape=(None, Model.N_LENGTH, n_features)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=4, activation='relu')))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=4, activation='relu')))
        self.model.add(TimeDistributed(Dropout(Model.DROPOUT_RATE)))
        # Add max-pooling layer
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        # The extracted features are then flattened and provided to the LSTM model to read
        self.model.add(TimeDistributed(Flatten()))
        # Add LSTM Model
        self.model.add(LSTM(Model.N_NODES))
        # Use Dropout layer to reduce overfitting of the model to the training data
        self.model.add(Dropout(Model.DROPOUT_RATE))
        # Add one hidden layer with 100 default nodes with relu activation function
        self.model.add(Dense(Model.N_NODES, activation='relu'))
        # Add output layer with 6 (Total number of classes) nodes with softmax activation function
        self.model.add(Dense(n_output, activation='softmax'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs=25, batch_size=64, verbose=0):
        self.model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, x_test, batch_size=64):
        return self.model.predict(x=x_test, batch_size=batch_size)

    def evaluate(self, x_test, y_test, batch_size=64, verbose=0):
        _, accuracy = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        return accuracy


if __name__ == '__main__':
    run()
