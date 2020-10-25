import keras.utils
import numpy as np
import pandas as pa

from Source.config import Data


class Driver:
    # Create driver in order to load dataset and
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = Driver.__load_data()
        print('Dataset Loaded Successfully!')

    # Main function for loading dataset
    @staticmethod
    def __load_data():
        def load_file(path):
            data = pa.read_csv(filepath_or_buffer=path, header=None, delim_whitespace=True)
            return data.values

        # Collect dataset filenames
        def generate_data_path(base_address, group):
            filenames = list()
            filepath = base_address + group + '/Inertial Signals/'
            for name in Data.FEATURE_TYPE:
                for dimension in Data.DIMENSION:
                    filenames += [filepath + name + dimension + group + '.txt']

            return filenames

        # Combine inertial signals and create a suitable data
        def combine_features(base_address, group):
            filenames = generate_data_path(base_address, group)
            combined_data = list()
            for name in filenames:
                data = load_file(name)
                combined_data.append(data)
            # Use dstack to create a 3D data like [samples, time_steps, features]
            combined_data = np.dstack(combined_data)
            return combined_data

        # Load x_train, y_train, x_test and y_test data
        def load_dataset(base_address=Data.BASE_ADDRESS):
            x_train = combine_features(base_address, 'train')
            y_train = load_file(base_address + 'train/' + 'y_train.txt')

            x_test = combine_features(base_address, 'test')
            y_test = load_file(base_address + 'test/' + 'y_test.txt')

            # Zero-Offset
            y_test = y_test - 1
            y_train = y_train - 1

            # One-Hot encoding
            y_test = keras.utils.to_categorical(y_test)
            y_train = keras.utils.to_categorical(y_train)

            return x_train, y_train, x_test, y_test

        return load_dataset()


if __name__ == '__main__':
    d = Driver()
    print(d.x_train.shape[0])
