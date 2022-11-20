import pandas
import numpy as np

class DataParser:

    needed_columns = []
    target = []

    def __init__(self, needed_columns, target):
        self.needed_columns = needed_columns
        self.target = target
        return

    def load_data(self):
        return

    def load_audio_file(self):
        return

    def parse_labels(self):
        labels = pandas.read_csv('FeedbackExperiment/labels.csv')
        header = np.asarray(labels.keys())
        index_result = np.asarray([np.where(header == x) for x in self.target]).flatten()
        y = np.asarray(labels.iloc[:, index_result].copy())


        return y

    def load_data(self):
        labels = pandas.read_csv('FeedbackExperiment/labels.csv')
        header = np.asarray(labels.keys())
        indices_parameters = np.asarray([np.where(header == x) for x in self.needed_columns]).flatten()
        index_result = np.asarray([np.where(header == x) for x in self.target]).flatten()
        X = np.asarray(labels.iloc[:, indices_parameters].copy())
        y = np.asarray(labels.iloc[:, index_result].copy())

        return X, y



