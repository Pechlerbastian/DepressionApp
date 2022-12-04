import pandas as pd
import numpy as np
import librosa as lb

class DataParser:

    delimiter = None
    needed_columns = []
    target = []

    def __init__(self, needed_columns, target, delimiter=','):
        self.needed_columns = needed_columns
        self.target = target
        self.delimiter = delimiter
        return

    def load_audio_file(self):
        return

    def parse_labels(self, fold_number=0):
        develop = pd.read_csv('FeedbackExperiment-folds/'+str(fold_number)+'/dev.csv', sep=self.delimiter, usecols=['filename', 'VoiceRating'])
        test = pd.read_csv('FeedbackExperiment-folds/'+str(fold_number)+'/test.csv', sep=self.delimiter, usecols=['filename', 'VoiceRating'])
        train = pd.read_csv('FeedbackExperiment-folds/'+str(fold_number)+'/train.csv', sep=self.delimiter, usecols=['filename', 'VoiceRating'])
        duration_frames = []
        for file_name in train['filename']:
            samples, sample_rate = lb.load('FeedbackExperiment/'+file_name, sr=16000) # TODO use hyperparam
            duration = lb.get_duration(y=samples, sr=sample_rate)
            duration_frames.append(int(duration*16000))
        train.rename(columns={'VoiceRating': 'label'}, inplace=True)
        train['duration_frames'] = duration_frames
        train.to_csv(path_or_buf='FeedbackExperiment-folds/'+str(fold_number)+'/train_adjusted.csv')

        duration_frames = []
        for file_name in develop['filename']:
            samples, sample_rate = lb.load('FeedbackExperiment/'+file_name, sr=16000) # TODO use hyperparam
            duration = lb.get_duration(y=samples, sr=sample_rate)
            duration_frames.append(int(duration*16000))

        develop.rename(columns={'VoiceRating': 'label'}, inplace=True)
        develop['duration_frames'] = duration_frames
        develop.to_csv(path_or_buf='FeedbackExperiment-folds/' + str(fold_number) + '/dev_adjusted.csv')

        duration_frames = []
        for file_name in test['filename']:
            samples, sample_rate = lb.load('FeedbackExperiment/'+file_name, sr=16000) # TODO use hyperparam
            duration = lb.get_duration(y=samples, sr=sample_rate)
            duration_frames.append(int(duration*16000))

        test.rename(columns={'VoiceRating': 'label'}, inplace=True)
        test['duration_frames'] = duration_frames
        test.to_csv(path_or_buf='FeedbackExperiment-folds/' + str(fold_number) + '/test_adjusted.csv')

        return train, develop, test

    def process_data(self):
        return

    def load_data(self):
        labels = pd.read_csv('FeedbackExperiment/labels.csv')
        header = np.asarray(labels.keys())
        indices_parameters = np.asarray([np.where(header == x) for x in self.needed_columns]).flatten()
        index_result = np.asarray([np.where(header == x) for x in self.target]).flatten()
        X = np.asarray(labels.iloc[:, indices_parameters].copy())
        y = np.asarray(labels.iloc[:, index_result].copy())

        return X, y



