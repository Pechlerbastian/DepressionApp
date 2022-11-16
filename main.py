#import deepspectrumlite
#from deepspectrumlite import data_adapter
#from deepspectrumlite import amplitude_to_db
import numpy as np
import tensorflow
from deepspectrumlite import TransferBaseModel,AugmentableModel
from DataParser import DataParser
from deepspectrumlite import DataPipeline
from deepspectrumlite import HyperParameterList

if __name__ == '__main__':
    parser = DataParser(['filename', 'PatientGroup', 'SubjectID', 'TrialGroup', 'TrialValence', 'ReappraisalID', 'Try', 'ProvidedFeedback'],
                        ['VoiceRating'])

    files, targets = parser.load_labels()
    X = np.asarray(files[0])
    y = targets
    pipeline = DataPipeline(data_classes=['r'], enable_gpu=False, enable_augmentation=False, run_id=1, hparams=HyperParameterList)
    pipeline.get_filenames()
    pipeline.generate_spectrogram()
   # X_train, y_train, X_val, y_val = tensorflow.keras.train_test_split(X, y, 0.2)
    model = AugmentableModel()
#    model.compile()
    model.compile(optimizer='Adam')
    model.fit(X, y, batch_size=32, validation_split=0.2)

    model.summary()
