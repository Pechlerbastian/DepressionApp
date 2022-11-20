#import deepspectrumlite
#from deepspectrumlite import data_adapter
#from deepspectrumlite import amplitude_to_db
import json
import os

import numpy as np
import tensorflow
from deepspectrumlite import TransferBaseModel ,AugmentableModel
from DataParser import DataParser
from deepspectrumlite import DataPipeline
from deepspectrumlite import HyperParameterList
from deepspectrumlite.lib.data.parser import ComParEParser

if __name__ == '__main__':
    parser = DataParser(['filename', 'PatientGroup', 'SubjectID', 'TrialGroup', 'TrialValence', 'ReappraisalID', 'Try', 'ProvidedFeedback'],
                        ['VoiceRating'])
    model_dir = 'model/'
    files, targets = parser.load_data()
    X = np.asarray(files[0])
    y = targets
    f = open('hyper_param.json')
    data = json.load(f)
    f.close()

    data_classes = data

    if data_classes is None:
        raise ValueError('no data classes defined')

    hyp_params = HyperParameterList(config_file_name='hyper_param.json')
    parser = DataParser(['filename', 'PatientGroup', 'SubjectID', 'TrialGroup', 'TrialValence', 'ReappraisalID', 'Try', 'ProvidedFeedback'],
                        ['VoiceRating'])
    for i in range(10):
        tensorboard_dir = hyp_params.get_values(iteration_no=i)['tb_experiment']

        log_dir = os.path.join(model_dir, 'logs', tensorboard_dir)

        run_identifier = hyp_params.get_values(iteration_no=i)['tb_run_id'] + '_config_' + str(i)

        run_log_dir = os.path.join(log_dir, run_identifier)
        model_dir = os.path.join(model_dir, 'models', tensorboard_dir, run_identifier)

        train_data_pipeline = DataPipeline(name='train_data_set', data_classes=data_classes,
                                           enable_gpu=True, verbose=False, enable_augmentation=False,
                                           hparams=hyp_params.get_values(iteration_no=i), run_id=i)
        train_data = parser.parse_labels()
        train_data_pipeline.set_data(train_data)
        train_data_pipeline.set_filename_prepend(prepend_filename_str="FeedbackExperiment/Control/Kontrollgruppe/AM0767/Negative")
        train_data_pipeline.preprocess()
        train_data_pipeline.up_sample()
        train_dataset = train_data_pipeline.pipeline(cache=True)

        print(train_data_pipeline.get_filenames())

   # X_train, y_train, X_val, y_val = tensorflow.keras.train_test_split(X, y, 0.2)
        model = TransferBaseModel(hyp_params.get_hparams(),
                    train_data_pipeline.get_model_input_shape(),
                    run_dir=run_log_dir,
                    data_classes=data_classes,
                    use_ram=True,
                    run_id=i,
                    verbose=False)
        #    model.compile()
        model.compile(optimizer='Adam')
        model.fit(X, y, batch_size=32, validation_split=0.2)

        model.summary()
