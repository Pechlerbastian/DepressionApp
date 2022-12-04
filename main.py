#import deepspectrumlite
#from deepspectrumlite import data_adapter
#from deepspectrumlite import amplitude_to_db
import json
import logging
import os
import shutil
from tensorboard.plugins.hparams import api as hp
import numpy as np
import tensorflow
from deepspectrumlite import TransferBaseModel ,AugmentableModel
from DataParser import DataParser
from deepspectrumlite import DataPipeline
from deepspectrumlite import HyperParameterList
from deepspectrumlite.lib.data.parser import ComParEParser

METRIC_ACCURACY = 'accuracy'
METRIC_PRECISION = 'precision'
METRIC_RECALL = 'recall'
METRIC_F_SCORE = 'f1_score'
METRIC_MAE = 'mae'
METRIC_RMSE = 'rmse'
METRIC_MSE = 'mse'
METRIC_LOSS = 'loss'

if __name__ == '__main__':
    log = logging.getLogger(__name__)
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
    tensorboard_initialised = False
    hyp_params = HyperParameterList(config_file_name='hyper_param.json')
    parser = DataParser(['filename', 'PatientGroup', 'SubjectID', 'TrialGroup', 'TrialValence', 'ReappraisalID', 'Try', 'ProvidedFeedback'],
                        ['label'])
    max_iterations = hyp_params.get_max_iteration()
    log.info('Loaded hyperparameter configuration.')
    log.info("Recognised combinations of settings: " + str(max_iterations) + "")
    enable_cache = False
    data_dir = 'FeedbackExperiment'
    verbose = False
    for i in range(max_iterations):
        hparam_values = hyp_params.get_values(iteration_no=i)
        hparam_values_tensorboard = hyp_params.get_values_tensorboard(iteration_no=i)
        tensorboard_dir = hparam_values['tb_experiment']

        log_dir = os.path.join(model_dir, 'logs', tensorboard_dir)

        run_identifier = hyp_params.get_values(iteration_no=i)['tb_run_id'] + '_config_' + str(i)

        run_log_dir = os.path.join(log_dir, run_identifier)
        model_dir = os.path.join(model_dir, 'models', tensorboard_dir, run_identifier)
        # delete old log
        if os.path.isdir(run_log_dir):
            shutil.rmtree(run_log_dir)

        if not tensorboard_initialised:
            # create tensorboard
            with tensorflow.summary.create_file_writer(log_dir).as_default():
                hp.hparams_config(
                    hparams=hyp_params.get_hparams(),
                    metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy'),
                             hp.Metric(METRIC_PRECISION, display_name='precision'),
                             hp.Metric(METRIC_RECALL, display_name='unweighted recall'),
                             hp.Metric(METRIC_F_SCORE, display_name='f1 score'),
                             hp.Metric(METRIC_MAE, display_name='mae'),
                             hp.Metric(METRIC_RMSE, display_name='rmse')
                             ],
                )
                tensorboard_initialised = True
        # Use a label file parser to load data
        label_parser_key = hparam_values['label_parser']

        train_data, devel_data, test_data = parser.parse_labels()


        np.random.seed(0)
        tensorflow.compat.v1.set_random_seed(0)

        log.info('--- Starting trial: %s' % run_identifier)
        log.info({h.name: hparam_values_tensorboard[h] for h in hparam_values_tensorboard})

        log.info("Load data pipeline ...")
        ########### TRAIN DATA ###########
        train_data_pipeline = DataPipeline(name='train_data_set', data_classes=data_classes,
                                           enable_gpu=True, verbose=True, enable_augmentation=False,
                                           hparams=hparam_values, run_id=i)
        train_data_pipeline.set_data(train_data)
        train_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        train_data_pipeline.preprocess()
        train_data_pipeline.up_sample()
        train_dataset = train_data_pipeline.pipeline(cache=enable_cache)

        ########### DEVEL DATA ###########
        devel_data_pipeline = DataPipeline(name='devel_data_set', data_classes=data_classes,
                                           enable_gpu=True, verbose=True, enable_augmentation=False,
                                           hparams=hparam_values, run_id=i)
        devel_data_pipeline.set_data(devel_data)
        devel_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        devel_dataset = devel_data_pipeline.pipeline(cache=enable_cache, shuffle=False, drop_remainder=False)

        ########### TEST DATA ###########
        test_data_pipeline = DataPipeline(name='test_data_set', data_classes=data_classes,
                                          enable_gpu=True, verbose=True, enable_augmentation=False,
                                          hparams=hparam_values, run_id=i)
        test_data_pipeline.set_data(test_data)
        test_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        test_dataset = test_data_pipeline.pipeline(cache=enable_cache, shuffle=False, drop_remainder=False)

        log.info("All data pipelines have been successfully loaded.")
        log.info("Caching in memory is: " + str(enable_cache))

        model_name = hparam_values['model_name']

        available_ai_models = {
            'TransferBaseModel': TransferBaseModel
        }

        if model_name in available_ai_models:
            model = available_ai_models[model_name](hyp_params,
                                                    train_data_pipeline.get_model_input_shape(),
                                                    run_dir=run_log_dir,
                                                    data_classes=data_classes,
                                                    use_ram=True,
                                                    run_id=i,
                                                    verbose=verbose)

            model.run(train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      devel_dataset=devel_dataset,
                      save_model=True,
                      save_dir=model_dir)
        else:
            ValueError("Unknown model name: " + model_name)
