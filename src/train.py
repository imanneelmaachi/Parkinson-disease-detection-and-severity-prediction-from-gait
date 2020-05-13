import numpy as np
import argparse
# fix random seed for reproducibility
np.random.seed(2) #2
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import os
import datetime
#from src.data_utils2 import Datas
from src.results import Results,Results_level

from src.algo import multiple_cnn1D, multiple_cnn1D5_level
from src.data_utils import Data

def train( model, datas, lr, log_filename, filename):
    """

    :param model: Initial untrained model
    :param datas:  data object
    :param lr: learning rate
    :param log_filename: filename where the training results will be saved ( for each epoch)
    :param filename: file where the weights will be saved
    :return:  trained model
    """
    X_train = datas.X_train
    y_train = datas.y_train
    X_val = datas.X_val
    y_val = datas.y_val
    logger = CSVLogger(log_filename, separator=',', append=True)
    for i in (np.arange(1,4)*5):  # 10-20    1-10

        checkpointer = ModelCheckpoint(filepath=filename , monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')

        callbacks_list = [checkpointer, early_stopping, logger]

        history = model.fit(np.split(X_train,X_train.shape[2], axis=2), \
                            # history  = model.fit(X_data,\
                            y_train, \
                            verbose=1, \
                            shuffle=True, \
                            epochs= 200,\
                            batch_size=800, \
                            # validation_data=(X_val, y_val),\
                            validation_data=(np.split(X_val, X_val.shape[2], axis=2), y_val), \
                            callbacks=callbacks_list)

        model.load_weights(filename)
        lr =  lr / 2
        rms = optimizers.Nadam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
        return model


def ablation_study(args):
    '''
    Function that performs the ablation study
    :param args:  Input arguments
    :return:
    '''
    features = np.arange(1, 19)
    folder = os.path.join(args.output, args.exp_name  + '_' + datetime.datetime.now().strftime("%m_%d"),
                             datetime.datetime.now().strftime(
                                 "%H_%M"))
    if not os.path.exists(folder):
        os.makedirs(folder)
    for j in range(1, 9):
        exp_name = args.exp_name + str(j)
        subfolder = os.path.join(folder, 'feature_' + str(j) )
        file_result_patients = os.path.join(subfolder,'res_pat.csv')
        file_result_segments = os.path.join(subfolder,'res_seg.csv')
        #filename = subfolder + "weights.hdf5"
        model_file = os.path.join(subfolder, "model.json")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        feature_delete = j
        feature_delete_r = j + 9
        features_i = np.delete(features, [feature_delete, feature_delete_r])
        val_results = Results(file_result_segments, file_result_patients)
        #datas = Datas(args.input_data, 1,  100, features=features_i)
        #datas.load(norm=None)
        datas = Data(args.input_data, 1, 100, pk_level=False)
        for i in range(0, 10):
            lr = 0.001
            print('fold', str(i))
            log_filename = os.path.join(subfolder, "training_" + str(i) + ".csv")
            w_filename = os.path.join(subfolder, "weights_" + str(i) + ".hdf5")
            datas.separate_fold(i)
            model = multiple_cnn1D(datas.X_data.shape[2])
            model_json = model.to_json()
            with open(model_file, "w") as json_file:
                json_file.write(model_json)

            model = train(model, datas, lr, log_filename, w_filename)

            print('Validation !!!!!!!')
            val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)


def train_classifier(args):
    '''
    Function that performs the detection of Parkinson
    :param args: Input arguments
    :return:
    '''
    exp_name = args.exp_name
    subfolder = os.path.join(args.output, exp_name +'_' + datetime.datetime.now().strftime("%m_%d"), datetime.datetime.now().strftime(
        "%H_%M"))
    file_result_patients = os.path.join(subfolder,'res_pat.csv')
    file_result_segments = os.path.join(subfolder,'res_seg.csv')
    model_file = os.path.join(subfolder, "model.json")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    val_results = Results(file_result_segments, file_result_patients)
    datas = Data(args.input_data, 1, 100, pk_level= False )

    for i in range(0, 10):
        lr = 0.001
        model = multiple_cnn1D(datas.X_data.shape[2])
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        print('fold', str(i))
        datas.separate_fold(i)
        log_filename = os.path.join( subfolder ,"training_" + str(i) + ".csv")
        w_filename = os.path.join(subfolder ,"weights_" + str(i) + ".hdf5")
        model = train(model, datas, lr, log_filename, w_filename)
        print('Validation !!')
        val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)

def train_severity(args):
    '''

    :param args: Input arguments
    :return:
    '''
    features = np.arange(1, 19)


    exp_name = args.exp_name

    subfolder = os.path.join(args.output, exp_name + '_' + datetime.datetime.now().strftime("%m_%d"), datetime.datetime.now().strftime(
        "%H_%M"))
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    file_result_patients = os.path.join(subfolder ,'res_pat.csv')
    file_result_segments = os.path.join(subfolder ,'res_seg.csv')

    model_file = os.path.join(subfolder, "model.json")

    val_results = Results_level(file_result_segments, file_result_patients, subfolder )
    datas = Data(args.input_data, 1, 100)  # modif
    lr = 0.001
    for i in range(0,10):

        model = multiple_cnn1D5_level(datas.X_data.shape[2])
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
        print('fold', str(i))
        datas.separate_fold(i)
        log_filename = os.path.join(subfolder, "trainig" + str(i) + ".csv")
        w_filename = os.path.join(subfolder ,"weights_" + str(i) + ".hdf5")
        model = train(model, datas, lr, log_filename,  w_filename )
        print('Validation !!')
        val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_data", default='../data', type=str)
    parser.add_argument("-exp_name", default='train_classifier', type=str, help = 'ablation ; train_classifier ; train_severity')
    parser.add_argument("-output", default='output', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.exp_name == 'ablation' :
        ablation_study(args)
    if args.exp_name == 'train_classifier' :
        train_classifier(args)

    if args.exp_name == 'train_severity':
        train_severity(args)
