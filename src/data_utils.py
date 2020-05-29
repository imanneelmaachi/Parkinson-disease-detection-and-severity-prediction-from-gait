# -*- coding: utf-8 -*-
"""

"""
import glob
import os
import random
import sys

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical

sys.path.append('../src')
np.random.seed(2)



class Data:

    def __init__(self,  input_data,  deep, gait_cycle, step=50, features=np.arange(1, 19), pk_level = True):
        '''

        :param load_or_get:  1: load data , 0: load preloaded datas ( npy)
        :param deep:  data in the format for deep learning algorithms
        :param gait_cycle: number of gait cycle per signal
        :param step: overlap between gait signals
        :param features: signals to be loaded ( coming from sensors)
        :param pk_level: if true , y is the parkinson level according
        '''

        self.deep = deep
        self.step = step
        self.nb_gait_cycle = gait_cycle


        self.features_to_load = features
        self.nb_features = self.features_to_load.shape[0]
        ###############
        self.X_data = np.array([])  # np.ones((self.nb_gait_cycle,self.nb_features))
        self.y_data = np.array([])
        self.nb_data_per_person = np.array([0])


        files = sorted(glob.glob(os.path.join(input_data, '*txt')))
        self.ctrl_list = []
        self.pk_list = []
        for file in files:

            if file.find(".txt") != -1:  # if control ("01.txt")
                if file.find("Co") != -1:  # if control
                    self.ctrl_list.append(file)
                elif file.find("Pt") != -1:  # if control
                    if pk_level:
                        if  file.find('SiPt02')!= -1:
                            pass
                        elif file.find('SiPt07')!= -1:
                            pass
                        else:
                            self.pk_list.append(file)
                    else :
                        self.pk_list.append(file)

        random.shuffle(self.ctrl_list)
        random.shuffle(self.pk_list)
        self.pk_level = pk_level
        if pk_level == True:
            self.levels = pd.read_csv( os.path.join(input_data, "demographics.csv"))
            self.levels.set_index('ID', inplace=True)
        self.load(norm=None)


    def separate_fold(self, fold_number, total_fold=10):
        '''

        :param fold_number: Fold number
        :param total_fold: Total number of fols
        :return:
        '''
        proportion = 1 / total_fold  # .10 for 10 folds
        X = [self.X_ctrl, self.X_park]
        y = [self.y_ctrl, self.y_park]
        patients = [self.nb_data_per_person[:self.last_ctrl_patient], self.nb_data_per_person[self.last_ctrl_patient:]] # counts separated by classe
        patients[1]= patients[1] - patients[1][0]
        diff_count = np.diff(self.nb_data_per_person)
        diff_count = [diff_count[:self.last_ctrl_patient], diff_count[self.last_ctrl_patient:]]
        self.count_val = np.array([0])
        self.count_train = np.array([0])
        for i in range(len(X)):
            nbr_patients =  int(len(patients[i]) *proportion)
            start_patient = int(fold_number*nbr_patients )
            end_patient = (fold_number+1)*nbr_patients
            id_start = patients[i][start_patient]  # segment start
            id_end = patients[i][end_patient]  # end segment
            if i ==0 :
                self.X_val = X[i][id_start:id_end,:,:]
                self.X_train = np.delete(X[i], np.arange(id_start,id_end) , 0)

                self.y_val = y[i][id_start:id_end]
                self.y_train = np.delete(y[i], np.arange(id_start,id_end) , 0)


                self.count_val = np.append(self.count_val, diff_count[i][start_patient: end_patient])
                self.count_train = np.append(self.count_train, np.delete(diff_count[i], np.arange(start_patient, end_patient)))



            else:
                start_patient = start_patient #+ patients[0].shape[0]  # patients0.shape 0 is the number of patients in the first class
                end_patient =  end_patient# +patients[0].shape[0]
                self.X_val = np.vstack((self.X_val, X[i][id_start:id_end,:,:]))
                self.X_train = np.vstack((self.X_train, np.delete(X[i], np.arange(id_start,id_end) , 0) ))

                self.y_val = np.vstack((self.y_val, y[i][id_start:id_end] ))
                self.y_train = np.vstack((self.y_train, np.delete(y[i], np.arange(id_start,id_end) , 0) ))

                self.count_val = np.append( self.count_val , diff_count[i][start_patient: end_patient])
                self.count_train = np.append(self.count_train,np.delete(diff_count[i], np.arange(start_patient, end_patient)) )

        self.count_val = np.cumsum(self.count_val)
        self.count_train = np.cumsum(self.count_train )


    def load(self, norm = 'std'):
        print("load training control ")
        self.load_data(self.ctrl_list, 0)
        if self.deep == 1:
            self.last_ctrl= self.X_data.shape[2]
            self.last_ctrl_patient = len(self.nb_data_per_person)
        print("load training parkinson ")


        self.load_data(self.pk_list, 1)  # ncycle, nfeature, nombre de data


        ## all datas are loaded at this point, preprocessing now
        if self.deep == 1:
            self.X_data = self.X_data.transpose(2,0 , 1)  #0, 1

            if norm == 'std ':
                self.normalize()
            elif norm == 'l2':
                self.X_data = self.normalize_l2(self.X_data)

        if self.pk_level:
            self.one_hot_encoding()

        if self.deep == 1:
            self.X_ctrl = self.X_data[:self.last_ctrl]
            self.y_ctrl =  self.y_data[:self.last_ctrl]
            self.X_park = self.X_data[self.last_ctrl:]
            self.y_park = self.y_data[self.last_ctrl:]

        print("saving training ")
        np.save("Xdata", self.X_data)
        np.save("ydata", self.y_data)
        np.save('data_person',self.nb_data_per_person)
        np.save('ctrl_list', self.ctrl_list)
        np.save('pk_list', self.pk_list)

    def normalize(self):
        '''

        :return: Normalize to have a mean =  and std =1
        '''
        mean_train = np.mean(self.X_train,(0,1))
        std_train = np.std(self.X_train,(0,1))
        self.X_train= (self.X_data - mean_train) / std_train
        self.X_test= (self.X_test - mean_train) / std_train
    def one_hot_encoding(self):
        '''

        :return: return one hot encoding vector for severity prediction
        '''
        self.y_data[self.y_data<=4]=0
        self.y_data[(4<self.y_data) & (self.y_data <15)]=1
        self.y_data[(15<= self.y_data) & (self.y_data<25)]=2
        self.y_data[(25<= self.y_data) & (self.y_data<35)] = 3
        self.y_data[35<= self.y_data] = 4
        self.y_data = to_categorical(self.y_data)

    def normalize_l2(self, data):
        '''

        :param data:  Function to perform L2 normalization
        :return:
        '''
        data = keras.backend.l2_normalize(data, axis=(1, 2))
        data = tf.keras.backend.get_value(data)
        return data


    def load_data(self, liste, y):
        '''

        :param liste: list of patients filepaths
        :param y: 0 for control, 1 for parkinson
        :return:
        '''

        for i in range(0, len(liste)):
            datas = np.loadtxt(liste[i])  # num cycle, n features
            datas = datas[:, self.features_to_load]
            print(datas.shape[0])
            if  self.pk_level :
                y =self.find_level(liste[i])

            if self.deep == 1:
                X_data, y_data , self.nb_data_per_person = self.generate_datas(datas, y, self.nb_data_per_person)
            else:
                X_data, y_data = self.generate_datas_ml(datas, y)
            if (self.X_data).size == 0:
                self.X_data = X_data
                self.y_data = y_data
            else:
                if self.deep == 1:
                    self.X_data = np.dstack((self.X_data, X_data))
                else:
                    self.X_data = np.vstack((self.X_data, X_data))  # shape nb data --- vector size
                self.y_data = np.vstack((self.y_data, y_data))

            print(X_data.shape, self.X_data.shape)



    def find_level(self,file):
        '''

        :param file: Dataframe
        :return:
        '''
        start = '\\'
        end = '_'
        id = (file.split(start))[1].split(end)[0]
        y = self.levels.loc[id,'UPDRS']
        return y


    def generate_datas(self, datas, y, data_list):
        '''

        :param datas:  datas loaded for 1 patient
        :param y: label of the patient
        :param data_list: list containing the number of segments per patients
        :return:
        '''
        count = 0
        X_data = np.array([])
        y_data = np.array([])
        nb_datas = int(datas.shape[0] - self.nb_gait_cycle)
        for start in range(0, nb_datas, self.step):
            end = start + self.nb_gait_cycle
            data = datas[start:end, :]
            if X_data.size == 0:
                X_data = data
                y_data = y
            else:
                if (self.deep == 1):
                    X_data = np.dstack((X_data, data))
                else:
                    X_data = np.vstack((X_data, data))
                y_data = np.vstack((y_data, y))
            count = count + 1
        data_list = np.append(data_list, count+ data_list[-1])
        return X_data, y_data, data_list


    def get_datas(self):
        return self.X_data, self.y_data, self.X_test, self.y_test, self.X_val, self.y_val

