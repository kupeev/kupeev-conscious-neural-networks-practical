#conn_aux.py

import sys
import os
from matplotlib.widgets import Button
import builtins
import shutil
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision as tv
from pathlib import Path
import datetime
import re

from general import is_empty, d0a, Str, i2d, t2i, i2t, struct



def get_info_for_save(conntype_4_make_joint_examples__test, nimabs, data__make_examples, am_per_case,\
        list_nimabs_vanilla = []):
    '''
    conntype_4_make_joint_examples__test = 'stoch' or 'vanilla'

    returns
    save
        whether to save for given nimabs
    file_suffix
        parameters of image nimabs
            'gt5_MO0_MS5_01'
                gt5: gt label is 5
            MO0
                appilcation M to O (observed) give label 0
            MS5
                appilcation M to S (seen) give label 5
            01
                0
                    appilcation M to Obs  NotEq to gt
                1
                    appilcation M to Seen Eq to gt
    '''


    '''
    save am_per_case for _00
    save am_per_case for _01
    save am_per_case for _10
    save am_per_case for _11

    where 1 index reflects:
        M(obs) == gt(obs)
    where 2 index reflects:
        M(seen) == gt(obs)

    '''
    if not hasattr(get_info_for_save, "processed__data__make_examples"):

        nimabs_values = []
        # Extract nimabs values from each file path
        for path in data__make_examples['filenames_op4']:
            nimabs_match = re.search(r'nimabs=(\d+)', path)
            if nimabs_match:
                nimabs_values.append(int(nimabs_match.group(1)))
        #nimabs_values[0:5] [103, 114, 117, 12, 123]

        '''
        niabs_to_index index is the index in the list data__make_examples['filenames_op4']
            data__make_examples['filenames_op4'][0:3]:
            'E:\\AE\\Projects\\iterative-semiotics-networks\\data\\result_data.aug0.simhconn=0\\run3\\ATEb\\tens\\0\\nimabs=103.nb=103.nim=0.pt',
            'E:\\AE\\Projects\\iterative-semiotics-networks\\data\\result_data.aug0.simhconn=0\\run3\\ATEb\\tens\\0\\nimabs=114.nb=114.nim=0.pt',
            'E:\\AE\\Projects\\iterative-semiotics-networks\\data\\result_data.aug0.simhconn=0\\run3\\ATEb\\tens\\0\\nimabs=117.nb=117.nim=0.pt']
        '''

        niabs_to_index = dict()
        for index, value in enumerate(nimabs_values):
            niabs_to_index[value] = index

        '''
        ind_by_nimabs[103] = 0
        ind_by_nimabs[114] = 1
        ind_by_nimabs[117] = 2
        '''

        II_0X = (data__make_examples['y_test'] != data__make_examples['predicted_op5']).numpy()
        II_1X = (data__make_examples['y_test'] == data__make_examples['predicted_op5']).numpy()
        II_X0 = (data__make_examples['y_test'] != data__make_examples['predicted_op4']).numpy()
        II_X1 = (data__make_examples['y_test'] == data__make_examples['predicted_op4']).numpy()

        II_00 = II_0X & II_X0
        II_01 = II_0X & II_X1
        II_10 = II_1X & II_X0
        II_11 = II_1X & II_X1

        nimabs_00_all = np.array(nimabs_values)[II_00]
        nimabs_01_all = np.array(nimabs_values)[II_01]
        nimabs_10_all = np.array(nimabs_values)[II_10]
        nimabs_11_all = np.array(nimabs_values)[II_11]

        nimabs_00 = np.random.permutation(nimabs_00_all)[0:\
                min(len(np.random.permutation(nimabs_00_all)),am_per_case)]
        nimabs_01 = np.random.permutation(nimabs_01_all)[0:\
                min(len(np.random.permutation(nimabs_01_all)),am_per_case)]
        nimabs_10 = np.random.permutation(nimabs_10_all)[0:\
                min(len(np.random.permutation(nimabs_10_all)),am_per_case)]
        nimabs_11 = np.random.permutation(nimabs_11_all)[0:\
                min(len(np.random.permutation(nimabs_11_all)),am_per_case)]

        get_info_for_save.nimabs_00 = nimabs_00
        get_info_for_save.nimabs_01 = nimabs_01
        get_info_for_save.nimabs_10 = nimabs_10
        get_info_for_save.nimabs_11 = nimabs_11
        get_info_for_save.niabs_to_index = niabs_to_index

        get_info_for_save.processed__data__make_examples = 1
    #if not hasattr(get_info_for_save, "processed__data__make_examples"):

    niabs_to_index = get_info_for_save.niabs_to_index


    if conntype_4_make_joint_examples__test == 'stoch':
        if nimabs in get_info_for_save.nimabs_00:
            file_suffix =\
                'gt' + str(data__make_examples['y_test'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MO' + str(data__make_examples['predicted_op5'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MS' + str(data__make_examples['predicted_op4'] [niabs_to_index[nimabs]].numpy()) + '_'+\
                '00'
            save = 1
        elif nimabs in get_info_for_save.nimabs_01:
            file_suffix =\
                'gt' + str(data__make_examples['y_test'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MO' + str(data__make_examples['predicted_op5'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MS' + str(data__make_examples['predicted_op4'] [niabs_to_index[nimabs]].numpy()) + '_'+\
                '01'
            save = 1
        elif nimabs in get_info_for_save.nimabs_10:
            file_suffix =\
                'gt' + str(data__make_examples['y_test'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MO' + str(data__make_examples['predicted_op5'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MS' + str(data__make_examples['predicted_op4'] [niabs_to_index[nimabs]].numpy()) + '_'+\
                '10'
            save = 1
        elif nimabs in get_info_for_save.nimabs_11:
            file_suffix =\
                'gt' + str(data__make_examples['y_test'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MO' + str(data__make_examples['predicted_op5'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MS' + str(data__make_examples['predicted_op4'] [niabs_to_index[nimabs]].numpy()) + '_'+\
                '11'
            save = 1
        else:
            file_suffix = []
            save = 0
    elif conntype_4_make_joint_examples__test == 'vanilla':
        if nimabs in list_nimabs_vanilla:

            if nimabs == 962:
                tmp=10

            file_suffix =\
                'gt' + str(data__make_examples['y_test'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MO' + str(data__make_examples['predicted_op5'] [niabs_to_index[nimabs]].numpy()) + '_' +\
                'MS' + str(data__make_examples['predicted_op4'] [niabs_to_index[nimabs]].numpy())
            save = 1
        else:
            file_suffix = []
            save = 0


    return save, file_suffix


'''
    data__make_examples = dict()
    data__make_examples['filenames_op4'] = filenames_op4
    data__make_examples['predicted_op4'] = predicted_op4
    data__make_examples['filenames_op5'] = filenames_op5
    data__make_examples['predicted_op5'] = predicted_op5
    data__make_examples['y_test'] = y_test
    data__make_examples['num_epochs'] = num_epochs
    data__make_examples['overrid_train_data_op4'] = overrid_train_data_s_op4[nR]
    data__make_examples['overrid_train_data_op5'] = overrid_train_data_s_op5[nR]


'''