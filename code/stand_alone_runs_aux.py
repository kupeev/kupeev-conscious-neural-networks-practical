#stand_alone_runs_aux.py

import sys
import os
from matplotlib.widgets import Button
import builtins
from general import is_empty, d0a, Str
import shutil
from sys import platform
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

from conn import dir_data, train_data_op4, test_data_op4, eu_op, op3_num_total, \
        eu_scale, extenc_am_aug, extenc_alpha, val_data_mt

class struct():
    pass


seed = 0
np.random.seed(seed=seed)
torch.manual_seed(seed)

aux6 = 0

if aux6:
    di00 = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug6June04'
    di0 = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug6June04'
    di0mt = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug6June04\result_mt_models'

    di1_s = [\
        r'model_rmnist_5aug6_Ep100000_p1',
        r'model_rmnist_6aug6_Ep100000_p1',
        r'model_rmnist_7aug6_Ep100000_p1',
        r'model_rmnist_8aug6_Ep100000_p1',
        r'model_rmnist_9aug6_Ep100000_p1',
        r'model_rmnist_10_aug6_Ep100000_p1',
        r'model_rmnist_20_aug6_Ep100000_p1',
        r'model_rmnist_30_aug6_Ep100000_p1',
        r'model_rmnist_40_aug6_Ep100000_p1',
        r'model_rmnist_50_aug6_Ep100000_p1',
    ]

    di1mt_s = [\
        r'model_rmnist_5aug6_Ep10000_p1',
        r'model_rmnist_6aug6_Ep10000_p1',
        r'model_rmnist_7aug6_Ep10000_p1',
        r'model_rmnist_8aug6_Ep10000_p1',
        r'model_rmnist_9aug6_Ep10000_p1',
        r'model_rmnist_10_aug6_Ep10000_p1',
        r'model_rmnist_20_aug6_Ep10000_p1',
        r'model_rmnist_30_aug6_Ep10000_p1',
        r'model_rmnist_40_aug6_Ep10000_p1',
        r'model_rmnist_50_aug6_Ep10000_p1',
    ]

else:
    di00 = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug0June04'
    di0 = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug0June04\models'
    di0mt = r'E:\AE\Projects\iterative-semiotics-networks\data\AllDataCopiedAug0June04\models_mt'


    di1_s = [\
        r'model_rmnist_5aug0_Ep100000_p1',
        r'model_rmnist_6aug0_Ep100000_p1',
        r'model_rmnist_7aug0_Ep100000_p1',
        r'model_rmnist_8aug0_Ep100000_p1',
        r'model_rmnist_9aug0_Ep100000_p1',
        r'model_rmnist_10aug0_Ep100000_p1',
        r'model_rmnist_20aug0_Ep100000_p1',
        r'model_rmnist_30aug0_Ep100000_p1',
        r'model_rmnist_40aug0_Ep100000_p1',
        r'model_rmnist_50aug0_Ep100000_p1',
    ]

    di1mt_s = [\
        r'model_rmnist_5aug0_Ep10000_p1',
        r'model_rmnist_6aug0_Ep10000_p1',
        r'model_rmnist_7aug0_Ep10000_p1',
        r'model_rmnist_8aug0_Ep10000_p1',
        r'model_rmnist_9aug0_Ep10000_p1',
        r'model_rmnist_10aug0_Ep10000_p1',
        r'model_rmnist_20aug0_Ep10000_p1',
        r'model_rmnist_30aug0_Ep10000_p1',
        r'model_rmnist_40aug0_Ep10000_p1',
        r'model_rmnist_50aug0_Ep10000_p1',
    ]


def run():
    # for /f "delims=" %F in ('dir /b "G:\Downloads\AllDataCopiedAug6June04"') do @echo.%F
    # for /d %F in ("G:\Downloads\AllDataCopiedAug6June04\*") do @echo %~nxF
    for (n,di1) in enumerate(di1_s):
        di = os.path.join(di0, di1)

        n_epochs = np.load(di + r'/n_epochs.npy', allow_pickle=True)
        losses = np.load(di + r'/losses.npy', allow_pickle=True)
        plt.plot(n_epochs,losses)
        plt.title('min loss = ' + Str(np.min(losses)))
        fn_png0 = 'loss_n=' + str(n)
        if 1:
            fn_png = os.path.join(di00, fn_png0)
            print('plot saved in ' + fn_png)
            plt.savefig(fn_png)
        else:
            plt.show()
        plt.close()
    #for (n,di1) in enumerate(di1_s):


    for (n,di1mt) in enumerate(di1mt_s):
        dimt = os.path.join(di0mt, di1mt)

        n_epochs = np.load(dimt + r'/n_epochs_mt.npy', allow_pickle=True)
        losses = np.load(dimt + r'/losses_mt.npy', allow_pickle=True)
        plt.plot(n_epochs,losses)
        plt.title('min loss = ' + Str(np.min(losses)))
        fn_png0 = 'mt_loss_n=' + str(n)
        if 1:
            fn_png = os.path.join(di00, fn_png0)
            print('plot saved in ' + fn_png)
            plt.savefig(fn_png)
        else:
            plt.show()
        plt.close()
    #for (n,di1) in enumerate(di1_s):

    tmp=10

    return

def run2():

    folder_path = r'G:\Downloads\models_mt\models_mt'

    # Iterate over each folder
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_dir):
            continue

        # Find the epochXXX_mt.pth file
        model_file = None
        for file_name in os.listdir(folder_dir):
            if file_name.startswith('epoch') and file_name.endswith('_mt.pth'):
                number = int(''.join(filter(str.isdigit, file_name)))
                break


        # Find the mt_hist.txt file and retrieve the loss value
        hist_file = os.path.join(folder_dir, 'mt_hist.txt')
        with open(hist_file, 'r') as file:
            for line in file:
                if 'epoch = ' + str(number) in line:
                    mt_loss = float(line.split("loss = ")[1].split()[0])
                    #print('mt_loss = ' + str(mt_loss))
                    break

        print(f"Epoch: {folder_name}, Loss: {mt_loss}, Model File: {model_file}")
    #for folder_name in os.listdir(folder_path):

    '''
    
    Epoch: model_rmnist_5aug0_Ep10000_p1, Loss: 0.03, Model File: None
    Epoch: model_rmnist_6aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_7aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_8aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_9aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_10aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_20aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_30aug0_Ep10000_p1, Loss: 0.04, Model File: None
    Epoch: model_rmnist_40aug0_Ep10000_p1, Loss: 0.03, Model File: None
    Epoch: model_rmnist_50aug0_Ep10000_p1, Loss: 0.04, Model File: None
    
    '''

#def run2():



if __name__ == '__main__':
    #run()
    run2()
# if __name__ == '__main__':




