# make_plots2_3graphs.py


import sys
import os
from matplotlib.widgets import Button
import builtins
import shutil
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
import datetime

from general import is_empty, d0a, Str, i2d, t2i, i2t, struct
import re
import matplotlib.pyplot as plt
import numpy as np

from aside2 import get_vals

res_dict_fn = r'res_dict.reinit2.npy'
res_dict_vanilla_fn = r'res_dict.reinit2.vanilla.npy'

process_labels = 0

all_Rs = ['R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R20', 'R30', 'R40', 'R50']

num_epochs_s_actually_used = [25, 50, 100, 200]

di = r'E:\AE\Projects\iterative-semiotics-networks\docs\out_make_plots2'

if not os.path.exists(di):
    os.makedirs(di)


def disp_3_s(res_dict, res_dict_v):
# _v stands for vanilla
    num_epochs_s_all, _ = get_vals(res_dict)

    assert set(num_epochs_s_actually_used).issubset(set(num_epochs_s_all))

    Stoch_num_epoch_s = []
    Vanil_num_epoch_s = []
    Bench_num_epoch_s = []


    for num_epochs in num_epochs_s_actually_used:
        #_v stands for vanilla
        Stoch, Vanil, Bench = disp_3(num_epochs, res_dict, res_dict_v)
        Stoch_num_epoch_s.append(Stoch)
        Vanil_num_epoch_s.append(Vanil)
        Bench_num_epoch_s.append(Bench)
    disp_3('all_epochs', res_dict, res_dict_v,\
           Stoch_num_epoch_s=Stoch_num_epoch_s,\
           Vanil_num_epoch_s = Vanil_num_epoch_s,\
           Bench_num_epoch_s=Bench_num_epoch_s)

    return

def disp_3(num_epochs, res_dict, res_dict_v,\
        Stoch_num_epoch_s = None,\
        Vanil_num_epoch_s = None,\
        Bench_num_epoch_s = None):
    # like disp_pair(), if num_epochs=='all_epochs' then
    # Bench_num_epoch, Stoch_num_epoch_s are used
    #
    #for ev num_epochs enum all nR's
    #acumnuilating Stoch and Bench, say
    #   Stoch: [63.7, 68.3, 72.6, 72.9, 77.1, 74.8, 82.1, 85.9, 88.8, 88.2]
    #   Bench: [63.9, 64.1, 68.6, 69.5, 70.4, 67.8, 82.1, 84.0, 85.8, 88.5]

    #num_epochs_s, nR_s = get_vals(res_dict)

    _, nR_s = get_vals(res_dict)

    args = [all_Rs[n] for n, elt in enumerate(all_Rs)]

    arg_values = dict()
    #{'R5': 0, 'R6': 1, 'R7': 2, 'R8': 3, 'R9': 4, 'R10': 5, 'R20': 6, 'R30': 7, 'R40': 8, 'R50': 9}
    for n, arg in enumerate(args):
        arg_values[arg] = n

    # Mapping of strings to numerical values
    #    {'R5': 0, 'R6': 1, 'R7': 2, 'R8': 3, 'R9': 4, 'R10': 5, 'R20': 6, 'R30': 7, 'R40': 8, 'R50': 9}

    x = np.array(np.arange(len(all_Rs)))
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    if isinstance(num_epochs, int):

        Stoch = dict()
        Vanil = dict()
        Bench = dict()

        Stoch['overall'] = []
        Vanil['overall'] = []
        Bench['overall'] = []
        for nR in x:
            # nR is like enymerator of 10 vals of 'Accuracy on test set'
            # in May19Multirun_op4.max_iter=3.txt
            res = res_dict[(num_epochs, nR, 'overall', 'op4')]
            #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 'overall', 'acc_op4': 45.5}
            res_v = res_dict_v[(num_epochs, nR, 'overall', 'op4')]
            assert res['nR'] == nR
            acc = res['acc_op4']
            acc_v = res_v['acc_op4']
            Stoch['overall'].append(acc)
            Vanil['overall'].append(acc_v)

            # nR is like enymerator of 10 vals of 'Accuracy on test set'
            # in May19Multirun_op5.max_iter=3.txt
            res = res_dict[(num_epochs, nR, 'overall', 'op5')]
            #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 'overall', 'acc_op4': 45.5}
            assert res['nR'] == nR
            acc = res['acc_op5']
            Bench['overall'].append(acc)

            
        assert nR == len(Stoch['overall']) - 1
        assert nR == len(Bench['overall']) - 1

        plt.scatter(x, Stoch['overall'], marker='o', color='blue', label='Stoch')
        plt.scatter(x, Vanil['overall'], marker='o', color='green', label='Vanilla')
        plt.scatter(x, Bench['overall'], marker='s', color='red', label='Bench')
        plt.plot(x, Stoch['overall'], '-o', color='blue', alpha=0.5)
        plt.plot(x, Vanil['overall'], '-o', color='green', alpha=0.5)
        plt.plot(x, Bench['overall'], '-o', color='red', alpha=0.5)
        plt.title('# of train epochs = ' + str(num_epochs))

        fn_png0 = 'max_iter=' + str(num_epochs) + '.png'
        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

        #------------------

        y = [ stoch - bench for stoch, bench in zip(Stoch['overall'], Bench['overall'])]
        plt.plot(x, y, '-o', color='blue', alpha=0.5)
        # Set the y-axis limits symmetrically around y = 0
        y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
        plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
        # Add a horizontal line at y = 0
        plt.axhline(0, color='black', linewidth=0.5)

        plt.title('Stoch - Bench for # of train epochs = ' + str(num_epochs))
        fn_png0 = 'max_iter=' + str(num_epochs) + '_D.png'
        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

        tmp=10

    else:#if/else isinstance(, int):

        assert num_epochs == 'all_epochs'

        for n, (Stoch, Vanil, Bench) in enumerate(zip(Stoch_num_epoch_s, Vanil_num_epoch_s, Bench_num_epoch_s)):
            if n==0:
                StochMax = Stoch['overall']
                VanilMax = Vanil['overall']
                BenchMax = Bench['overall']
            else:
                StochMax = [max(x,y) for x, y in zip(StochMax, Stoch['overall'])]
                VanilMax = [max(x,y) for x, y in zip(VanilMax, Vanil['overall'])]
                BenchMax = [max(x,y) for x, y in zip(BenchMax, Bench['overall'])]


        plt.plot(x, StochMax, '-o', color='blue', alpha=0.5)
        plt.plot(x, VanilMax, '-o', color='green', alpha=0.5)
        plt.plot(x, BenchMax, '-o', color='red', alpha=0.5)
        plt.scatter(x, StochMax, marker='o', color='blue', label='pointwise max Stoch (over all train sets)')
        plt.scatter(x, VanilMax, marker='o', color='green', label='pointwise max Vanil (over all train sets)')
        plt.scatter(x, BenchMax, marker='s', color='red', label='pointwise max Bench (over all train sets)')


        plt.title('pointwise: max Stoch, max Vanil, max Bench')
        fn_png0 = 'over_all_iters' + '.png'

        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

        #-----
        for n, (Stoch, Bench) in enumerate(zip(Stoch_num_epoch_s, Bench_num_epoch_s)):
            if n==0:
                StochMax = Stoch['overall']
                BenchMax = Bench['overall']
            else:
                StochMax = [max(x,y) for x, y in zip(StochMax, Stoch['overall'])]
                BenchMax = [max(x,y) for x, y in zip(BenchMax, Bench['overall'])]
                tmp=10

        y = [ stoch - bench for stoch, bench in zip(StochMax, BenchMax)]
        plt.plot(x, y, '-o', color='blue', alpha=0.5)
        # Set the y-axis limits symmetrically around y = 0
        y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
        plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
        # Add a horizontal line at y = 0
        plt.axhline(0, color='black', linewidth=0.5)

        plt.title('pointwise max Stoch - pointwise max Bench (over all train sets)')
        fn_png0 = 'over_all_iters' + '_D.png'

        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()


        tmp=10


    return Stoch, Vanil, Bench

#def disp_3(num_epochs, res_dict):


if __name__ == '__main__':
    res_dict = np.load(res_dict_fn, allow_pickle=True).tolist()
    res_dict_v = np.load(res_dict_vanilla_fn, allow_pickle=True).tolist()

    #num_epochs_s, nR_s = get_vals(res_dict)

    disp_3_s(res_dict, res_dict_v)

# if __name__ == '__main__':

'''
res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()
num_epochs_s, nR_s = get_vals(res_dict)
    num_epochs_s =  [15, 25]
    nR_s =  [0, 5, 6]

np.save('res_dict.npy', res_dict)
#res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()

res_dict[(num_epochs, nR, 'overall', 'op4')] = res
res_dict[(num_epochs, nR, 'overall', 'op5')] = res
res_dict[(num_epochs, nR, nlab, 'op4')] = res
res_dict[(num_epochs, nR, nlab, 'op5')] = res
    
res['num_epochs'] = num_epochs
res['nR'] = nR
res['nlab'] = nlab/'overall'
res['acc_op4/acc_op5'] = accuracy_op4*100
res_dict[(num_epochs, nR, 'overall', 'op4')] = res

disp_3_s()
disp_3()
get_vals()

for nR, (overrid_train_data_op4, overrid_train_data_op5) in enumerate(zip(overrid_train_data_s_op4, overrid_train_data_s_op5)):

nR_s =  [0, 5, 6] consist of indexes of 

overrid_train_data_s_op4 = [
    r'E:\AE\Projects\iterative-semiotics-networks\data\run0\ATRb\tens',
    r'E:\AE\Projects\iterative-semiotics-networks\data\run1\ATRb\tens',
    ...
    r'E:\AE\Projects\iterative-semiotics-networks\data\run9\ATRb\tens'
]

        
# NUMKI, ARKI ('R5', 'R6',..)
    
'''