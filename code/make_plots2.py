# make_plots2.py


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


process_labels = 0

all_Rs = ['R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R20', 'R30', 'R40', 'R50']

num_epochs_s_actually_used = [25, 50, 100, 200]

di = r'E:\AE\Projects\iterative-semiotics-networks\docs\out_make_plots2'

if not os.path.exists(di):
    os.makedirs(di)


def disp_pairs2(res_dict):

    num_epochs_s_all, _ = get_vals(res_dict)

    assert set(num_epochs_s_actually_used).issubset(set(num_epochs_s_all))

    Conn_num_epoch_s = []
    Bench_num_epoch_s = []


    for num_epochs in num_epochs_s_actually_used:
        Conn, Bench = disp_pair2(num_epochs, res_dict)
        Conn_num_epoch_s.append(Conn)
        Bench_num_epoch_s.append(Bench)
    disp_pair2('all_epochs', res_dict, Conn_num_epoch_s=Conn_num_epoch_s, \
            Bench_num_epoch_s=Bench_num_epoch_s)

    return

def disp_pair2(num_epochs, res_dict, Conn_num_epoch_s = None, Bench_num_epoch_s = None):
    # like disp_pair(), if num_epochs=='all_epochs' then
    # Bench_num_epoch, mConn_num_epoch_s are used
    #
    #dlya num_epochs bezim po po nR-arkam
    #acumnuilieruem Conn and Bench, say
    #   Conn: [63.7, 68.3, 72.6, 72.9, 77.1, 74.8, 82.1, 85.9, 88.8, 88.2]
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

        Conn = dict()
        Bench = dict()

        Conn['overall'] = []
        for nR in x:
            # nR is like enymerator of 10 vals of 'Accuracy on test set'
            # in May19Multirun_op4.max_iter=3.txt
            res = res_dict[(num_epochs, nR, 'overall', 'op4')]
            #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 'overall', 'acc_op4': 45.5}
            assert res['nR'] == nR
            acc = res['acc_op4']
            Conn['overall'].append(acc)
        assert nR == len(Conn['overall']) - 1

        Bench['overall'] = []
        for nR in x:
            # nR is like enymerator of 10 vals of 'Accuracy on test set'
            # in May19Multirun_op5.max_iter=3.txt
            res = res_dict[(num_epochs, nR, 'overall', 'op5')]
            #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 'overall', 'acc_op4': 45.5}
            assert res['nR'] == nR
            acc = res['acc_op5']
            Bench['overall'].append(acc)
        assert nR == len(Bench['overall']) - 1

        plt.scatter(x, Conn['overall'], marker='o', color='blue', label='Conn')
        plt.scatter(x, Bench['overall'], marker='s', color='red', label='Bench')
        plt.plot(x, Conn['overall'], '-o', color='blue', alpha=0.5)
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

        y = [ conn - bench for conn, bench in zip(Conn['overall'], Bench['overall'])]
        plt.plot(x, y, '-o', color='blue', alpha=0.5)
        # Set the y-axis limits symmetrically around y = 0
        y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
        plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
        # Add a horizontal line at y = 0
        plt.axhline(0, color='black', linewidth=0.5)

        plt.title('Conn - Bench for # of train epochs = ' + str(num_epochs))
        fn_png0 = 'max_iter=' + str(num_epochs) + '_D.png'
        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

        nlab_range = [] if not process_labels else range(10)
        for nlab in nlab_range:
            Conn[nlab] = []
            for nR in x:
                # nR is like enymerator of 10 vals of 'Accuracy on test set'
                # in May19Multirun_op4.max_iter=3.txt
                res = res_dict[(num_epochs, nR, nlab, 'op4')]
                #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 9, 'acc_op4': 45.5}
                assert res['nR'] == nR
                acc = res['acc_op4']
                Conn[nlab].append(acc)
            assert nR == len(Conn[nlab]) - 1

            Bench[nlab] = []
            for nR in x:
                # nR is like enymerator of 10 vals of 'Accuracy on test set'
                # in May19Multirun_op5.max_iter=3.txt
                res = res_dict[(num_epochs, nR, nlab, 'op5')]
                #Out[20]: {'num_epochs': 3, 'nR': 0, 'nlab': 9, 'acc_op5': 45.5}
                assert res['nR'] == nR
                acc = res['acc_op5']
                Bench[nlab].append(acc)
            assert nR == len(Bench[nlab]) - 1

            plt.scatter(x, Conn[nlab], marker='o', color='blue', label='Conn')
            plt.scatter(x, Bench[nlab], marker='s', color='red', label='Bench')
            plt.plot(x, Conn[nlab], '-o', color='blue', alpha=0.5)
            plt.plot(x, Bench[nlab], '-o', color='red', alpha=0.5)
            plt.title('# of train epochs = ' + str(num_epochs) + ' label ' + str(nlab))
            plt.xticks(x, args)
            plt.ylabel('Accuracy')
            plt.legend()

            fn_png0 = 'lab_' +str(nlab) + '_max_iter=' + str(num_epochs) + '.png'
            fn_png = os.path.join(di, fn_png0)
            print('plot saved in ' + fn_png)
            plt.savefig(os.path.join(di, fn_png))
            #plt.show()
            plt.close()

            # -------------
            y = [ conn - bench for conn, bench in zip(Conn[nlab], Bench[nlab])]

            plt.plot(x, y, '-o', color='blue', alpha=0.5)
            # Set the y-axis limits symmetrically around y = 0
            y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
            plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
            # Add a horizontal line at y = 0
            plt.axhline(0, color='black', linewidth=0.5)

            plt.title('Conn - Bench for # of train epochs = ' + str(num_epochs) + ' label ' + str(nlab))
            fn_png0 = 'lab_' +str(nlab) + '_max_iter=' + str(num_epochs) + '_D.png'
            plt.xticks(x, args)
            plt.ylabel('Accuracy')
            plt.legend()

            fn_png = os.path.join(di, fn_png0)
            print('plot saved in ' + fn_png)
            plt.savefig(os.path.join(di, fn_png))
            #plt.show()
            plt.close()

        #for nlab in range(10):

        tmp=10

    else:#if/else isinstance(, int):

        assert num_epochs == 'all_epochs'

        for n, (Conn, Bench) in enumerate(zip(Conn_num_epoch_s, Bench_num_epoch_s)):


            if n==0:
                ConnMax = Conn['overall']
                BenchMax = Bench['overall']
            else:
                ConnMax = [max(x,y) for x, y in zip(ConnMax, Conn['overall'])]
                BenchMax = [max(x,y) for x, y in zip(BenchMax, Bench['overall'])]


        plt.plot(x, ConnMax, '-o', color='blue', alpha=0.5)
        plt.plot(x, BenchMax, '-o', color='red', alpha=0.5)
        plt.scatter(x, ConnMax, marker='o', color='blue', label='pointwise max Conn (over all train sets)')
        plt.scatter(x, BenchMax, marker='s', color='red', label='pointwise max Bench (over all train sets)')


        plt.title('pointwise max Conn and pointwise max Bench')
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
        for n, (Conn, Bench) in enumerate(zip(Conn_num_epoch_s, Bench_num_epoch_s)):
            if n==0:
                ConnMax = Conn['overall']
                BenchMax = Bench['overall']
            else:
                ConnMax = [max(x,y) for x, y in zip(ConnMax, Conn['overall'])]
                BenchMax = [max(x,y) for x, y in zip(BenchMax, Bench['overall'])]
                tmp=10

        y = [ conn - bench for conn, bench in zip(ConnMax, BenchMax)]
        plt.plot(x, y, '-o', color='blue', alpha=0.5)
        # Set the y-axis limits symmetrically around y = 0
        y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
        plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
        # Add a horizontal line at y = 0
        plt.axhline(0, color='black', linewidth=0.5)

        plt.title('pointwise max Conn - pointwise max Bench (over all train sets)')
        fn_png0 = 'over_all_iters' + '_D.png'

        plt.xticks(x, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()


        nlab_range = [] if not process_labels else range(10)
        for nlab in nlab_range:
            for n, (Conn, Bench) in enumerate(zip(Conn_num_epoch_s, Bench_num_epoch_s)):
                plt.plot(x, Conn[nlab], '-o', color='blue', alpha=0.5)
                plt.plot(x, Bench[nlab], '-o', color='red', alpha=0.5)
                if n == 0:
                    plt.scatter(x, Conn[nlab], marker='o', color='blue', label='Conn')
                    plt.scatter(x, Bench[nlab], marker='s', color='red', label='Bench')

            plt.title('overall = ' + str(num_epochs) + ' label ' + str(nlab))
            plt.xticks(x, args)
            plt.ylabel('Accuracy')
            plt.legend()
            fn_png0 = 'lab_' + str(nlab) + '_overall_override_by_bench' +  '.png'

            fn_png = os.path.join(di, fn_png0)
            print('plot saved in ' + fn_png)
            plt.savefig(os.path.join(di, fn_png))
            #plt.show()
            plt.close()

            #-----------

            for n, (Conn, Bench) in enumerate(zip(Conn_num_epoch_s, Bench_num_epoch_s)):

               if n==0:
                    ConnMax = Conn[nlab]
                    BenchMax = Bench[nlab]
               else:
                    ConnMax = [max(x,y) for x, y in zip(ConnMax, Conn[nlab])]
                    BenchMax = [max(x,y) for x, y in zip(BenchMax,Bench[nlab])]
                    tmp=10

               y = [ conn - bench for conn, bench in zip(ConnMax, BenchMax)]
               plt.plot(x, y, '-o', color='blue', alpha=0.5)
               # Set the y-axis limits symmetrically around y = 0
               y_abs_max = max(np.abs(np.min(y)), np.abs(np.max(y)))
               plt.ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
               # Add a horizontal line at y = 0
               plt.axhline(0, color='black', linewidth=0.5)

               plt.title('max(Conn) - max(Bench) over all train sets for nlab='+str(nlab))
               fn_png0 = 'lab_' + str(nlab) + '_overall_override_by_bench' +  '_D.png'

               plt.xticks(x, args)
               plt.ylabel('Accuracy')
               plt.legend()

               fn_png = os.path.join(di, fn_png0)
               print('plot saved in ' + fn_png)
               plt.savefig(os.path.join(di, fn_png))
               #plt.show()
               plt.close()


        #for nlab in range(10):

        tmp=10


    return Conn, Bench

#def disp_pair2(num_epochs, res_dict):

if __name__ == '__main__':
    res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()

    #num_epochs_s, nR_s = get_vals(res_dict)

    disp_pairs2(res_dict)

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

disp_pairs2()
disp_pair2()
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