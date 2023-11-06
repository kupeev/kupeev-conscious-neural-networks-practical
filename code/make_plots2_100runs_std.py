# make_plots2_100runs_std.py


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

from aside2 import get_vals, get_vals_100runs


process_labels = 0

all_Rs = ['R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R20', 'R30', 'R40', 'R50']

num_epochs_s_actually_used = [25, 50, 100, 200]

di = r'E:\AE\Projects\iterative-semiotics-networks\docs\out_make_plots2'

if not os.path.exists(di):
    os.makedirs(di)


def disp_pairs2(res_dict):

    num_epochs_s_all = get_vals_100runs(res_dict)[0]
    assert set(num_epochs_s_actually_used).issubset(set(num_epochs_s_all))


    #equiv to array of len=len(all_Rs) of lists
    #ev list: all acc values for this R (over all num_epochs_s and seed2's
    all_conns_prec = dict()
    for i in range(len(all_Rs)):
        all_conns_prec[i] = []
    all_benches_prec = dict()
    for i in range(len(all_Rs)):
        all_benches_prec[i] = []

    for num_epochs in num_epochs_s_actually_used:
        all_conns_prec, all_benches_prec = disp_pair2(num_epochs, res_dict,\
                all_conns_prec, all_benches_prec)
    disp_pair2('all_epochs', res_dict, all_conns_prec, all_benches_prec)

    return

def disp_pair2(num_epochs, res_dict, all_conns_prec, all_benches_prec):
    # like disp_pair(), if num_epochs=='all_epochs' then
    # Bench_num_epoch, mConn_num_epoch_s are used
    #
    #for ev num_epochs build graph over all nR's
    #   Conn: [63.7, 68.3, 72.6, 72.9, 77.1, 74.8, 82.1, 85.9, 88.8, 88.2]
    #   Bench: [63.9, 64.1, 68.6, 69.5, 70.4, 67.8, 82.1, 84.0, 85.8, 88.5]

    #num_epochs_s, nR_s = get_vals(res_dict)

    _,nR_s,_,_,_ = get_vals_100runs(res_dict)
    assert np.array_equal(nR_s,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    args = [all_Rs[n] for n, elt in enumerate(all_Rs)]

    arg_values = dict()
    #{'R5': 0, 'R6': 1, 'R7': 2, 'R8': 3, 'R9': 4, 'R10': 5, 'R20': 6, 'R30': 7, 'R40': 8, 'R50': 9}
    for n, arg in enumerate(args):
        arg_values[arg] = n

    # Mapping of strings to numerical values
    #    {'R5': 0, 'R6': 1, 'R7': 2, 'R8': 3, 'R9': 4, 'R10': 5, 'R20': 6, 'R30': 7, 'R40': 8, 'R50': 9}

    nR_s = np.array(np.arange(len(all_Rs)))
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    if isinstance(num_epochs, int):

        ConnMean = []
        ConnStd = []
        for nR in nR_s:
            acc_s = []
            for res in res_dict:
                if res[0] == num_epochs and res[1] == nR and res[2] == 'overall' \
                        and res[3] == 'op4':
                    acc_s.append(res_dict[res]['acc_op4'])
            all_conns_prec[nR] += list(acc_s)
            acc_s_mean = np.mean(acc_s)
            acc_s_std = np.std(acc_s)
            ConnMean.append(acc_s_mean)
            ConnStd.append(acc_s_std)
        assert nR == len(ConnMean) - 1

        BenchMean = []
        BenchStd = []
        for nR in nR_s:
            acc_s = []
            for res in res_dict:
                if res[0] == num_epochs and res[1] == nR and res[2] == 'overall' \
                        and res[3] == 'op5':
                    acc_s.append(res_dict[res]['acc_op5'])
            all_benches_prec[nR] += list(acc_s)
            acc_s_mean = np.mean(acc_s)
            acc_s_std = np.std(acc_s)
            BenchMean.append(acc_s_mean)
            BenchStd.append(acc_s_std)
        assert nR == len(BenchMean) - 1

        if 0:
            up = [ ConnMean[i] + ConnStd[i]/2 for i in range(len(ConnMean)) ]
            down = [ ConnMean[i] - ConnStd[i]/2 for i in range(len(ConnMean)) ]
            plt.scatter(nR_s, ConnMean, marker='o', color='blue', label='Conn')
            plt.scatter(nR_s, up, marker='D', color='blue', label='Conn')
            plt.scatter(nR_s, down, marker='D', color='blue', label='Conn')
            plt.plot(nR_s, ConnMean, '-o', color='blue', alpha=0.5)
            plt.plot(nR_s, up, '--o', color='blue', alpha=0.5)
            plt.plot(nR_s, down, '--o', color='blue', alpha=0.5)
        else:
            up = [ConnMean[i] + ConnStd[i]/2 for i in range(len(ConnMean))]
            down = [ConnMean[i] - ConnStd[i]/2 for i in range(len(ConnMean))]

            plt.plot(nR_s, ConnMean, '-o', color='blue', alpha=0.5, label='Conn: mean')
            plt.plot(nR_s, up, '--', color='blue', alpha=0.5, label='Conn: mean-std/2, mean+std/2')
            plt.plot(nR_s, down, '--', color='blue', alpha=0.5)

            plt.legend()

        if 0:
            up = [ BenchMean[i] + BenchStd[i]/2 for i in range(len(BenchMean)) ]
            down = [ BenchMean[i] - BenchStd[i]/2 for i in range(len(BenchMean)) ]
            plt.scatter(nR_s, BenchMean, marker='o', color='red', label='Bench')
            plt.scatter(nR_s, up, marker='D', color='red', label='Bench')
            plt.scatter(nR_s, down, marker='D', color='red', label='Bench')
            plt.plot(nR_s, BenchMean, '-o', color='red', alpha=0.5)
            plt.plot(nR_s, up, '--o', color='red', alpha=0.5)
            plt.plot(nR_s, down, '--o', color='red', alpha=0.5)
        else:
            up = [BenchMean[i] + BenchStd[i]/2 for i in range(len(BenchMean))]
            down = [BenchMean[i] - BenchStd[i]/2 for i in range(len(BenchMean))]

            plt.plot(nR_s, BenchMean, '-o', color='red', alpha=0.5, label='Bench: mean')
            plt.plot(nR_s, up, '--', color='red', alpha=0.5, label='Bench: mean-std/2, mean+std/2')
            plt.plot(nR_s, down, '--', color='red', alpha=0.5)

            plt.legend()

        plt.title('# of train epochs = ' + str(num_epochs))

        fn_png0 = 'max_iter=' + str(num_epochs) + '.png'
        plt.xticks(nR_s, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

    else:#if/else isinstance(, int):

        assert num_epochs == 'all_epochs'
        ConnMean = []
        ConnStd = []
        for nR in nR_s:
            acc_s_mean = np.mean(all_conns_prec[nR])
            acc_s_std = np.std(all_conns_prec[nR])
            ConnMean.append(acc_s_mean)
            ConnStd.append(acc_s_std)

        BenchMean = []
        BenchStd = []
        for nR in nR_s:
            acc_s_mean = np.mean(all_benches_prec[nR])
            acc_s_std = np.std(all_benches_prec[nR])
            BenchMean.append(acc_s_mean)
            BenchStd.append(acc_s_std)


        up = [ConnMean[i] + ConnStd[i]/2 for i in range(len(ConnMean))]
        down = [ConnMean[i] - ConnStd[i]/2 for i in range(len(ConnMean))]

        plt.plot(nR_s, ConnMean, '-o', color='blue', alpha=0.5, label='Conn: mean')
        plt.plot(nR_s, up, '--', color='blue', alpha=0.5, label='Conn: mean-std/2, mean+std/2')
        plt.plot(nR_s, down, '--', color='blue', alpha=0.5)

        plt.legend()


        up = [BenchMean[i] + BenchStd[i]/2 for i in range(len(BenchMean))]
        down = [BenchMean[i] - BenchStd[i]/2 for i in range(len(BenchMean))]

        plt.plot(nR_s, BenchMean, '-o', color='red', alpha=0.5, label='Bench: mean')
        plt.plot(nR_s, up, '--', color='red', alpha=0.5, label='Bench: mean-std/2, mean+std/2')
        plt.plot(nR_s, down, '--', color='red', alpha=0.5)

        plt.legend()


        #plt.title('Conn and Bench Over Series of Runs for Different Initializations and the Number of Trained Epochs')
        plt.title('Stoch and Bench Over the Entire Series of Runs')
        fn_png0 = 'over_all_iters' + '.png'

        plt.xticks(nR_s, args)
        plt.ylabel('Accuracy')
        plt.legend()

        fn_png = os.path.join(di, fn_png0)
        print('plot saved in ' + fn_png)
        plt.savefig(os.path.join(di, fn_png))
        #plt.show()
        plt.close()

    return all_conns_prec, all_benches_prec

#def disp_pair2(num_epochs, res_dict):

if __name__ == '__main__':
    res_dict = np.load('res_dict.cpu.npy', allow_pickle=True).tolist()

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
    
res['num_epochs'] = num_epochs
res['nR'] = nR
res['acc_op4/acc_op5'] = accuracy_op4*100
res_dict[(num_epochs, nR, 'overall', 'op4')] = res

=========== wth seed2

        for n, res in enumerate(res_dict):
            print(res_dict[res])
            #{'num_epochs': 15, 'nR': 5, 'nlab': 'overall', 'acc_op4': 73.5, 'seed2': 2000}

        for n, res in enumerate(res_dict):
            print(res)


        for n, res in enumerate(res_dict):
            if res[2] != 'overall':
                print(res_dict[res])
                #{'num_epochs': 3, 'nR': 9, 'nlab': 9, 'acc_op5': 87.5}

        for n, res in enumerate(res_dict):
            if res[2] != 'overall':
                print(res)
    
'''