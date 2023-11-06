dbg20 = 1 #development of aside2

dbg21 = 0  # debug for map_examples_to_attractors: test consistency of data folders

dbg18 = 0  # bug search

dbg11 = 0  # low assert level
dbg13 = 0  # debug prp for loss
dbg14 = 0  # short run
dbg15 = 1  # for eu_op = 'op2', 'op3', 'op2op3': amHist = 500
dbg17 = 1  # for eu_op = 'op2', 'op3', 'op2op3': batch_size = 1

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

from general import is_empty, d0a, Str, i2d, t2i, i2t, struct

load_trained = 1

eu = 1
eu_op = 'op2op3'  # 'op2op3'

# at op3 the test data is sampled (op3_num_total samples) from
# standart mnist test data, then convereed to attractors
# (in map_examples_to_attractors) and then is written to ATE data folder.
# we need the same test data (without conversion for attractors)
# for running  the benchmark torch classifier. This folder (mnist_test_data_folder)
# is created as in op3, with skipping coversion to attractors
make_test_data_folder = 0


tr_data_mt = 0  # use val data for model tuning, besides the standard train
# for 'op1':
#   load ready model
#   new model at each epoch saves to fn with _mt

val_data_mt = 0  # use val data for model tuning
# if op0:
#   additionally to say \rmnist_50_aug6\train
#   we create rmnist_50_aug6\train_proper and
#   we create rmnist_50_aug6\train_validate
# if op1:
#   we perform training at train_proper (instead of train)
#   perform trainbing at train_proper and model refimenent at train_validate
# if op2, op3: no change: use the whole train data


assert not (val_data_mt and tr_data_mt)

op_batch = 1  # op2', 'op3

batch_size = 4  # 256 #win: 1: 1 min and 15 sec 64: 27 sec, 128 - 25 sec
if dbg17 and eu_op in ['op2', 'op3', 'op2op3']:
    batch_size = 1

skip = 0
save_loss_epochs = 2500
if dbg14:
    save_loss_epochs = 1
h2 = 0

epochs = 100000
if dbg14:
    epochs = 5
elif tr_data_mt:
    epochs = 10000

# fully conv autoenc, based on the network from Udacity:
#       https://github.com/udacity/pytorchcrashcourse MIT License Copyright (c) 2017 Udacity
fconv = 1  # fully conv autoenc

blk = 1  # netw arch of belkin
blk_prs = struct()
if fconv == 0:
    blk_prs.add = 0
    blk_prs.lr = 1e-5  # deflt: 1e-4
    blk_prs.seed = 2
    blk_prs.wi = 2048
elif fconv == 1:
    blk_prs.lr = 1e-5  # deflt: 1e-4 our:1e-5
    blk_prs.seed = 3
    if 0:
        blk_prs.krn = 1
        blk_prs.pad = 0
    else:
        blk_prs.krn = 3
        blk_prs.pad = 1

# use augm for proroducing trains and test data (op0)
extenc_am_aug = 6  # 0 means no augmentation
extenc_alpha = 1.

# if fconv, lat_dimension is not used
if h2:
    lat_dimension = 10
else:
    lat_dimension = 100

hconn = 1  # ('op2', 'op3', 'op2op3')
if hconn:
    aug_h = 0
    hconn_simulatenonhconn = 0
    if hconn_simulatenonhconn:
        amHist = 1
    elif dbg15:
        amHist = 500
    else:
        amHist = 1000

    alpha = 1.
    betha = 2.6
    averaging = 'mean'  # 'mean' or 'fro'

if eu:
    if eu_op in ['op2', 'op3', 'op2op3']:
        if hconn:
            # k_h is the of descents wth aug
            k_h = nsteps_p1 = 30
        else:
            nsteps_p1 = 100
        if dbg21:
            k_h = nsteps_p1 = 5
else:
    nsteps_p1 = 50
    nsteps_p2 = 50

eu_scale = None  # '18x18'
heavy_network = 1

if eu_scale:
    assert eu_scale == '18x18'
    len_mnist_patch = 18 * 18
    sz_mnist_patch = 18
else:
    len_mnist_patch = 28 * 28
    sz_mnist_patch = 28

is_win = (sys.platform[:3] == "win")

op3_num_total = 1000  # note that shuffle = True on dataload creation, thus we pick these 1000 randomly

if is_win:
    eu_op_db_spec_s = [ \
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_5aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_6aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_7aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_8aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_9aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_10-50_aug0\rmnist_10aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_10-50_aug0\rmnist_20aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_10-50_aug0\rmnist_30aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_10-50_aug0\rmnist_40aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_10-50_aug0\rmnist_50aug0\train\tens',
    ]
    model_fn_op2op3_s = [ \
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_5aug0_Ep10000_p1/epoch335_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_6aug0_Ep10000_p1/epoch4751_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_7aug0_Ep10000_p1/epoch790_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_8aug0_Ep10000_p1/epoch3931_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_9aug0_Ep10000_p1/epoch6352_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_10aug0_Ep10000_p1/epoch9884_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_20aug0_Ep10000_p1/epoch124_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_30aug0_Ep10000_p1/epoch9023_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_40aug0_Ep10000_p1/epoch5610_mt.pth',
        r'E:\AE\Projects\iterative-semiotics-networks\models\models_mt/model_rmnist_50aug0_Ep10000_p1/epoch3332_mt.pth',
    ]

    if tr_data_mt:  # where from the models are taken
        # no sense meanwhile
        tr_data_mt_models = [
            r'E:\AE\Projects\iterative-semiotics-networks\models\model_rmnist_5aug0_Ep5_p1.pth',
            r'E:\AE\Projects\iterative-semiotics-networks\models\model_rmnist_6aug0_Ep5_p1.pth',
            r'E:\AE\Projects\iterative-semiotics-networks\models\model_rmnist_7aug0_Ep5_p1.pth',
            r'E:\AE\Projects\iterative-semiotics-networks\models\model_rmnist_8aug0_Ep5_p1.pth',
            r'E:\AE\Projects\iterative-semiotics-networks\models\model_rmnist_9aug0_Ep5_p1.pth',
        ]

    # for make_test_data_folder
    mnist_test_data_folder = r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\test_data'
else:

    eu_op_db_spec_s = [ \
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_5aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_6aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_7aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_8aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_9aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_10aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_20aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_30aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_40aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/MNIST/rmnist_50aug0/train/tens',
    ]
    model_fn_op2op3_s = [ \
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_5aug0_Ep10000_p1/epoch335_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_6aug0_Ep10000_p1/epoch4751_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_7aug0_Ep10000_p1/epoch790_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_8aug0_Ep10000_p1/epoch3931_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_9aug0_Ep10000_p1/epoch6352_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_10aug0_Ep10000_p1/epoch9884_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_20aug0_Ep10000_p1/epoch124_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_30aug0_Ep10000_p1/epoch9023_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_40aug0_Ep10000_p1/epoch5610_mt.pth',
        r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_50aug0_Ep10000_p1/epoch3332_mt.pth',
    ]

    if tr_data_mt:  # where from the models are taken
        tr_data_mt_models = [
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_5aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_6aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_7aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_8aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_9aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_10_aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_20_aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_30_aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_40_aug6_Ep100000_p1.pth',
            r'/home/research/iterative-semiotics-networks/models/models_mt/model_rmnist_50_aug6_Ep100000_p1.pth',
        ]

    # for make_test_data_folder
    mnist_test_data_folder = r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/test_data'
# if/else is_win:


if dbg14:
    eu_op_db_spec_s = [eu_op_db_spec_s[0], eu_op_db_spec_s[1]]
    model_fn_op2op3_s = [model_fn_op2op3_s[0], model_fn_op2op3_s[1]]
    if tr_data_mt:
        tr_data_mt_models = [tr_data_mt_models[0], tr_data_mt_models[1]]

if eu_op != 'op1':
    assert len(eu_op_db_spec_s) == len(model_fn_op2op3_s)

assert hconn==0 or (not (dbg15 and hconn_simulatenonhconn))

train_data_op4 = r'E:\AE\Projects\iterative-semiotics-networks\data\runs.aug0\run9\ATRb\tens'
test_data_op4  = r'E:\AE\Projects\iterative-semiotics-networks\data\runs.aug0\run9\ATEb\tens'

dbg_get_aug_geom = 0
dbg_get_aug_radiom = 0

if eu:
    if eu_op in ['op2', 'op3', 'op2op3']:
        assert load_trained == 1
    elif eu_op in ['op1']:
        assert load_trained == 0

''' 
autoencoders are associated wth the persons. Persons include loaders.

 ( op0 ) 
    what is doing: 
        create TRa (train data of mnist/RMNIST) in 
        eu_op_db_spec 
        and eg rmnist_10.pkl.gz
    inp: download 
    out: see 'what is doing'
    how to run: 
        stand_alone_runs.py/make_rmnist, search 'op0', set n=10 or 50 etc make_rmnist() argument there
    rem:
        read_train_data_from_standard_MNIST_data_location is used         
( op1 ) 
    what is doing:
        create auto enc trained at TRa  and put 
        it to model dir (loader of TRa is associated wth P1)
    inp:
        eu_op_db_spec     
    out:
        model representing auto enc in the model dir 
    how to run:
        ensure model_fn, eu_op_db_spec
        conn
( op2 )  
    what is doing:
        using TRb (=Tra) create attractors ATRb and put it to folder
        dir_data_out_png = dir_data + 'ATRb/png'
        dir_data_out_tens = dir_data + 'ATRb/tens'
        this dir will be used in op4 as train_data_op4
    inp:
        if eu_op_db_spec is not empty, use it
        else standart creation of autoenc at labs_TRb 
    out:
        dir_data_out_png = dir_data + 'ATRb/png'
        dir_data_out_tens = dir_data + 'ATRb/tens'
    how to run:
        define model_fn_op2op3             
        ensure model_fn
        conn
        asssert what are dir_data_out_png dir_data_out_tens


( op3 )
    what is doing: 
        using TEb (test data of mnist) crate attractors ATEb and put it to folder 
        dir_data_out_png = dir_data + 'ATEb/png'
        dir_data_out_tens = dir_data + 'ATEb/tens'
        this dir will be used in op4 as test_data_op4 
    inp:
        standart test data of mnist
            called is get_mnist_data_dataloader(labs_TRb, train_data_mnist=0) 
                my_dataloader = torch.utils.data.DataLoader(shuffle=True)
                created are train-images-idx3-ubyte etc in dir_data/MNIST/raw
                then op3_num_total is used for creation  ATEb            

    out:
        dir_data_out_png = dir_data + 'ATEb/png'
        dir_data_out_tens = dir_data + 'ATEb/tens'
    how to run: 
        define model_fn_op2op3             
        ensure model_fn
        conn


( op4 )

    Note: in a new version is implemented by aside.py

    what is doing: 
        using N1 (eg as netw of Alaa) 
        train at train_data_op4
        test atr test_data_op4
        get results at ATEb 
    inp:

        train_data_op4
        test_data_op4

        !
        these should be equal to 
            dir_data_out_png
            dir_data_out_tens
            dir_data_out_png
            dir_data_out_tens
        made at op2 and op3            

    out:
        text output

    how to run:
        define train_data_op4, test_data_op4
        run stand_alone_runs.py
    rems
        overrid_train_data = train_data_op4, overrid_test_data = test_data_op4    


( op5 )

    Note: in a new version is implemented by aside.py

    what is doing: 
        using M: (M=N1 meanwhile) 
        get res_M at TEb: (test data of mnist)

        the specific n for MNIST is defined in all_baselines() 
    inp:
    out:
    how to run: 
        run all_baselines/stand_alone_runs.py
        in stand_alone_runs.py ensure assignmnet of overrid_train_data, eg 
            overrid_train_data = r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_1000\train\tens' 
    rem:
        read_test_data_from_standard_MNIST_data_location is used

all_baselines:
    by default operetes wuth dir_data where to loads/creates MNST/pkl/pkl files
    other args:
        overrid_train_data = None, read_test_data_from_standard_MNIST_data_location = None,\
        n_test_elts = None         
'''

assert not make_test_data_folder or eu_op == 'op3'

is_notebook = hasattr(builtins, "__IPYTHON__")
if is_notebook:
    if 1:  # with this displays as sep wnd in Jup
        # https://stackoverflow.com/questions/37365357/when-i-use-matplotlib-in-jupyter-notebook-it-always-raise-matplotlib-is-curren
        import matplotlib

        matplotlib.use('TkAgg')
    elif 0:
        # does not work in Jup
        #  UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
        pass
else:
    pass

if hconn:
    from imgaug import augmenters as iaa
    from imgaug.augmentables.batches import Batch
    from imgaug import imgaug
    import imgaug as ia

    imgaug.seed(1)
    #
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

plt.rcParams['figure.dpi'] = 200

normal_run = 0

randomize_descends = 0
display_lat_space = 0
lat_ranges = (-10, 10)

if blk:
    seed = blk_prs.seed
else:
    seed = 0
np.random.seed(seed=seed)
torch.manual_seed(seed)

init_to_digit = 0  # randomly init to digit or to random

# pick 20 (or batch_size) images, for each one im
#   pick 100 (nim1s) im1 in eps of im
#       ensure for all H(im1) eq_h H(m)
#           where eq_h is defined: if raznue lat, to 100, else: L2 of freqs
# TBD: distinguish train and test
# explore_resiliance()
test_resiliance = 0  # test resiliance in old fashion
tr = struct
tr.eps = 0.05
tr.nim1s = 20
# disp_res = 0/1 => call explore_resiliance()/disp_res()
tr.disp_res = 0

# display_lat_space = 1
display_attractors_which_person = 'p1'  # 'p1' or 'p2'
display_attractors_added_noise_level = None  # 0.0007#None#0.0007#0.001#if None or eg 0.0005
display_trace_started_with_yx = None  # (-10.,-20) #started with a point in the lat space, or None
assign_labs_to_attractors = 1  # for display_lat_space

n_bins_att_1d = 10000  # 10000

dbg9 = 0  # short mode for display_lat_space
if dbg9:
    n_samplings_for_fixed_points = 200
    n_bins_att_1d = 300

# qqq
assert test_resiliance + display_lat_space + normal_run + eu == 1

assert not make_test_data_folder or (op_batch and hconn), 'make_test_data_folder => op_batch and hconn'

randomize_steps = 0

dbg2 = 0  # disp
dbg4 = 0  # short explore_resiliance
dbg8 = 0  # save the images for every cycle
dbg9 = 0

desc_pars = struct()
# define addes noise unif at [0;eps]
# eps is decreased by relax_step
desc_pars.eps0 = 0.05  # 0.005
desc_pars.eps_relax_atstep = 0.5
desc_pars.ndesc_for_hist = 100  # #of it
desc_pars.nsteps_desc = 5000
desc_pars.thresh_hists_as_equal = 10e-5

T = 0.002  # thresh lat closure


train_labs = dict()

if eu:
    labs_TRa = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labs_TRb = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if eu_op == 'op1':
        train_labs['p1'] = labs_TRa
elif 1:
    train_labs['p1'] = [1, 3, 5, 7, 9]
    train_labs['p2'] = [0, 2, 4, 6, 8]
elif 0:
    train_labs['p1'] = [1]
    train_labs['p2'] = [0]
elif 0:
    train_labs['p1'] = []
    train_labs['p2'] = []

init_seeds_simple_examples = 1

n_steps_to_label_attractor_in_O2S_step = 0
assert n_steps_to_label_attractor_in_O2S_step == 0, 'TBD: to move n_steps_to_label_attractor_in_O2S_step processing ' \
                                                    'from __init__.conn to after optional training'

nsteps_p1_2_p2 = 1
nsteps_p2_2_p1 = 1

save_fig_interval = 50

dbg_nrm = 0

assert nsteps_p1_2_p2 == 1 and nsteps_p2_2_p1 == 1

# assert not hconn or not op_batch, 'hconn => not op_batch'

n_samplings_for_fixed_points = 2000

if display_trace_started_with_yx:
    n_samplings_for_fixed_points = 1

dir_root = os.path.split(os.path.realpath(__file__))[0]
dir_models = os.path.join(dir_root, 'models')
if not os.path.exists(dir_models):
    os.makedirs(dir_models)
dir_out = os.path.join(dir_root, 'out')
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
dir_data = os.path.join(dir_root, 'data')
if not os.path.exists(dir_data):
    os.makedirs(dir_data)

shutil.copyfile(os.path.join(dir_root, 'conn.py'), os.path.join(dir_out, 'conn.copy.txt'))


def get_mnist_data_dataloader(labs, train_data_mnist, overriden_location=None, sz_stri=None, shuffle = False):
    # labs is arr of labs;
    # loads and reads from dir_data/or reads from dir_data
    # if overriden_location, then reds db from there
    # if sz_stri = , say, 18x18, then scale

    if overriden_location:
        transforrms = [torch.tensor]
    else:
        transforrms = [tv.transforms.ToTensor()]

    if overriden_location:
        '''
        #Problem_2
        cant add above tv.transforms.Normalize to target_transform (under norm_images) 
        may be:

        Transform the dataset beforehand and save it to disk. You need to write a small code for that separately. And use this folder during training.
        https://stackoverflow.com/questions/73467584/can-i-pre-transform-images-when-training-a-deep-learning-model
        '''

        assert not sz_stri

    if sz_stri:
        assert sz_stri == '18x18'
        transforrms += [tv.transforms.Resize((18, 18))]

    transform = tv.transforms.Compose(transforrms)

    if overriden_location:
        data = tv.datasets.DatasetFolder(overriden_location, torch.load, target_transform=transform, extensions=('pt'))

        if not dbg20:
            # data.class_to_idx
            # {'0': 0,
            # ...
            #  '9': 9}
            for key in data.class_to_idx.keys():
                #print (data.class_to_idx[key])
                assert data.class_to_idx[key] == int(key)

        # by some unknown reason the assignment target_transform = torch.tensor
        # does not convert to data.targets to tensor thus do it by hand.
        # TBD: try fix it
        data.targets = torch.tensor(data.targets)

        my_dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle)
    else:
        #we want shuffle
        # make if not maked yet <raw> in dir_data and dowloade there train-images-idx3-ubyte,. train-images-idx3-ubyte.gz ets
        my_dataloader = torch.utils.data.DataLoader(
            tv.datasets.MNIST(dir_data, train=train_data_mnist,
                              transform=transform,
                              download=True),
            batch_size=batch_size,
            shuffle=True)

    tmptmp = 0
    if tmptmp:
        for images, labels in my_dataloader:
            break

    if labs != None and set(labs) != set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        assert not overriden_location, 'currently labs subsetting does not work for overriden_location, TBD '
        for i, lab in enumerate(labs):
            if i == 0:
                ii = (my_dataloader.dataset.targets == lab)
            else:
                ii = torch.logical_or(ii, my_dataloader.dataset.targets == lab)

        '''
        my_dataloader.dataset.targets 
            tensor([0, 0, 0, 0, 0
            tensor([5, 0, 4,  ..., 5, 6, 8])
        '''

        data = my_dataloader.dataset.data[ii]

        # torch.Tensor.type(data) 'torch.ByteTensor'
        data = data.to(torch.float)
        # torch.Tensor.type(data) 'torch.FloatTensor'

        data /= 255.

        # torch.Size([30508, 28, 28])
        data.unsqueeze_(1).shape
        # torch.Size([30508, 1, 28, 28])

        labels = my_dataloader.dataset.targets[ii]

        my_dataset = my_dataset_class(data, labels)
        my_dataloader2 = torch.utils.data.DataLoader(my_dataset, shuffle=True, batch_size=batch_size)

        ##x = get_x(my_dataloader2)[0]
        # x.shape torch.Size([128, 1, 28, 28])
        # d0a(t2i((get_x(my_dataloader2)[0])[0]))
        return my_dataloader2
    else:
        # torch.Tensor.type(get_x(my_dataloader)[0]) 'torch.FloatTensor'
        # d0a(t2i((get_x(my_dataloader)[0])[0]))
        return my_dataloader


def find_t(val, lst):
    # tesor form of find
    # find(5, [1, 2, 10]) == []
    # find(10, [1, 2, 10,10]) == [2,3]
    """
    lst = []
    lst.append(1)
    lst.append(2)
    lst.append(10)
    val = 3
    """

    ii = [i for (i, val1) in enumerate(lst) if torch.equal(val1, val)]
    return ii


def H(autoencoder, im, ndesc_for_hist, lat=[], eps=[]):
    # return histogram (distribution) of lat values for eps-descents
    # starting from an input im or from an input lat
    global glb
    if is_empty(eps):
        eps0 = desc_pars.eps0
    else:
        eps0 = eps
    if lat:
        assert is_empty(im)
        im = glb.p1.autoencoder.decode.forward(lat)

    atts = []
    for ndesc in range(ndesc_for_hist):
        if 0 and ndesc % 20 == 0:
            print('ndesc=' + str(ndesc))
        eps = eps0
        im2 = im
        for nstep in range(desc_pars.nsteps_desc):
            im2 = add_rand(im2, eps)
            eps *= desc_pars.eps_relax_atstep
            lat2 = glb.p1.autoencoder.encode.forward(im2)
            im2 = glb.p1.autoencoder.decode.forward(lat2)
            lat2 = glb.p1.autoencoder.encode.forward(im2)
        atts.append(lat2)

    tmp = 10
    # conv att to hist

    # hist.atts consist of attractors
    # hist.freqs consist of their frequences
    hist = struct()
    hist.atts = []
    hist.freqs = []
    for att in atts:
        ii = find_t(att, hist.atts)
        if ii:
            hist.freqs[ii[0]] += 1
        else:
            hist.atts.append(att)
            hist.freqs.append(1)
    # for att in atts:

    for n1 in range(len(hist.atts)):
        # print(np.arange(10,-1, -1)) [8 7 6 5 4 3 2 1 0]
        for n2 in range(len(hist.atts)):
            if not (n1 < n2):
                continue
            else:
                if hist.freqs[n2] > 0 and torch.norm(hist.atts[n1] - hist.atts[n2]) < desc_pars.thresh_hists_as_equal:
                    hist.freqs[n1] += hist.freqs[n2]
                    hist.freqs[n2] = 0
        # for n2 in range(len(hist.atts) ):
    # for n1 in range(len(hist.atts) ):

    atts = []
    freqs = []
    for (att, freq) in zip(hist.atts, hist.freqs):
        if freq > 0:
            freqs.append(freq)
            atts.append(att.detach().numpy())

    freqs = [freqs[i] / sum(freqs) for i in range(len(freqs))]
    hist = {'atts': atts, 'freqs': freqs}
    # print_hist(hist)
    return hist


# def H(autoencoder, im, ndesc_for_hist, lat = []):

def init_pars():
    pars = struct()

    pars.delay = 0.05  # Delay in seconds

    pars.val = 4

    pars.seed_digits = dict()
    pars.seed_digits['p1'] = ['1', '3', '5', '7', '9']
    pars.seed_digits['p2'] = ['0', '2', '4', '6', '8']
    pars.seeds_per_digit = 10

    # mean_strat, max_strat
    pars.strat = 'max_strat'

    return pars  # qa


if is_win:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # otherwise
    #   PyTorch no longer supports this GPU because it is too old
    #       https://discuss.pytorch.org/t/solved-pytorch-no-longer-supports-this-gpu-because-it-is-too-old/15444/6
    device = 'cpu'
else:
    device = 'cuda'


def cosid(x):
    return torch.cos(x) - x


class Network(nn.Module):
    def __init__(self, lat_dimension):
        # see solution PyTorch-Autoencoder-ConvolutionalNN.ipynb https://github.com/udacity/pytorchcrashcourse
        super(Network, self).__init__()

        if fconv:

            if blk:
                # Encoder

                # x5 conv256 strod 2 leakly rely

                krn = blk_prs.krn
                pad = blk_prs.pad
                self.conv1 = nn.Conv2d(1, 256, krn, stride=2, padding=pad)
                self.conv2 = nn.Conv2d(256, 256, krn, stride=2, padding=pad)
                self.conv3 = nn.Conv2d(256, 256, krn, stride=2, padding=pad)
                self.conv4 = nn.Conv2d(256, 256, krn, stride=2, padding=pad)
                if h2:
                    self.conv5 = nn.Conv2d(256, 10, krn, stride=2, padding=pad)
                else:
                    self.conv5 = nn.Conv2d(256, 256, krn, stride=2, padding=pad)

                # Decoder
                # x5 (conv256 strid 1 leaky rely -> bilinear)
                # x2 (conv256 strid 1 leaky rely)
                #    (conv3 strid 1  leaky rely)

                if h2:
                    self.upconv1 = nn.Conv2d(10, 256, krn, stride=1, padding=pad)
                else:
                    self.upconv1 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)
                self.upconv2 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)
                self.upconv3 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)
                self.upconv4 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)
                self.upconv5 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)

                self.Bupconv1 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)
                self.Bupconv2 = nn.Conv2d(256, 256, krn, stride=1, padding=pad)

                self.out = nn.Conv2d(256, 1, krn, stride=1, padding=pad)

            else:  # if blk / else

                assert False, 'github version in preparation'
            # if/else blk:

        else:  # if/else fconv
            if blk:
                # 2000 Examples, MNIST #wi 2048 depth 10
                # Adam, 1e-4 Default PyTorch 2
                wi = blk_prs.wi
                self.enc_linear1 = nn.Linear(len_mnist_patch, wi)
                self.enc_linear2 = nn.Linear(wi, wi)
                self.enc_linear3 = nn.Linear(wi, wi)
                self.enc_linear4 = nn.Linear(wi, wi)
                self.enc_linear5 = nn.Linear(wi, wi)
                self.enc_linear6 = nn.Linear(wi, wi)
                self.enc_linear7 = nn.Linear(wi, wi)
                self.enc_linear8 = nn.Linear(wi, wi)
                self.enc_linear9 = nn.Linear(wi, wi)
                self.enc_linear10 = nn.Linear(wi, lat_dimension)
                # Decoder
                self.dec_linear1 = nn.Linear(lat_dimension, wi)
                self.dec_linear2 = nn.Linear(wi, wi)
                self.dec_linear3 = nn.Linear(wi, wi)
                self.dec_linear4 = nn.Linear(wi, wi)
                self.dec_linear5 = nn.Linear(wi, wi)
                self.dec_linear6 = nn.Linear(wi, wi)
                self.dec_linear7 = nn.Linear(wi, wi)
                self.dec_linear8 = nn.Linear(wi, wi)
                self.dec_linear9 = nn.Linear(wi, wi)
                self.dec_linear10 = nn.Linear(wi, len_mnist_patch)

            else:  # if/else blk
                # super().__init__()
                # Encoder
                self.enc_linear1 = nn.Linear(len_mnist_patch, 512)
                self.enc_linear1_1 = nn.Linear(512, 512)
                self.enc_linear1_2 = nn.Linear(512, 512)
                self.enc_linear1_3 = nn.Linear(512, 512)
                if heavy_network:
                    self.enc_linear1_4 = nn.Linear(512, 512)
                    self.enc_linear1_5 = nn.Linear(512, 512)
                    self.enc_linear1_6 = nn.Linear(512, 512)
                self.enc_linear2 = nn.Linear(512, lat_dimension)

                # Decoder
                self.dec_linear1 = nn.Linear(lat_dimension, 512)
                self.dec_linear1_1 = nn.Linear(512, 512)
                self.dec_linear1_2 = nn.Linear(512, 512)
                self.dec_linear1_3 = nn.Linear(512, 512)
                if heavy_network:
                    self.dec_linear1_4 = nn.Linear(512, 512)
                    self.dec_linear1_5 = nn.Linear(512, 512)
                    self.dec_linear1_6 = nn.Linear(512, 512)
                self.dec_linear2 = nn.Linear(512, len_mnist_patch)
            # if/else blk:
        # if/else fconv:

        if h2:
            self.log_softmax = nn.functional.log_softmax

    # see solution PyTorch-Autoencoder-ConvolutionalNN.ipynb https://github.com/udacity/pytorchcrashcourse
    def encode2(self, x):
        if fconv:
            if blk:
                if skip:
                    assert False, 'last present in /July25.2023 B: removal of an autoencoder implementation/  ' \
                            ' see comments on skip definition'
                else:
                    x = cosid(self.conv1(x))
                    x = cosid(self.conv2(x))
                    x = cosid(self.conv3(x))
                    x = cosid(self.conv4(x))
                    if h2:
                        x = self.conv5(x)
                        x1 = cosid(x)
                        dig_out = self.log_softmax(x)  # torch.Size([8, 10, 1, 1])
                        dig_out = torch.squeeze(dig_out)  # torch.Size([8, 10])
                        return x1, dig_out
                    else:
                        x = cosid(self.conv5(x))
            else:
                assert False, 'github version in preparation'
        else:  # if fconv/else
            if blk:
                x = torch.flatten(x, start_dim=1)
                x = cosid(self.enc_linear1(x) + blk_prs.add)
                x = cosid(self.enc_linear2(x) + blk_prs.add)
                x = cosid(self.enc_linear3(x) + blk_prs.add)
                x = cosid(self.enc_linear4(x) + blk_prs.add)
                x = cosid(self.enc_linear5(x) + blk_prs.add)
                x = cosid(self.enc_linear6(x) + blk_prs.add)
                x = cosid(self.enc_linear7(x) + blk_prs.add)
                x = cosid(self.enc_linear8(x) + blk_prs.add)
                x = cosid(self.enc_linear9(x) + blk_prs.add)
                if h2:
                    x = self.enc_linear10(x)
                    x1 = cosid(x)
                    dig_out = self.log_softmax(x)
                    return x1, dig_out
                else:
                    x = cosid(self.enc_linear10(x))
            else:
                x = torch.flatten(x, start_dim=1)
                x = F.relu(self.enc_linear1(x) + 1)
                x = F.relu(self.enc_linear1_1(x) + 1)
                x = F.relu(self.enc_linear1_2(x) + 1)
                x = F.relu(self.enc_linear1_3(x) + 1)
                if heavy_network:
                    x = F.relu(self.enc_linear1_4(x) + 1)
                    x = F.relu(self.enc_linear1_5(x) + 1)
                    x = F.relu(self.enc_linear1_6(x) + 1)

                x = self.enc_linear2(x)
                if h2:
                    dig_out = self.log_softmax(x)
                    return x, dig_out
            # if/else blk:
        # if/else fconv:
        return x

    def decode2(self, z):
        if fconv:
            if blk:
                if skip:
                    assert False, 'last present in /July25.2023 B: removal of an autoencoder implementation/  ' \
                            ' see comments on skip definition'
                else:
                    z = cosid(self.upconv1(z))
                    z = nn.Upsample(scale_factor=2)(z)
                    z = cosid(self.upconv2(z))
                    z = nn.Upsample(scale_factor=2)(z)
                    z = cosid(self.upconv3(z))
                    z = nn.Upsample(scale_factor=2)(z)
                    z = cosid(self.upconv4(z))
                    z = nn.Upsample(scale_factor=2)(z)
                    z = cosid(self.upconv5(z))
                    # z = nn.Upsample(scale_factor=2)(z)
                    z = nn.Upsample(size=[28, 28])(z)

                    z = cosid(self.Bupconv1(z))
                    z = cosid(self.Bupconv2(z))
                    z = cosid(self.out(z))

                    tmp = 10
            else:
                assert False, 'github version in preparation'
        else:
            if blk:
                z = cosid(self.dec_linear1(z) + blk_prs.add)
                z = cosid(self.dec_linear2(z) + blk_prs.add)
                z = cosid(self.dec_linear3(z) + blk_prs.add)
                z = cosid(self.dec_linear4(z) + blk_prs.add)
                z = cosid(self.dec_linear5(z) + blk_prs.add)
                z = cosid(self.dec_linear6(z) + blk_prs.add)
                z = cosid(self.dec_linear7(z) + blk_prs.add)
                z = cosid(self.dec_linear8(z) + blk_prs.add)
                z = cosid(self.dec_linear9(z) + blk_prs.add)
                z = cosid(self.dec_linear10(z) + blk_prs.add)
                z = z.reshape((-1, 1, sz_mnist_patch, sz_mnist_patch))
            else:
                z = F.relu(self.dec_linear1(z) + 1)
                z = F.relu(self.dec_linear1_1(z) + 1)
                z = F.relu(self.dec_linear1_2(z) + 1)
                z = F.relu(self.dec_linear1_3(z) + 1)
                if heavy_network:
                    z = F.relu(self.dec_linear1_4(z) + 1)
                    z = F.relu(self.dec_linear1_5(z) + 1)
                    z = F.relu(self.dec_linear1_6(z) + 1)
                z = torch.sigmoid(self.dec_linear2(z))
                z = z.reshape((-1, 1, sz_mnist_patch, sz_mnist_patch))
        return z

    def forward(self, x):
        if h2:
            z1, dig_out = self.encode2(x)
            t = self.decode2(z1)
            return t, dig_out
        else:
            z = self.encode2(x)
            t = self.decode2(z)
            return t


def get_x(my_dataloader):
    for x, y in my_dataloader:
        break
    return x, y


def train(autoencoder, data, epochs, eu_op_db_spec, dir_interm_models):
    if not os.path.exists(dir_interm_models):
        os.makedirs(dir_interm_models)

    if blk:
        opt = torch.optim.Adam(autoencoder.parameters(), lr=blk_prs.lr)
    else:
        opt = torch.optim.Adam(autoencoder.parameters())
    losses = []
    n_epochs = []

    if tr_data_mt:
        mt_historical_info = dict()

    for epoch in range(epochs):
        losses_epoch = []
        for nbatch, (x, y) in enumerate(data):

            if dbg14:
                if nbatch == 10:
                    break

            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if randomize_descends:
                x += torch.randn(x.shape)

            if h2:  # https://nextjournal.com/gkoehler/pytorch-mnist
                (x_hat, dig_out) = autoencoder(x)
                assert (x.shape == x_hat.shape)
                loss1 = torch.mean(torch.linalg.norm(x - x_hat, dim=(2, 3)))
                loss2 = F.nll_loss(dig_out, y)

                if dbg13:
                    loss1 -= loss1
                loss = loss1 + loss2

                loss1_np = float(loss1.detach().cpu().numpy())
                loss2_np = float(loss2.detach().cpu().numpy())
                loss_np = float(loss.detach().cpu().numpy())

                if dbg14 or (epoch % 20 == 0 and nbatch % 10 == 0):
                    print(get_id_rmnist_dir(eu_op_db_spec) + ':' + \
                          ' epoch=' + str(epoch) + ' nbatch=' + str(nbatch) + \
                          ' loss1=' + Str(loss1_np, prec=3) + \
                          ' loss2=' + Str(loss2_np, prec=3) + \
                          ' loss=' + Str(loss_np, prec=3))
                    tmp = 10
            else:
                x_hat = autoencoder(x)
                assert (x.shape == x_hat.shape)
                loss = torch.mean(torch.linalg.norm(x - x_hat, dim=(2, 3)))
                loss_np = float(loss.detach().cpu().numpy())

                if dbg14 or (epoch % 20 == 0 and nbatch % 10 == 0):
                    stri = '  ' + datetime.datetime.now().strftime('%d.%m.%y,%H:%M:%S.%f')
                    print(get_id_rmnist_dir(eu_op_db_spec) + ':' + ' epoch=' + str(epoch) + ' nbatch=' + str(nbatch) + \
                          ' loss=' + Str(loss_np, prec=3) + stri)

            loss.backward()
            opt.step()
            losses_epoch.append(loss_np)

        # for nbatch, (x, y) in enumerate(data):
        n_epochs.append(epoch)

        loss_epoch = np.mean(losses_epoch)
        losses.append(loss_epoch)

        print('epoch=' + str(epoch) + ' loss_epoch=' + str(loss_epoch))

        if tr_data_mt:

            # may be will not wrire there
            fn = os.path.join(dir_interm_models, 'epoch' + str(epoch) + '_mt' + '.pth')

            cnd1 = (epoch == 0)
            cnd2 = (epoch > 0 and loss_epoch < mt_historical_info['best_loss'])
            if cnd1 or cnd2:
                if cnd2:
                    os.remove(mt_historical_info['best_model'])
                mt_historical_info['best_model'] = fn
                mt_historical_info['best_loss'] = loss_epoch
                torch.save(autoencoder.state_dict(), fn)

            f = open(os.path.join(dir_interm_models, 'mt_hist.txt'), 'wt' if epoch == 0 else 'at')
            stri = 'epoch = ' + str(epoch) + ' loss = ' + Str(loss_epoch) + \
                   ' model_fn: ' + fn
            f.write(stri + '\n')
            f.close()

        if epoch % save_loss_epochs == 0:
            fn = os.path.join(dir_interm_models, 'epoch' + str(epoch) + '.pth')
            torch.save(autoencoder.state_dict(), fn)

            np.save(os.path.join(dir_interm_models, 'n_epochs' + ('_mt' if tr_data_mt else '') + '.npy'), n_epochs)
            np.save(os.path.join(dir_interm_models, 'losses' + ('_mt' if tr_data_mt else '') + '.npy'), losses)

    # for epoch in range(epochs):

    tmp = 10
    return autoencoder


from torch.utils.data import Dataset


class my_dataset_class(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def visualize_batches():
    data = torch.utils.data.DataLoader(
        tv.datasets.MNIST(dir_data,
                          transform=tv.transforms.ToTensor(),
                          download=True),
        batch_size=128,
        shuffle=True)

    for ind, (x, y) in enumerate(data):
        if ind == 200:
            break
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(x[index].numpy().squeeze(), cmap='gray_r')


def actions_cnt(fig):
    if glb.cnt == 35:
        tmp = 10
    glb.cnt += 1
    if glb.cnt % save_fig_interval == 0:
        fig.savefig(dir_out + r'/gui_cnt_' + str(glb.cnt) + '.png')


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encode.forward(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


# plot_latent(autoencoder, data)

def plot_reconstructed(autoencoder, rx_inp=(-25, 25), ry_inp=(-25, 25), n_small_ims_per_coord=12, \
                       x_bin_s=[], y_bin_s=[], sz_bin_s=[], fn_png=[], lat_circle_s=[], \
                       trace_yx=None):
    # visualize the 2-d lat space by griding abd diaplayin  in every grid square
    # the image corresponding to the grid center
    # if y_bin_s is not empty,
    #   display in y_bin, v_bun a sircle wit the radius proportional to sz_bin
    #   (this option is for displayng fixed points, represented as y_bin_s, x_bin_s, sz_bin_s
    #
    # if trace_yx:
    #   plot_reconstructed receives trace_yx which is
    #   N points starting with trace_started_with_yx,
    #   trace_started_with_yx is displ in red
    #   other trace_yx points in magenta

    # displaying
    # ry x rx is rectangle, grided, calc im_small for the senter of any small recta,
    # put im_small to a large image using i,j
    # then disp the large im with labele annotation
    #
    #   ry x rx is rectangle
    #   the large im is a suare (n_small_ims_per_coord), axes annotation reflects diffrent ranges
    #   TBD: why not a square?
    #

    assert False, 'github version in preparation'


# def plot_reconstructed()


def get_fixed_point(rx=(-10, 10), ry=(-10, 10), trace_started_with_yx=None, im=[]):
    # randomly pick a random vector /lat/ in a 2-d lat space,
    # returns the fixed point (the result of convergence of lat)
    # or a circle (searching starting with N1 find repetition)
    # returns None if not found (for display_attractors_added_noise_level > 0)
    #
    # if trace_started_with_yx, simply return N points starting with trace_started_with_yx
    #
    global glb

    if trace_started_with_yx:
        assert lat_dimension == 2, 'meanwhile'
        # just observe some trace
        N = 50
        y = trace_started_with_yx[0]
        x = trace_started_with_yx[1]
    elif display_lat_space and lat_dimension > 2:
        N = 2000  # max n of points
        N1 = 1000  # starting from N1 search the circles
    else:
        N = 800  # max n of points
        N1 = 500  # starting from N1 search the circles

    lat = torch.Tensor(np.random.uniform(low=lat_ranges[0], high=lat_ranges[1], size=(lat_dimension)))

    if 1:
        lat_sav = lat.detach().numpy()
        lat_sav_sum = np.sum(lat_sav)
        print('lat_sav_sum=' + str(lat_sav_sum))

    circle_search_mode = 0
    circle_lats = []

    if trace_started_with_yx:
        retval = []
        retval.append([lat])
    else:
        retval = None
    for n in range(N):
        lat_prev = lat
        if display_attractors_which_person == 'p1':
            lat = glb.p1.autoencoder.encode2(glb.p1.autoencoder.decode2(lat))
        elif display_attractors_which_person == 'p2':
            lat = glb.p2.autoencoder.encode2(glb.p2.autoencoder.decode2(lat))

        if display_attractors_added_noise_level:
            lat += torch.tensor(
                np.random.uniform(-display_attractors_added_noise_level, display_attractors_added_noise_level, (1, 2)))

        if trace_started_with_yx:
            retval.append(lat)
        else:
            delta = torch.norm(lat_prev - lat)
            if torch.norm(delta) < T:
                retval = lat[0]
                if display_lat_space and lat_dimension > 2:
                    print('on exit from the loop n=' + str(n) + ' str(lat[0][0])+' + str(lat[0][0]) \
                          + ' delta =  ' + str(delta))
                break
            if n == N1:
                circle_search_mode = 1
                circle_lats.append(lat)
            elif circle_search_mode:
                assert n >= N1
                lat_in_circle = 0
                for nc in reversed(range(len(circle_lats))):
                    circle_lat = circle_lats[nc]
                    if torch.norm(circle_lat - lat) < T:
                        circle_lats = circle_lats[nc::]
                        lat_in_circle = 1
                        break
                if lat_in_circle:
                    retval = circle_lats
                    break
                else:
                    circle_lats.append(lat)
            # if circle_search_mode:
        # if/else trace_started_with_yx:

    # for n in range(N):

    if retval == None:

        if display_attractors_added_noise_level == None:
            tmp = 10

        # assert display_attractors_added_noise_level != None

    return retval


# def get_fixed_point():

def add_rand(lat, eps):
    lat2 = torch.clone(lat)
    lat2 += torch.rand(lat2.shape) * eps - eps / 2.
    return lat2


def print_hist(hst):
    for (att, freq) in zip(hst['atts'], hst['freqs']):
        print(str(att) + ' ' + str(freq))
    print('- - - - ')


def disp_res():
    global glb
    save_data = np.load(dir_out + '/' + 'hist_data.npy', allow_pickle=True).item()
    for n, im_elt in enumerate(save_data['im_elt_s']):
        print(n)

    # dict_keys(['nim', 'im', 'im1_hists'])
    if 0:
        """
        n=0
        [[-10.591725   8.386535]] 1.0

        n=6 
        - - - - 
        [[-0.6270979  0.0996547]] 0.98
        [[-10.591725   8.386535]] 0.02

        n=3 
        - - - - 
        [[-0.01997615 -0.01715906]] 1.0
        - - - - 
        [[-0.6270981   0.09965463]] 0.38
        [[-0.01997617 -0.01715907]] 0.62


        """
        # 0:9, 11, 12
        # 10
        #   [[0.04399448 - 2.4489336]]         0.99
        #   [[-1.8199754 - 3.9348106]]         0.01
        # linux: mefuzar

        for n in range(20):
            # n=1
            print('n=' + str(n))
            im_elt = save_data['im_elt_s'][n]
            # d0a(im_elt['im'])
            for hist_im1 in im_elt['im1_hists']:
                print_hist(hist_im1)
                break

        n = 8
        im_elt = save_data['im_elt_s'][n]
        d0a(im_elt['im'])

    tmp = 10


def explore_resiliance():
    global glb

    train_data_mnist = 0
    data = get_mnist_data_dataloader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train_data_mnist)  # CALL 1
    for ims, _ in data:
        break
    """        
    for nim, im in enumerate(ims):
        break
    """
    save_data = dict()
    save_data['desc_pars'] = desc_pars
    save_data['tr'] = tr
    save_data['im_elt_s'] = []
    for nim, im in enumerate(ims):
        if dbg4 and nim >= 2:
            break
        im_elt = dict()
        im_elt['nim'] = nim
        im_elt['im'] = t2i(im)
        im_elt['im1_hists'] = []
        print('nim=' + str(nim) + '----------------------')
        for nim1 in range(tr.nim1s):
            if dbg4 and nim1 >= 2:
                break
            print('    nim1=' + str(nim1))
            if nim1 == 0:
                im1 = add_rand(im, 0)
            else:
                im1 = add_rand(im, tr.eps)
                hist_im1 = H(glb.p1.autoencoder, im1, desc_pars.ndesc_for_hist)
            im_elt['im1_hists'].append(hist_im1)
            print_hist(hist_im1)
            tmp = 10
        # for nim1 in range(tr.nim1s):
        save_data['im_elt_s'].append(im_elt)
        np.save(dir_out + '/' + 'hist_data.npy', save_data)
    # for im in ims:
    # np.save(dir_out + '/' + 'hist_data.npy', save_data)

    return


# def explore_resiliance():


class Index:
    ind = 0

    def pauseresume(self, event):
        global glb
        glb.button_pauseresume_clicked = 1
        # plt.draw()

    def stop(self, event):
        global glb
        glb.button_stop_clicked = 1
        # plt.draw()


def init_graphisc():
    fig = plt.figure(figsize=(4.9, 3))
    ax_x = fig.add_subplot(224)
    ax_y = fig.add_subplot(223)
    ax_z = fig.add_subplot(221)
    ax_t = fig.add_subplot(222)

    plt.subplots_adjust(wspace=2)
    # fig.tight_layout()
    plt.ion()  # Turns interactive mode on (probably unnecessary)
    fig.show()  # Initially shows the figure
    plt.show()

    callback = Index()

    ax_pauseresume = plt.axes([0.55, 0.02, 0.2, 0.055])
    ax_stop = plt.axes([0.81, 0.02, 0.1, 0.055])

    btn_pauseresume = Button(ax_pauseresume, 'Pause/Resume')
    btn_pauseresume.label.set_fontsize('5')
    btn_pauseresume.on_clicked(callback.pauseresume)

    btn_stop = Button(ax_stop, 'Stop')
    btn_stop.label.set_fontsize('5')
    btn_stop.on_clicked(callback.stop)

    return fig, ax_x, ax_y, ax_z, ax_t, btn_pauseresume, btn_stop


def iter_mngr_p2_2_p1(pars, fig, ax, ax_txt, on_start=0):
    # access lat_t and transform to im_x
    #       using p1.lat_2_im:
    #
    #   disp im_t and im_x

    global glb

    button_stop_clicked = 0
    if on_start:
        pass
    else:
        paused = 0
        for j in range(nsteps_p2_2_p1):
            if glb.button_pauseresume_clicked:
                paused = 1 if paused == 0 else 0
                glb.button_pauseresume_clicked = 0
            if paused:
                while not glb.button_pauseresume_clicked:
                    plt.pause(pars.delay)
            else:
                update_axes(ax, ax_txt)
                glb.im_x = glb.p1.lat_2_im(pars, glb.lat_t, autoencoder=glb.p1.autoencoder)
                ax.imshow(t2i(glb.im_x), cmap='gray')
                actions_cnt(fig)
            plt.pause(pars.delay)
            button_stop_clicked = glb.button_stop_clicked
            if button_stop_clicked:
                break

    return button_stop_clicked


def iter_mngr_p1(pars, fig, ax_x, ax_y, ax_txt, ay_txt):
    # access im_x and performs nsteps_p1 iterations, at every it-n:
    #   transforms im_x to lat_y
    #       using O2S_step:
    #           encoder op + [n_steps_to_label_attractor_in_O2S_step]
    #   transforms lat_y to im_x;
    #       using lat_2_im:
    #           decode op
    #
    #   disp lat_y and im_x

    global glb

    button_stop_clicked = 0
    paused = 0
    for j in range(nsteps_p1):
        if glb.button_pauseresume_clicked:
            paused = 1 if paused == 0 else 0
            glb.button_pauseresume_clicked = 0
        if paused:
            while not glb.button_pauseresume_clicked:
                plt.pause(pars.delay)
        else:

            glb.lat_y, res_lab = glb.p1.O2S_step(pars, glb.im_x, autoencoder=glb.p1.autoencoder, key='p1')
            glb.im_x = glb.p1.lat_2_im(pars, glb.lat_y, autoencoder=glb.p1.autoencoder)

            if 0:  # dont perform O2S_step above for displaying  initial glb.im_x
                for k in range(10000):
                    if k % 1000 == 0:
                        print(k)
                    glb.lat_y, _, _ = glb.im_x, res_lab = glb.p1.O2S_step(pars, glb.im_x)

                pdf_prefix = 'tmp'
                suptitle = pdf_prefix
                d0a(t2i(glb.im_x), save_to_and_close_the_pdf=1, di_sav_dbg=dir_out, pdf_prefix=pdf_prefix,
                    suptitle=suptitle)
                tmp = 10

            update_axes(ax_x, ax_txt)
            stri = ay_txt + ' ' + 'res_lab=' + res_lab
            stri += '\n'
            stri2 = str(list(glb.lat_y.detach().numpy()))
            if dbg2:
                stri2 = 'iter_mngr_p1(): j=' + str(j) + ' cnt=' + str(glb.cnt) + stri2
                print(stri2)
            stri += ' lat_y=' + stri2
            update_axes(ax_y, stri)

            im_y = glb.p1.lat_2_im(pars, glb.lat_y, autoencoder=glb.p1.autoencoder)
            ax_y.imshow(t2i(im_y), cmap='gray')
            ax_x.imshow(t2i(glb.im_x), cmap='gray')
            actions_cnt(fig)
        plt.pause(pars.delay)
        button_stop_clicked = glb.button_stop_clicked
        if button_stop_clicked:
            break

        # fig.canvas.draw() # Draws the image to the screen
    # for j in range(nsteps_p1):

    return button_stop_clicked


def iter_mngr_p1_2_p2(pars, fig, ax, ax_txt):
    # access lat_y and transform to im_z
    #       using p1.lat_2_im:
    #
    #   disp lat_y and im_z

    global glb

    button_stop_clicked = 0
    paused = 0
    for j in range(nsteps_p1_2_p2):
        if glb.button_pauseresume_clicked:
            paused = 1 if paused == 0 else 0
            glb.button_pauseresume_clicked = 0
        if paused:
            while not glb.button_pauseresume_clicked:
                plt.pause(pars.delay)
        else:
            update_axes(ax, ax_txt)
            glb.im_z = glb.p1.lat_2_im(pars, glb.lat_y, autoencoder=glb.p1.autoencoder)
            ax.imshow(t2i(glb.im_z), cmap='gray')
            actions_cnt(fig)
        plt.pause(pars.delay)
        button_stop_clicked = glb.button_stop_clicked
        if button_stop_clicked:
            break

    return button_stop_clicked


def iter_mngr_p2(pars, fig, ax_z, ax_t, ax_txt, at_txt):
    # access im_z and performs nsteps_p2 iterations, at every it-n:
    #   transforms im_z to im_t
    #       using O2S_step:
    #           encode op + [n_steps_to_label_attractor_in_O2S_step]
    #   transforms im_t to im_z;
    #       using lat_2_im:
    #           decode op
    #
    #   disp im_t and im_z

    global glb

    button_stop_clicked = 0
    paused = 0
    for j in range(nsteps_p2):
        if glb.button_pauseresume_clicked:
            paused = 1 if paused == 0 else 0
            glb.button_pauseresume_clicked = 0
        if paused:
            while not glb.button_pauseresume_clicked:
                plt.pause(pars.delay)
        else:

            glb.lat_t, res_lab = glb.p2.O2S_step(pars, glb.im_z, autoencoder=glb.p2.autoencoder, key='p2')
            glb.im_z = glb.p2.lat_2_im(pars, glb.lat_t, autoencoder=glb.p2.autoencoder)

            update_axes(ax_z, ax_txt)
            stri = at_txt + ' ' + 'res_lab=' + res_lab
            stri += '\n'
            stri2 = str(list(glb.lat_t.detach().numpy()))
            if dbg2:
                stri2 = 'iter_mngr_p2(): j=' + str(j) + ' cnt=' + str(glb.cnt) + stri2
                print(stri2)
            stri += ' im_t=' + stri2
            update_axes(ax_t, stri)

            if glb.cnt == 400:
                tmp = 10

            im_t = glb.p2.lat_2_im(pars, glb.lat_t, autoencoder=glb.p2.autoencoder)
            ax_t.imshow(t2i(im_t), cmap='gray')
            ax_z.imshow(t2i(glb.im_z), cmap='gray')
            actions_cnt(fig)
        plt.pause(pars.delay)
        button_stop_clicked = glb.button_stop_clicked
        if button_stop_clicked:
            break

        # fig.canvas.draw() # Draws the image to the screen
    # for j in range(nsteps_p2):

    return button_stop_clicked


def update_axes(ax, txt):
    ax.clear()  # Clears the previous image
    # ax.title.set_text(txt)
    global glb
    ax.set_title(txt + '   ' + str(glb.cnt), fontsize=7)
    ax.set_xticks([]);
    ax.set_yticks([])


def run_conn(pars, fig, ax_x, ax_y, ax_z, ax_t):
    global glb
    on_start = 1
    while (1):
        print('cnt=' + str(glb.cnt))
        button_stop_clicked = iter_mngr_p2_2_p1(pars, fig, ax_x, 'x: observed by P1', on_start=on_start)
        if button_stop_clicked:
            break
        button_stop_clicked = iter_mngr_p1(pars, fig, ax_x, ax_y, 'x: observed by P1', 'y: seen by P1')
        if button_stop_clicked:
            break
        button_stop_clicked = iter_mngr_p1_2_p2(pars, fig, ax_z, 'z: observed by P2')
        if button_stop_clicked:
            break
        button_stop_clicked = iter_mngr_p2(pars, fig, ax_z, ax_t, 'z: observed by P2', 't: seen by P2')
        if button_stop_clicked:
            break

        on_start = 0

        # fig.canvas.draw() # Draws the image to the screen

        if button_stop_clicked:
            break
    # while(1):


# def run_conn():

def close_conn():
    plt.close('all')


def init_global_vars(pars, eu_op_db_spec):  # qq
    global glb
    glb = struct()
    glb.button_stop_clicked = 0
    glb.button_pauseresume_clicked = 0

    if init_to_digit:
        train_data_mnist = 0
        data = get_mnist_data_dataloader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train_data_mnist)  # CALL 2
        for images, _ in data:
            break
        n_init = np.random.randint(len(images))
        glb.im_x = images[n_init]  # torch.Size([1, 28, 28])
    else:
        glb.im_x = i2t(np.random.rand(sz_mnist_patch, sz_mnist_patch, 1).astype('float32'))
    # d0a(t2i(glb.im_x))
    # im = np.random.randint(7, size=(height,width))

    glb.p1 = Person(pars, 'p1', eu_op_db_spec)

    if eu:
        pass
    else:
        glb.p2 = Person(pars, 'p2', eu_op_db_spec)

    glb.cnt = 0


class Person():
    #   p_key = 'p1', 'p2'
    def __init__(self, pars, p_key, eu_op_db_spec):
        super(Person, self).__init__()
        global glb
        global train_labs

        self.autoencoder = Network(lat_dimension).to(device)

        stri = '_op1' if eu and eu_op in ['op1', 'op2', 'op3'] else ''

        stri += '_' + eu_scale if eu_scale else ''

        if eu_op in ['op2', 'op3', 'op2op3']:
            pass
        elif eu_op_db_spec:
            self.data = get_mnist_data_dataloader(None, None, overriden_location=eu_op_db_spec)
        else:
            train_data_mnist = 1
            self.data = get_mnist_data_dataloader(train_labs[p_key], train_data_mnist)
        '''
        for images, labs in self.data:
            break
        '''

        self.p_key = p_key

        if n_steps_to_label_attractor_in_O2S_step:
            # forming seeds_lat, seeds_lab for n_steps_to_label_attractor_in_O2S_step > 0
            #
            # first, form seeds[digit]
            #
            #
            seeds = dict()
            n_filled_lists = 0
            for digit in pars.seed_digits[p_key]:  # digit: '1
                seeds[digit] = []
            filled = 0
            train_data_mnist2 = 0
            data = get_mnist_data_dataloader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train_data_mnist2)  # CALL 4
            for ims, labs in data:
                for i, im in enumerate(ims):
                    # im torch.Size([1, 28, 28])
                    lab = labs[i]
                    lab = lab.detach().numpy()  # lab: 9
                    # im = t2i(im[0])#(28, 28, 1)
                    im_lat = self.autoencoder.encode2(im)
                    if 0 and dbg_nrm:  # nrm in lat
                        nrm = torch.norm(
                            self.autoencoder.encode.forward(self.autoencoder.decode.forward(im_lat)) - im_lat)
                        nrm = nrm.detach().numpy()
                        print('nrm=' + str(nrm))
                        if nrm > 0.1:
                            continue
                    if 0 and dbg_nrm:  # nrm in im space
                        nrm = torch.norm(im - self.autoencoder.decode(im_lat)[0])
                        nrm = nrm.detach().numpy()
                        print('nrm=' + str(nrm))
                        if nrm > 3:
                            continue
                    digit = str(lab)
                    if str(lab) in pars.seed_digits[p_key]:
                        if len(seeds[digit]) < pars.seeds_per_digit:
                            seeds[digit].append(im_lat)  # d0a(t2i(self.autoencoder.decode(im_lat)[0]))
                            # d0a(t2i(im),suptitle='im');d0a(t2i(self.autoencoder.decode(im_lat)[0]),suptitle='im=obs(seen(im))')
                            if i == 41 and int(digit) == 8:  # LAB_hard_attractors
                                # d0a(t2i(im))
                                # d0a(t2i(self.autoencoder.decode(  im_lat  )[0]), suptitle= 'autoencoder.decode(autoencoder.encode.forward(im))')
                                # d0a(t2i(self.autoencoder.decode(  self.autoencoder.encode.forward(im)  )[0]), suptitle= 'autoencoder.decode(autoencoder.encode.forward(im))')
                                # im_lat2 = self.autoencoder.encode.forward(self.autoencoder.decode.forward(im_lat))
                                # torch.norm(im_lat2-im_lat)
                                tmp = 10
                            if len(seeds[digit]) == 4 and int(digit) == 4:  # LAB_hard_attractors
                                tmp = 10
                                # d0a(t2i(im), suptitle = 'im') like 4
                                # d0a(t2i(self.autoencoder.decode(  self.autoencoder.encode.forward(im)  )[0]), suptitle= 'autoencoder.decode(autoencoder.encode.forward(im))') like 6
                            if len(seeds[digit]) == pars.seeds_per_digit:
                                n_filled_lists += 1
                                if n_filled_lists == len(pars.seed_digits[p_key]):
                                    filled = 1
                                    break
                # for i, im in enum(ims):
                if filled:
                    break
            # for ims, labs in data:
            assert filled
            #
            # reformat seeds[digit] to
            # seeds_lat, seeds_lab
            #
            seeds_lat = []
            seeds_lab = []
            for digit in seeds.keys():
                n_im_lat = 0
                for n_im_lat in range(len(seeds[digit])):
                    im_lat = seeds[digit][n_im_lat]
                    # d0a(t2i(autoencoder.decode(im_lat)[0]))
                    seeds_lat.append(im_lat)
                    seeds_lab.append(int(digit))
            self.seeds_lat = seeds_lat
            self.seeds_lab = seeds_lab
        # if n_steps_to_label_attractor_in_O2S_step:

        if p_key == 'p1':
            self.eps_p1 = desc_pars.eps0
        elif p_key == 'p2':
            self.eps_p2 = desc_pars.eps0

        return

    # def __init__(self, pars, p_key):

    def O2S_step(self, pars, im, autoencoder=[], key=[]):
        # inp: im: torch.Size([1, 28, 28])
        # convert to latent
        # then optionally n_steps_to_label_attractor_in_O2S_step
        #   way 1: find lat_i_m by enum //meanwhile
        #   way 2: end is via gradi

        if glb.cnt == 35:
            tmp = 10
        # d0a(t2i(im))
        lat = autoencoder.encode2(im)  # tensor([[14.3086, -3.0289]], grad_fn=<AddmmBackward>)

        if randomize_steps:

            if glb.cnt > 1000:
                tmp = 10
            if key == 'p1':
                lat = add_rand(lat, self.eps_p1)
                self.eps_p1 *= desc_pars.eps_relax_atstep
            elif key == 'p2':
                lat = add_rand(lat, self.eps_p2)
                self.eps_p2 *= desc_pars.eps_relax_atstep

        if n_steps_to_label_attractor_in_O2S_step > 0:
            for n_step_to_attractor in range(n_steps_to_label_attractor_in_O2S_step):
                att_x_s = []
                vf_s = []
                f_s = []
                for seed_lat in self.seeds_lat:  # tensor([[  9.2365, -19.0491]], grad_fn=<AddmmBackward>)
                    d = abs(seed_lat - lat)
                    d = torch.norm(d)  # tensor(20.4395, grad_fn=<CopyBackwards>)
                    # v = e_(lat, seed_lat) exp(-)
                    lambd = 20.
                    d /= lambd
                    # print(d)
                    force = torch.exp(-d)
                    v = seed_lat - lat
                    att_x_s.append(v)
                    v /= torch.norm(v)
                    vf = v * force
                    f_s.append(force.detach().numpy())
                    vf_s.append(vf)
                # for seed_lat in self.seeds_lat: #tensor([[  9.2365, -19.0491]], grad_fn=<AddmmBackward>)
                if pars.strat == 'max_strat':
                    ind = np.argmax(f_s)
                    vf_a = vf_s[ind]  # vector to attractor, mult by force
                    v_a_maxnorm = att_x_s[ind] / 10  # shrt vector to attractor cant be jumpover
                    if torch.norm(vf_a) > torch.norm(v_a_maxnorm):
                        vf_a = v_a_maxnorm
                    res_lab = str(self.seeds_lab[ind])
                elif pars.strat == 'mean_strat':
                    vf_a = torch.mean(torch.squeeze(torch.stack(vf_s)), 0)
                    assert False, 'TBD: v_a_maxnorm'

                lat = lat + vf_a
                # d0a(  t2i(self.autoencoder.decode.forward(lat)[0]) )
                if glb.cnt == 35:
                    # ind 41
                    # res_lab 8
                    # d0a(  t2i(self.autoencoder.decode.forward(  self.seeds_lat[ind]     )[0]) ) like 3
                    # d0a(  t2i(self.autoencoder.decode.forward(lat)[0]) )
                    tmp = 10
            # for n_step_to_attractor in range(n_steps_to_label_attractor_in_O2S_step):
        else:
            res_lab = 'none'
        # if/else n_steps_to_label_attractor_in_O2S_step:

        return lat[0], res_lab

    # def O2S_step(self, pars, im):

    def lat_2_im(self, pars, lat, autoencoder=[]):
        # inp: latent
        # convert to im
        # im: torch.Size([1, 28, 28])

        im = autoencoder.decode2(lat)

        return im[0]

    # def lat_2_im(self, pars, lat, autoencoder=[]):


# class Person():

def get_aug_geom(alpha=1):
    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.04 * alpha)),  # random crops
        iaa.Affine(
            scale={"x": (1 - 0.3 * alpha, 1 + 0.3 * alpha), "y": (1 - 0.3 * alpha, 1 + 0.3 * alpha)},
            translate_percent={"x": (-0.1 * alpha, 0.1 * alpha), "y": (-0.1 * alpha, 0.1 * alpha)},
            rotate=(-10 * alpha, 10 * alpha),
            shear=(-6 * alpha, 6 * alpha)
        )
    ], random_order=True)  # apply augmenters in random order
    return seq


def get_aug_radiom(alpha=1):
    seq__radio_affine = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.6 * alpha))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((1 - 0.25 * alpha, 1 + 0.5 * alpha)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * alpha), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((1 - 0.2 * alpha, 1 + 0.2 * alpha), per_channel=0.2 * alpha),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True)  # apply augmenters in random order
    return seq__radio_affine


def get_aug(alpha=1):
    seq__radio_affine = iaa.Sequential([
        iaa.Crop(percent=(0, 0.04 * alpha)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.6 * alpha))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((1 - 0.25 * alpha, 1 + 0.5 * alpha)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * alpha), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((1 - 0.2 * alpha, 1 + 0.2 * alpha), per_channel=0.2 * alpha),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (1 - 0.3 * alpha, 1 + 0.3 * alpha), "y": (1 - 0.3 * alpha, 1 + 0.3 * alpha)},
            translate_percent={"x": (-0.1 * alpha, 0.1 * alpha), "y": (-0.1 * alpha, 0.1 * alpha)},
            rotate=(-10 * alpha, 10 * alpha),
            shear=(-6 * alpha, 6 * alpha)
        )
    ], random_order=True)  # apply augmenters in random order
    return seq__radio_affine


def get_aug_h(alpha=1):
    # print('alpha ='+str(alpha))
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5 * alpha),  # horizontally flip 50% of all images
            iaa.Flipud(0.2 * alpha),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (1 - 0.2 * alpha, 1 + 0.2 * alpha), "y": (1 - 0.2 * alpha, 1 + 0.2 * alpha)},
                translate_percent={"x": (-0.2 * alpha, 0.2 * alpha), "y": (-0.2 * alpha, 0.2 * alpha)},
                rotate=(-45 * alpha, 45 * alpha),
                shear=(-16 * alpha, 16 * alpha),
                order=[0, 1],
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0 * alpha)),
                               iaa.AverageBlur(k=(2, 7))
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * alpha), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01 * alpha, 0.1 * alpha), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03 * alpha, 0.15 * alpha), size_percent=(0.02 * alpha, 0.05 * alpha),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05 * alpha, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10 / 255. * alpha, 10 / 255. * alpha), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((1 - 0.5 * alpha, 1 + 0.5 * alpha), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((1 - 0.5 * alpha, 1.5 + 0.5 * alpha), per_channel=0.5),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(2 - 0.5 * alpha, 2 + 0.5 * alpha), sigma=0.25 * alpha)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01 * alpha, 0.05 * alpha)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    return seq


def map_examples_to_attractors(n_run, eu_op_loc):  # lab_hgkfdsjgtrklsg

    assert eu_op_loc in ['op2', 'op3']

    dir_data_run = os.path.join(dir_data, 'run' + str(n_run))
    if not os.path.exists(dir_data_run):
        os.makedirs(dir_data_run)

    if eu_op_loc == 'op2':
        dir_data_out_png = os.path.join(os.path.join(dir_data_run, 'ATRb'), 'png')
        dir_data_out_tens = os.path.join(os.path.join(dir_data_run, 'ATRb'), 'tens')
    if eu_op_loc == 'op3':
        if make_test_data_folder:
            dir_data_out_png = os.path.join(mnist_test_data_folder, 'png')
            dir_data_out_tens = os.path.join(mnist_test_data_folder, 'tens')
        else:
            dir_data_out_png = os.path.join(os.path.join(dir_data_run, 'ATEb'), 'png')
            dir_data_out_tens = os.path.join(os.path.join(dir_data_run, 'ATEb'), 'tens')
    if os.path.exists(dir_data_out_png):
        if dbg11:
            pass
        else:
            assert False, print('existing folder: ' + dir_data_out_png)
    else:
        os.makedirs(dir_data_out_png)
    if os.path.exists(dir_data_out_tens):
        if dbg11:
            pass
        else:
            assert False, print('existing folder: ' + dir_data_out_tens)
    else:
        os.makedirs(dir_data_out_tens)

    num_overall = 0
    break_from_loops = 0

    if 0 and dbg21:
        all_labs = []
        for nb, (ims, labs) in enumerate(glb.data_feed_autoenc):
            num_array_list = [lab.numpy() for lab in labs]
            integer_list = [int(item) for item in num_array_list]
            all_labs += integer_list
        #len(all_labs)
        all_labs = np.array(all_labs)
        #len(np.argwhere(all_labs==5))
        # == 892
        #
        # array([9, 3, 2, 5]) array([6, 4, 9, 0])
        tmp=10

    '''
    Lab_on_the_same_order_of_test_data_enumeration: 
        in enumerate(glb.data_feed_autoenc),
        even after initialization
            glb.data_feed_autoenc =  get_mnist_data_dataloader
                my_dataloader = torch.utils.data.DataLoader(
                    tv.datasets.MNIST(dir_data, train=train_data_mnist,
                                      transform=transform,
                                      download=True),
                    batch_size=batch_size,
                    shuffle=True)
        it guaranteed 
            that enumerate(glb.data_feed_autoenc)
            returns all mnist test data 
        but it is not guaranteed
            that the shuffled order of enumeratin is the same.
        to provide the same order we perform
            np.random.seed(seed=seed)
            torch.manual_seed(seed)
        before enumerate(glb.data_feed_autoenc)        
        we do it before actual enumeration
    '''
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    nimabs = 0
    for nb, (ims, labs) in enumerate(glb.data_feed_autoenc):  # torch.Size([8, 1, 28, 28])

        if dbg14 or (1 and dbg21):
            if nb == 2:
                break

        if op_batch:
            if make_test_data_folder:
                pass
            elif hconn:
                for nstep_p1 in range(nsteps_p1):
                    print('n_run=' + str(n_run) + ' nb=' + str(nb) + ' of ' + str(
                        len(glb.data_feed_autoenc)) + ' nstep_p1=' + str(nstep_p1))
                    if nstep_p1 == 0:
                        ims2 = ims.cpu().detach().numpy()  # (128, 1, 28, 28)
                        ims2a = np.expand_dims(ims2, 4)  # (128, 1, 28, 28, 1)
                        ims2_tile = np.tile(ims2a, amHist)  # 128, 1, 28, 28, 1000)
                        ims2_tile = np.transpose(ims2_tile, (0, 4, 2, 3, 1))
                        # ims2_tile.shape numpy.ndarray (2, 200, 28, 28, 1) for batch_size = 1
                        sh = ims2_tile.shape

                    if nstep_p1 < k_h:

                        ims2_tile = ims2_tile.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])  # (128000, 28, 28, 1)

                        if hconn_simulatenonhconn:
                            tile_processed = ims2_tile
                        else:
                            # ims2_tile.shape (1, 28, 28, 1)
                            aug = get_aug(alpha * betha ** (-nstep_p1))
                            micro_batch = Batch(images=ims2_tile)
                            micro_batch_processed = aug.augment_batch(micro_batch)
                            tile_processed = micro_batch_processed.images_aug
                            # tile_processed.shape (1, 28, 28, 1)
                        # (32000, 28, 28, 1)
                        tile_processed = tile_processed.reshape(sh[0], sh[1], sh[2], sh[3],\
                                sh[4])  # (128, 1000, 28, 28, 1)

                        # d0a(tile_processed[3,511,:,:,:]);d0a(tile_processed[3,530,:,:,:]) the same pic
                        disp1 = 0
                        if disp1:
                            # micro_batch_processed.images_aug[0:10].shape = (10, 28, 28, 1)
                            imgaug.show_grid(micro_batch_processed.images_aug[0:10], rows=4, cols=5)

                        # convert #(8, 1000, 28, 28, 1) to [8*1000, 1, 28, 28]
                        images_aug2 = tile_processed.transpose((0, 1, 4, 2, 3))
                        # (8, 1000, 1, 28, 28)
                        images_aug2 = images_aug2.reshape(sh[0] * sh[1], sh[4], sh[2], sh[3])
                        # (8000, 1, 28, 28)

                        images_aug3 = torch.from_numpy(images_aug2)  # torch.Size([800, 1, 28, 28])

                        # encode2 gets torch.Size([128, 1, 28a, 28b])
                        if is_win == 0:
                            images_aug3 = images_aug3.cuda()
                        # images_aug3.shape torch.Size([200, 1, 28, 28]) torch.Tensor
                        if h2:
                            ims3 = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(images_aug3)[0])
                        else:
                            ims3 = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(images_aug3))

                        # ims3.shape torch.Size([400, 1, 28, 28])

                        ims2_tile = ims3.cpu().detach().numpy().reshape(sh[0], sh[1], sh[2], sh[3], sh[4])
                        # ims2_tile.shape numpy.ndarray (2, 200, 28, 28, 1) for batch_size = 2
                    else:
                        assert False, "TBD, entering this point invokes memory problem"
                        ims3 = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(ims3))
                    tmp = 10
                # for nstep_p1 in range(nsteps_p1):

                # torch.Size([800, 1, 28, 28])
                ims3 = ims3.cpu().detach().numpy()
                disp2 = 0
                if disp2:
                    # as above
                    ims3a = np.swapaxes(ims3, 1, 3)
                    ims3b = np.swapaxes(ims3a, 1, 2)
                    imgaug.show_grid(ims3b[0:100], rows=10, cols=10)

                ims3 = ims3.reshape(sh)  # (8, 100, 28, 28, 1)
                ims3 = np.mean(ims3, axis=1, keepdims=1)  # (8, 1, 28, 28, 1)
                ims3 = np.squeeze(ims3, axis=1)  # (8, 28, 28, 1)
                ims3 = ims3.reshape(sh[0], sh[4], sh[2], sh[3])

                ims = torch.from_numpy(ims3)  # torch.Size([8, 1, 28, 28])
                tmp = 10

            else:  # eif/else hconn
                for nstep_p1 in range(nsteps_p1):
                    ims = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(ims))  # torch.Size([8, 1, 28, 28])

            for nim, im in enumerate(ims):
                # im torch.Size([1, 28, 28])
                # lab tensor(0)
                lab = labs[nim]
                dir_png_lab = dir_data_out_png + '/' + str(int(lab)) + '/'
                if not os.path.exists(dir_png_lab):
                    os.makedirs(dir_png_lab)
                fn = dir_png_lab + 'nimabs=' + str(nimabs) + '.nb=' + str(nb) + '.nim=' + str(nim) + '.png'

                tv.io.write_png(i2t(i2d(t2i(im))), fn)
                dir_tens_lab = dir_data_out_tens + '/' + str(int(lab)) + '/'
                if not os.path.exists(dir_tens_lab):
                    os.makedirs(dir_tens_lab)
                fn = dir_tens_lab + 'nimabs=' + str(nimabs) + '.nb=' + str(nb) + '.nim=' + str(nim) + '.pt'
                torch.save(im, fn)
                num_overall += 1

                if eu_op_loc == 'op3' and num_overall == op3_num_total:
                    break_from_loops = 1
                    break

                nimabs += 1

            # for nim, im in enumerate(ims):

            if break_from_loops:
                break

        else:  # if/else op_batch
            print('nb=' + str(nb))

            for (nim, im) in enumerate(ims):

                if dbg14:
                    if nim > 0:
                        break

                print('nb=' + str(nb) + ' nim=' + str(nim) + \
                      ' num_overall=' + str(num_overall))
                # im torch.Size([1, 28, 28])
                # lab tensor(0)
                lab = labs[nim]
                if hconn:
                    if 0:  # one aug at sing image
                        # im=ims[10]
                        # d0a(t2i(im)), d0a(im2)
                        for nstep_p1 in range(nsteps_p1):
                            # im is tensor, im2 is np arr
                            aug = get_aug(alpha * 2 ** (-nstep_p1))
                            im2 = aug(images=[t2i(im)])[0].copy()
                            im = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(i2t(im2)))[0]
                    elif 1:
                        # augs at propagated imagew (mini
                        # im=ims[10]
                        # d0a(t2i(im)), d0a(im2)

                        # torch.Size([1, 28, 28])
                        imnp = im.detach().numpy()
                        ims = []
                        for i in range(amHist):
                            ims.append(imnp)
                        ims = np.array(ims)

                        for nstep_p1 in range(nsteps_p1):
                            # print('nstep_p1='+str(nstep_p1))
                            if dbg_get_aug_geom:
                                aug = get_aug_geom(alpha * betha ** (-nstep_p1))
                            elif dbg_get_aug_radiom:
                                aug = get_aug_radiom(alpha * betha ** (-nstep_p1))
                            elif aug_h:
                                alpha_1 = alpha * betha ** (-nstep_p1)
                                if alpha_1 < 0.00015:
                                    break
                                else:
                                    aug = get_aug_h(alpha_1)
                            else:
                                aug = get_aug(alpha * betha ** (-nstep_p1))

                            # ims.shape (1000, 1, 28a, 28b)
                            ims2 = np.swapaxes(ims, 1, 3)
                            # (4, 28b, 28a, 1)
                            ims3 = np.swapaxes(ims2, 1, 2)
                            # (4, 28a, 28b, 1)
                            # imgaug.show_grid(ims3[0:100], rows=10, cols=10)
                            micro_batch = Batch(images=ims3)

                            # https://imgaug.readthedocs.io/en/latest/source/api_imgaug.html
                            # aug is applied to images of (n, 298, 447, 3)
                            micro_batch_processed = aug.augment_batch(micro_batch)

                            disp1 = 0
                            if disp1:
                                # micro_batch_processed.images_aug[0:10].shape = (10, 28, 28, 1)
                                imgaug.show_grid(micro_batch_processed.images_aug[0:10], rows=4, cols=5)

                            # micro_batch_processed.images_aug.shape
                            #    Out[15]: (1000, 28a, 28b, 1)

                            images_aug2 = micro_batch_processed.images_aug.transpose((0, 3, 1, 2))
                            # 0<- 0
                            # 1<- 3
                            # 2<- 1
                            # 3<- 2
                            images_aug3 = torch.from_numpy(images_aug2)

                            # torch.Size([1000, 1, 28, 28]
                            # encode2 gets torch.Size([128, 1, 28a, 28b])

                            ims3 = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(images_aug3))
                            # torch.Size([1000, 1, 28, 28])

                            ims4 = ims3.detach().numpy()
                            # (1000, 1, 28, 28)

                            disp2 = 0
                            if disp2:
                                # as above
                                ims4a = np.swapaxes(ims4, 1, 3)
                                ims4b = np.swapaxes(ims4a, 1, 2)
                                imgaug.show_grid(ims4b[0:100], rows=10, cols=10)

                            ims = ims4.copy()

                        # for nstep_p1 in range(nsteps_p1):

                        # ims4.shape (1000, 1, 28, 28)
                        # np.mean(ims4, axis=0).shape (1, 28, 28)

                        if averaging == 'mean':
                            im = np.mean(ims, axis=0)
                        elif averaging == 'median':
                            im = np.median(ims, axis=0)
                        elif averaging == 'fro':
                            im = np.linalg.norm(ims, axis=0)
                        # im.shape (1, 28, 28)

                        im = torch.from_numpy(im)
                        if 0:
                            imdisp = i2d(t2i(im))
                            # (28, 28, 1)
                            d0a(imdisp)
                        tmp = 10

                    # d0a(t2i(im)); d0a(im2)
                else:  # if/else hconn
                    assert False, 'op_batch:=0 in hconn:=0 mode is deprecated'
                    for nstep_p1 in range(nsteps_p1):
                        im = glb.p1.autoencoder.decode2(glb.p1.autoencoder.encode2(im))[0]

                # here im.shape = torch.Size([1, 28, 28])
                dir_png_lab = dir_data_out_png + '/' + str(int(lab)) + '/'
                if not os.path.exists(dir_png_lab):
                    os.makedirs(dir_png_lab)
                fn = dir_png_lab + 'nb=' + str(nb) + '.nim=' + str(nim) + '.png'
                tv.io.write_png(i2t(i2d(t2i(im))), fn)
                dir_tens_lab = dir_data_out_tens + '/' + str(int(lab)) + '/'
                if not os.path.exists(dir_tens_lab):
                    os.makedirs(dir_tens_lab)
                fn = dir_tens_lab + 'nb=' + str(nb) + '.nim=' + str(nim) + '.pt'
                torch.save(im, fn)
                num_overall += 1
                if eu_op_loc == 'op3' and num_overall == op3_num_total:
                    break_from_loops = 1
                    break
            # for (nim, im) in enumerate(ims):
            if break_from_loops:
                break
        # if/else op_batch:

        tmp=10

    # for nb, (ims, labs) in enumerate(glb.data_feed_autoenc):

    tmp=10


# def map_examples_to_attractors(n_run, eu_op_loc):

def disp_lat_space():
    ## create arr of fix pints wth repetitions att_y_s,att_x_s,
    #  create the bins:
    #       y_bin_s,x_bin_s - bin centers
    #  splash the arr of fix points by the bins

    ## will build y_bin_s, x_bin_s, sz_bin_s

    display_attractors = 1
    if display_attractors:

        # att_y_s,att_x_s - y,x of list fixed points with repetitions
        fixed_lats_s = []
        lat_circle_s = []
        for n in range(n_samplings_for_fixed_points):
            print('convey attractors: sample fixed point n=' + str(n) + ' of ' + str(n_samplings_for_fixed_points))
            im = []

            ret = get_fixed_point(trace_started_with_yx=display_trace_started_with_yx, im=im)
            if ret == None:
                continue
            if display_trace_started_with_yx:
                # N points starting with display_trace_started_with_yx
                trace_yx = [elt[0].detach().numpy() for elt in ret]
            elif isinstance(ret, list):  # ret is lsi of lats lat
                lat_circle_s.append([elt[0].detach().numpy() for elt in ret])
            else:  # ret is lat
                fixed_lats_s.append(ret.detach().numpy())

        if display_trace_started_with_yx:

            lat_circle_s = np.load(dir_out + '/' + 'lat_circle_s.npy', allow_pickle=True).tolist()
            lat_min_x = np.load(dir_out + '/' + 'lat_min_x.npy', allow_pickle=True).item()
            lat_max_x = np.load(dir_out + '/' + 'lat_max_x.npy', allow_pickle=True).item()
            lat_min_y = np.load(dir_out + '/' + 'lat_min_y.npy', allow_pickle=True).item()
            lat_max_y = np.load(dir_out + '/' + 'lat_max_y.npy', allow_pickle=True).item()
            x_bin_s = np.load(dir_out + '/' + 'x_bin_s.npy', allow_pickle=True).tolist()
            y_bin_s = np.load(dir_out + '/' + 'y_bin_s.npy', allow_pickle=True).tolist()
            sz_bin_s = np.load(dir_out + '/' + 'sz_bin_s.npy', allow_pickle=True).tolist()

            fn = dir_out + r'/attractors_' + str(len(np.argwhere(sz_bin_s))) + '_points_' + \
                 str(len(lat_circle_s)) + '_cls_' + \
                 str(epochs) + '_ephs_' + \
                 display_attractors_which_person

            if display_attractors_added_noise_level:
                fn += '_noise_' + str(display_attractors_added_noise_level)

        else:
            # eg
            #   att_y_s = [-0.6271177, 15.54361, 31.389591, 1.1957554, 0.045362953, 0.04401153, 0.0439755, 0.11923752, 0.012488337, -0.01999264]
            #   att_x_s = [0.0996438, -11.402555, 18.527416, 2.6584833, -0.10261989, -2.4489279, -2.4489472, 0.085783154, -1.2719728, -0.017169148]

            if lat_dimension == 2:

                att_y_s = [lat[0] for lat in fixed_lats_s]
                att_x_s = [lat[1] for lat in fixed_lats_s]

                lat_min_y = np.min(att_y_s)  # qqq
                lat_max_y = np.max(att_y_s)
                lat_min_x = np.min(att_x_s)
                lat_max_x = np.max(att_x_s)

                for lat_circle in lat_circle_s:
                    for lat in lat_circle:
                        lat_min_y = np.min([lat_min_y, lat[0]])
                        lat_max_y = np.max([lat_max_y, lat[0]])
                        lat_min_x = np.min([lat_min_x, lat[1]])
                        lat_max_x = np.max([lat_max_x, lat[1]])
                lat_min_y -= 1
                lat_max_y += 1
                lat_min_x -= 1
                lat_max_x += 1

                if att_y_s:
                    assert np.min(att_y_s) > lat_min_y
                    assert np.max(att_y_s) < lat_max_y
                    assert np.min(att_x_s) > lat_min_x
                    assert np.max(att_x_s) < lat_max_x

                bin_sz_y = (lat_max_y - lat_min_y) / n_bins_att_1d
                bin_sz_x = (lat_max_x - lat_min_x) / n_bins_att_1d
                assert n_bins_att_1d == len(np.arange(lat_min_y, lat_max_y, bin_sz_y))
                assert n_bins_att_1d == len(np.arange(lat_min_x, lat_max_x, bin_sz_x))
            else:
                assert False, 'exclde non-MIT code'

            print('construct attractor bins')
            # [x_row;
            # ...
            # x_row]
            x_row = np.arange(lat_min_x, lat_max_x, (lat_max_x - lat_min_x) / n_bins_att_1d) + bin_sz_x / 2
            arr1 = np.zeros(n_bins_att_1d)
            x_bin_s = arr1[:, None] + x_row
            x_bin_s = x_bin_s.flatten()

            # duplicate cols
            # [y_col,y_col...y_col]
            y_col = np.arange(lat_min_y, lat_max_y, (lat_max_y - lat_min_y) / n_bins_att_1d) + bin_sz_y / 2
            arr2 = np.zeros(n_bins_att_1d)
            y_bin_s = y_col[:, None] + arr2
            y_bin_s = y_bin_s.flatten()
            # np.array_equal(y_bin_s_NEW, np.array(y_bin_s))

            sz_bin_s = np.zeros(n_bins_att_1d * n_bins_att_1d)

            tmp = 10
            # splash att's by the bins, every time increasing the respective sz_bin   by 1
            for n_p in range(len(att_y_s)):
                print('splash atts by the bins: n_p=' + str(n_p) + ' of ' + str(len(att_y_s)))
                u = att_y_s[n_p]
                v = att_x_s[n_p]
                # by u,v find n_bin_y,n_bin_x
                n_bin_y = int((u - lat_min_y) / bin_sz_y)
                n_bin_x = int((v - lat_min_x) / bin_sz_x)

                n_bin2 = n_bin_y * n_bins_att_1d + n_bin_x
                sz_bin_s[n_bin2] += 1

                if display_attractors_added_noise_level == None:
                    if sys.platform == 'win32':
                        pass  # TBD why fails at win?
                    else:
                        # display_attractors_added_noise_level == None the u and v are not necess aligned with n_bin2
                        #   since get_fixed_point does not necess return fixed pointv may return None or a cicle
                        cnd1 = y_bin_s[n_bin2] - bin_sz_y / 2 <= u and u <= y_bin_s[n_bin2] + bin_sz_y / 2
                        cnd2 = x_bin_s[n_bin2] - bin_sz_x / 2 <= v and v <= x_bin_s[n_bin2] + bin_sz_x / 2
                        assert (cnd1 and cnd2)

            # normalize sz_bin_s for displaying
            #
            # n_samplings_for_fixed_points = 10000 -> div by 20
            # ==> k = 1/(20 *  n_samplings_for_fixed_points/10000)
            k = 1 / (20. * n_samplings_for_fixed_points / 10000)
            sz_bin_s = list(np.array(sz_bin_s) * k)

            print('before leaving bins wth nonzero sz_bin')
            # leave bins wth nonzero sz_bin
            #
            ii = np.argwhere(sz_bin_s)
            ii = [i[0] for i in ii]
            y_bin_s = [y_bin_s[i] for i in ii]
            x_bin_s = [x_bin_s[i] for i in ii]
            # TBD: add 1 to habdle the values like 0.5
            sz_bin_s = [sz_bin_s[i] for i in ii]
            fn = dir_out + r'/attractors_' + str(len(np.argwhere(sz_bin_s))) + '_points_' + \
                 str(len(lat_circle_s)) + '_cls_' + \
                 str(epochs) + '_ephs_' + \
                 display_attractors_which_person

            if display_attractors_added_noise_level:
                fn += '_noise_' + str(display_attractors_added_noise_level)
            f = open(fn + '.txt', 'w')

            for lat_circle in lat_circle_s:
                f.write("----------\n")
                for lat in lat_circle:
                    f.write(str(lat) + '\n')
            f.close()

            # clean lat_circle_s: remove lat_circle's close to other lat_circle's
            ii = []  # i's = 0 or 1
            for i in range(len(lat_circle_s)):
                i_xal = 1
                for j in np.arange(i + 1, len(lat_circle_s)):
                    # if lat_circle_s[i] is close to lat_circle_s[j]
                    # mark i by 0 (i_xal)
                    lat_circ_1 = lat_circle_s[i]
                    lat_1 = lat_circ_1[0]
                    lat_circ_2 = lat_circle_s[j]
                    for lat_2 in lat_circ_2:
                        if np.linalg.norm(lat_1 - lat_2) < T:
                            i_xal = 0
                            break
                    if i_xal == 0:
                        break
                # for j in np.arange(i + 1, len(lat_circle_s)):
                ii.append(i_xal)
            # for i in range(len(lat_circle_s)):

            lat_circle_s = [lat_circle_s[i] for i in range(len(lat_circle_s)) if ii[i]]

        # if/else display_trace_started_with_yx:
    else:  # if/else display_attractors
        y_bin_s = []
        x_bin_s = []
        lat_circle_s = []
        sz_bin_s = 0

    # if/else display_attractors:

    if display_attractors_which_person == 'p1':
        autoencoder = glb.p1.autoencoder
    else:
        autoencoder = glb.p2.autoencoder

    if display_trace_started_with_yx:
        """ 
        here trace_yx is N points starting with display_trace_started_with_yx,
        display_trace_started_with_yx is displayed in red
        """
        pass
    else:
        # save for further use wth display_trace_started_with_yx
        np.save(dir_out + '/' + 'lat_circle_s.npy', lat_circle_s)
        np.save(dir_out + '/' + 'lat_min_x.npy', lat_min_x)
        np.save(dir_out + '/' + 'lat_max_x.npy', lat_max_x)
        np.save(dir_out + '/' + 'lat_min_y.npy', lat_min_y)
        np.save(dir_out + '/' + 'lat_max_y.npy', lat_max_y)
        np.save(dir_out + '/' + 'x_bin_s.npy', x_bin_s)
        np.save(dir_out + '/' + 'y_bin_s.npy', y_bin_s)
        np.save(dir_out + '/' + 'sz_bin_s.npy', sz_bin_s)

        trace_yx = None
    # if/else display_trace_started_with_yx:

    plot_reconstructed(autoencoder, rx_inp=(lat_min_x, lat_max_x), ry_inp=(lat_min_y, lat_max_y), \
                       n_small_ims_per_coord=24, x_bin_s=x_bin_s, y_bin_s=y_bin_s, sz_bin_s=sz_bin_s, \
                       fn_png=fn + '.png', lat_circle_s=lat_circle_s, trace_yx=trace_yx)

    return


# def disp_lat_space:

def run_and_display(pars):
    fig, ax_x, ax_y, ax_z, ax_t, btn_pauseresume, btn_stop = init_graphisc()
    run_conn(pars, fig, ax_x, ax_y, ax_z, ax_t)
    close_conn()


def get_conn_hist(im, ListTraAt):
    # hst is of lengthn len(ListTraAt) + 1 (att not found)

    # form a batch wth len n_samples_make_hist - row #0
    # pass via autoenc wth adding noise
    # row #

    tmp = 10
    return None


# ----------------------------------------
#                seeds_lat.append(im_lat)
#                seeds_lab.append(int(digit))
#                d0a(t2i(autoencoder.decode.forward(self.seeds_lat[27])[0]))
#                d0a(t2i(autoencoder.decode.forward(im_lat)[0]))
# ----------------------------------------
# autoencoder.encode.forward(obs) # tensor([[ 705.4142, -993.1102]], grad_fn=<AddmmBackward>)
#            x_hat = autoencoder.decode.forward(z)
# ----------------------------------------
# self.p_key = p_key

# autoencoder.forward(x) == autoencoder(x)
# lat.detach().numpy()


# d0a(t2i(autoencoder.decode(  autoencoder.encode.forward(im)  )[0]) )

# ---------

# jupyter notebook

# TBD: pich nor only fromn the firs batchg

# d0a(t2i(im[0]))
# im = self.autoencoder.decode.forward(self.autoencoder.encode.forward(im))
#
# d0a(t2i(im2[0]))

if 0:
    for p in range(100):
        im2 = self.autoencoder.decode.forward(self.autoencoder.encode.forward(im2))
        # x2 = self.autoencoder.decode.forward(self.autoencoder.encode.forward(x2))

# im2 = self.autoencoder.decode.forward(self.autoencoder.encode.forward(im))

#   git pull origin
#   or git pull
#   #
#   git push origin
#   or git push

"""
work in display_lat_space mode:

disp_lat_space()
    if display_attractors:
        call with n_samplings_for_fixed_points 
            get_fixed_point, returns
                one fixed point (lat)
                / lat_circle
                // if display_trace_started_with_yx, simply return N points starting with 
                // display_trace_started_with_yx 
        att_y_s,att_x_s
        /also:lat_circle_s
        form pairs (y_bin,x_bin) all poss comb; sz_bin_s
        splash att's by the bins
        normalize sz_bin_s for displaying
        leave bins wth nonzero sz_bin
        write to fn += '_noise_'
        /also:clean lat_circle_s: remove lat_circle's close to other lat_circle's

        def plot_reconstructed(autoencoder, rx=(-25, 25), ry=(-25, 25), n_small_ims_per_coord=12, \
                               x_bin_s=[], y_bin_s=[], sz_bin_s=[], fn_png=[], lat_circle_s=[],\
                               trace_yx = None):

        // if trace_yx:
        //   plot_reconstructed receives trace_yx which is
        //   N points starting with display_trace_started_with_yx,
        //   display_trace_started_with_yx is displ in red
        //   other trace_yx points in magenta

"""


def get_id_rmnist_dir(eu_op_db_spec):
    # from 'E:\\AE\\Projects\\iterative-semiotics-networks\\data\\MNIST\\rmnist\\rmnist_10_aug6\\train\\tens'
    # extracts 'rmnist_10_aug6'
    for wrd in Path(eu_op_db_spec).parts[::-1]:
        if 'rmnist_' in wrd:
            break
    return wrd


def main_run():
    pars = init_pars()

    for n_run, eu_op_db_spec in enumerate(eu_op_db_spec_s):

        # TBD: it seems we may initialize just at n_run == 0
        #  TBD: it seems we  do not need to use eu_op_db_spec here
        init_global_vars(pars, eu_op_db_spec)

        if eu:
            if eu_op in ['op2', 'op3', 'op2op3']:
                glb.p1.model_fn = model_fn_op2op3_s[n_run]
            elif eu_op in ['op1']:
                id = get_id_rmnist_dir(eu_op_db_spec)
                glb.p1.model_fn = os.path.join(dir_models, 'model_') + id + \
                                  '_' + 'Ep' + str(epochs) + '_' + 'p1' + '.pth'
        else:
            assert False, 'define glb.p1.model_fn, glb.p2.model_fn'

        if load_trained:
            glb.p1.autoencoder.load_state_dict(torch.load(glb.p1.model_fn))
            if dbg18:
                # glb.p1.autoencoder = glb.p1.autoencoder.to("cuda",dtype=torch.float32)
                glb.p1.autoencoder.cuda()
            if not eu:
                glb.p2.autoencoder.load_state_dict(torch.load(glb.p2.model_fn))
        else:

            tmptmp = 0
            if tmptmp:
                for images, labels in glb.p1.data:
                    break

            if tr_data_mt:
                assert eu_op == 'op1'
                glb.p1.autoencoder.load_state_dict(torch.load(tr_data_mt_models[n_run]))

            glb.p1.autoencoder = train(glb.p1.autoencoder, glb.p1.data, epochs, eu_op_db_spec, \
                                       os.path.splitext(glb.p1.model_fn)[0])
            if tr_data_mt:
                pass
            else:
                torch.save(glb.p1.autoencoder.state_dict(), glb.p1.model_fn)
            if not eu:
                glb.p2.autoencoder = train(glb.p2.autoencoder, glb.p2.data, epochs, eu_op_db_spec, \
                                           os.path.splitext(glb.p1.model_fn)[0])
                torch.save(glb.p2.autoencoder.state_dict(), glb.p2.model_fn)

        if normal_run:
            run_and_display(pars)
        elif test_resiliance:
            if tr.disp_res:
                disp_res()
            else:
                explore_resiliance()
                sys.exit('normal ternimation after explore_resiliance()')
        elif display_lat_space:
            disp_lat_space()
        elif eu and eu_op == 'op2':
            train_data_mnist = 1
            glb.data_feed_autoenc = get_mnist_data_dataloader(labs_TRb, train_data_mnist,
                    overriden_location=eu_op_db_spec, shuffle = False) #shuffle = False see comment on op2op3
            map_examples_to_attractors(n_run, eu_op)
        elif eu and eu_op == 'op3':
            train_data_mnist = 0
            #standard mnist test data, act need it just n_run == 0:
            glb.data_feed_autoenc = get_mnist_data_dataloader(labs_TRb, train_data_mnist, \
                    sz_stri=eu_scale, shuffle = True) #shuffle = True see comment on op2op3
            map_examples_to_attractors(n_run, eu_op)
            if make_test_data_folder:
                sys.exit('normal ternimation at make_test_data_folder, \
                        data written to '+str(mnist_test_data_folder))
        elif eu and eu_op == 'op2op3':
            train_data_mnist = 1
            #shuffle = False, because now we want just want to map the training rmnist examples
            #from eu_op_db_spec to attrators, shufling will be made in aside2 at training
            glb.data_feed_autoenc = get_mnist_data_dataloader(labs_TRb, train_data_mnist, \
                    sz_stri=eu_scale, overriden_location=eu_op_db_spec, shuffle = False)
            map_examples_to_attractors(n_run, 'op2')
            train_data_mnist = 0
            #shuffle = True, because our test data is op3_num_total randomly picked examples from mnist test set
            glb.data_feed_autoenc = get_mnist_data_dataloader(labs_TRb, train_data_mnist, sz_stri=eu_scale, shuffle = True)
            map_examples_to_attractors(n_run, 'op3')

        tmp=10

    # for eu_op_db_spec in eu_op_db_spec_s:


# def main_run():


if __name__ == '__main__':
    main_run()
# if __name__ == '__main__':


