'''
if load_trained == 1:
    train and write the models to 'models' subfolder (specified by dir_models) of the orbits.py's folder
else:    
    reads the pretrained models from dir_models

writes MNIST data to 'data' folder specified by dir_data

if short_mode == 1, proceeds in a 'short' traing mode
'''    
load_trained = 0
short_mode = 1


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
from torch.utils.data import Dataset

from general import is_empty, d0a, Str, i2d, t2i, i2t, struct


# We acknowledge the following references
# for providing code examples that influenced the
# development of our CONN implementation:
#
# Aaron Stone, Richard Kalehoff, Jacob Walker,
# Islam Al-Afifi, and Patrick Donovan
# {\em PyTorchCrashCourse}
# https://github.com/udacity/pytorchcrashcourse
#
# Shashank Dhar,
# {\em A simple implementation of variational autoencoder algorithm (VAE) using the MNIST dataset},
# https://github.com/shashankdhar/VAE-MNIST
#
# Variational AutoEncoders (VAE) with PyTorch by Alexander Van de Kleut
# https://avandekleut.github.io/vae/
#
# Michael Nielsen
# Some examples trained on very reduced versions of the MNIST training set
# https://github.com/mnielsen/rmnist
#
# Gregor Koehler with Andrea Amantini and Philippa Markovics
# Remix of PyTorch Environment by Martin Kavalar
# MNIST Handwritten Digit Recognition in PyTorch
# https://nextjournal.com/gkoehler/pytorch-mnist


normal_run = 1


# at op3 the test data is sampled (op3_num_total samples) from
# standart mnist test data, then convereed to attractors
# (in map_examples_to_attractors) and then is written to ATE data folder.
# we need the same test data (without conversion for attractors)
# for running  the benchmark torch classifier. This folder (mnist_test_data_folder)
# is created as in op3, with skipping coversion to attractors
make_test_data_folder = 0

all_train_in_mt_mode = 0

val_data_mt = 0  # use val data for model tuning

op_batch = 1  # op2, 'op3, op2op3  enumerate and process images in batch

batch_size = 128  # 256 #win: 1: 1 min and 15 sec 64: 27 sec, 128 - 25 sec

# is for fconv and blk flags
# our implementation may resemble
# https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
# last present in the version: /July25.2023 B: removal of .... /
skip = 0
save_loss_epochs = 2500
if short_mode:
    save_loss_epochs = 1

epochs = 20
if short_mode:
    epochs = 5

fconv = 0  # fully conv autoenc

blk = 0  # netw arch of belkin
blk_prs = struct()

if fconv == 1 and blk == 0:
    assert False, 'github version in preparation'

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
lat_dimension = 100

hconn = 0  # ('op2', 'op3', 'op2op3') means stochastic

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

if normal_run:
    eu_op_db_spec_s = [None]

assert (not skip) or (fconv and flags) ,' skip => (fconv and flags)'


dbg_get_aug_radiom = 0



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


plt.rcParams['figure.dpi'] = 200

randomize_descends = 0
lat_ranges = (-10, 10)

if blk:
    seed = blk_prs.seed
else:
    seed = 0
np.random.seed(seed=seed)
torch.manual_seed(seed)

init_to_digit = 0  # randomly init to digit or to random


n_bins_att_1d = 10000  # 10000


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

if 1:
    train_labs['p1'] = [1, 3, 5, 7, 9]
    train_labs['p2'] = [0, 2, 4, 6, 8]
elif 0:
    train_labs['p1'] = [1]
    train_labs['p2'] = [0]
elif 0:
    train_labs['p1'] = []
    train_labs['p2'] = []

init_seeds_simple_examples = 1


nsteps_p1_2_p2 = 1
nsteps_p2_2_p1 = 1

save_fig_interval = 50

dbg_nrm = 0

assert nsteps_p1_2_p2 == 1 and nsteps_p2_2_p1 == 1


n_samplings_for_fixed_points = 2000


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

        if train_data_mnist == 0:
            tmp=10

        data = tv.datasets.DatasetFolder(overriden_location, torch.load, target_transform=transform, extensions=('pt'))

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


    if labs != None and set(labs) != set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        assert not overriden_location, 'currently labs subsetting does not work for overriden_location, TBD '
        for i, lab in enumerate(labs):
            if i == 0:
                ii = (my_dataloader.dataset.targets == lab)
            else:
                ii = torch.logical_or(ii, my_dataloader.dataset.targets == lab)

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

        return my_dataloader2
    else:
        return my_dataloader

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
                self.conv5 = nn.Conv2d(256, 256, krn, stride=2, padding=pad)

                # Decoder
                # x5 (conv256 strid 1 leaky rely -> bilinear)
                # x2 (conv256 strid 1 leaky rely)
                #    (conv3 strid 1  leaky rely)

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

                # Encoder

                # ...

                # Decoder
                # ...

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


    #def __init__(self, lat_dimension):

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
            # if/else blk:
        # if/else fconv:
        return x
    #def encode2(self, x):

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

    #def decode2(self, z):

    def forward(self, x):
        z = self.encode2(x)
        t = self.decode2(z)
        return t

#class Network(nn.Module):


def train(autoencoder, data, epochs, eu_op_db_spec, dir_interm_models):
    if not os.path.exists(dir_interm_models):
        os.makedirs(dir_interm_models)

    if blk:
        opt = torch.optim.Adam(autoencoder.parameters(), lr=blk_prs.lr)
    else:
        opt = torch.optim.Adam(autoencoder.parameters())
    losses = []
    n_epochs = []

    for epoch in range(epochs):
        losses_epoch = []
        for nbatch, (x, y) in enumerate(data):

            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if randomize_descends:
                x += torch.randn(x.shape)

            x_hat = autoencoder(x)
            assert (x.shape == x_hat.shape)
            loss = torch.mean(torch.linalg.norm(x - x_hat, dim=(2, 3)))
            loss_np = float(loss.detach().cpu().numpy())

            if short_mode or (epoch % 20 == 0 and nbatch % 10 == 0):
                stri = '  ' + datetime.datetime.now().strftime('%d.%m.%y,%H:%M:%S.%f')
                print(' epoch=' + str(epoch) + ' nbatch=' + str(nbatch) + \
                      ' loss=' + Str(loss_np, prec=3) + stri)

            loss.backward()
            opt.step()
            losses_epoch.append(loss_np)

        # for nbatch, (x, y) in enumerate(data):
        n_epochs.append(epoch)

        loss_epoch = np.mean(losses_epoch)
        losses.append(loss_epoch)

        print('epoch=' + str(epoch) + ' loss_epoch=' + str(loss_epoch))


        if epoch % save_loss_epochs == 0:
            fn = os.path.join(dir_interm_models, 'epoch' + str(epoch) + '.pth')
            torch.save(autoencoder.state_dict(), fn)

            np.save(os.path.join(dir_interm_models, 'n_epochs' +  '.npy'), n_epochs)
            np.save(os.path.join(dir_interm_models, 'losses' +  '.npy'), losses)

    # for epoch in range(epochs):

    tmp = 10
    return autoencoder


class my_dataset_class(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



def actions_cnt(fig):
    if glb.cnt == 35:
        tmp = 10
    glb.cnt += 1
    if glb.cnt % save_fig_interval == 0:
        fig.savefig(dir_out + r'/gui_cnt_' + str(glb.cnt) + '.png')


def disp_latent_space_wth_digits(autoencoder, rx_inp=(-25, 25), ry_inp=(-25, 25), n_small_ims_per_coord=12, \
                       x_bin_s=[], y_bin_s=[], sz_bin_s=[], fn_png=[], lat_circle_s=[], \
                       trace_yx=None):
    # visualize the 2-d lat space by griding abd diaplayin  in every grid square
    # the image corresponding to the grid center
    # if y_bin_s is not empty,
    #   display in y_bin, v_bun a sircle wit the radius proportional to sz_bin
    #   (this option is for displayng fixed points, represented as y_bin_s, x_bin_s, sz_bin_s
    #
    # if trace_yx:
    #   disp_latent_space_wth_digits receives trace_yx which is
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

# def disp_latent_space_wth_digits()


def add_rand(lat, eps):
    lat2 = torch.clone(lat)
    lat2 += torch.rand(lat2.shape) * eps - eps / 2.
    return lat2



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

            update_axes(ax_x, ax_txt)
            stri = ay_txt + ' ' + 'res_lab=' + res_lab
            if 0:
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
            if 0:
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

    glb.p2 = Person(pars, 'p2', eu_op_db_spec)

    glb.cnt = 0


class Person():
    #   p_key = 'p1', 'p2'
    def __init__(self, pars, p_key, eu_op_db_spec):
        super(Person, self).__init__()
        global glb
        global train_labs

        self.autoencoder = Network(lat_dimension).to(device)

        stri =  ''

        stri += '_' + eu_scale if eu_scale else ''

        if eu_op_db_spec:
            self.data = get_mnist_data_dataloader(None, None, overriden_location=eu_op_db_spec)
        else:
            train_data_mnist = 1
            self.data = get_mnist_data_dataloader(train_labs[p_key], train_data_mnist)
        '''
        for images, labs in self.data:
            break
        '''

        self.p_key = p_key

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

        res_lab = 'none'

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


def run_and_display(pars):
    fig, ax_x, ax_y, ax_z, ax_t, btn_pauseresume, btn_stop = init_graphisc()
    run_conn(pars, fig, ax_x, ax_y, ax_z, ax_t)
    close_conn()



def main_run():
    pars = init_pars()

    init_global_vars(pars, None)
    if normal_run:
        glb.p1.model_fn = dir_models + r'/model_' + str(lat_dimension) + '_' + 'epochs_' + str(
            epochs) + '_' + 'p1' + '.pth'
        glb.p2.model_fn = dir_models + r'/model_' + str(lat_dimension) + '_' + 'epochs_' + str(
            epochs) + '_' + 'p2' + '.pth'
    else:
        assert False, 'define glb.p1.model_fn, glb.p2.model_fn'

    if load_trained:
        glb.p1.autoencoder.load_state_dict(torch.load(glb.p1.model_fn))
        glb.p2.autoencoder.load_state_dict(torch.load(glb.p2.model_fn))
    else:

        glb.p1.autoencoder = train(glb.p1.autoencoder, glb.p1.data, epochs, None, \
                                   os.path.splitext(glb.p1.model_fn)[0])
        torch.save(glb.p1.autoencoder.state_dict(), glb.p1.model_fn)
        glb.p2.autoencoder = train(glb.p2.autoencoder, glb.p2.data, epochs, None, \
                                   os.path.splitext(glb.p1.model_fn)[0])
        torch.save(glb.p2.autoencoder.state_dict(), glb.p2.model_fn)

    if normal_run:
        run_and_display(pars)


# def main_run():

if __name__ == '__main__':
    main_run()
