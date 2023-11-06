# aside2.py
dbg19 = 0 #short
dbg22 = 0 #development of aside2 #2
dbg23 = 0 #bug serach
make_examples = 0#make_examples = 1: R8 200 epocks
dbg25 = 0#short for make_examples

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from conn import get_mnist_data_dataloader
from general import is_empty, d0a, Str, i2d, t2i, i2t, struct


seed = 0
np.random.seed(seed=seed)
torch.manual_seed(seed)


is_win = (sys.platform[:3] == "win")

# Define the neural network architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the MNIST dataset and transform it to tensors
transform = transforms.Compose([transforms.ToTensor()])

# build the network equiv to
# sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(500, 100))
# used by Michael_Nielsen https://github.com/mnielsen/rmnist
clf = dict()
clf['learning_rate_init'] = 0.001
clf['beta_1'] = 0.9
clf['beta_2'] = 0.999

clf['batch_size'] = 100
clf['hidden_layer_sizes'] = (500, 100)
clf['epsilon'] = 1e-08
clf['verbose'] = False#True

# Create an instance of the MLPClassifier model
input_size = 28 * 28  # the number of pixels in each image
hidden_sizes = clf['hidden_layer_sizes']  # the number of neurons in each hidden layer
output_size = 10  # the number of classes (digits) in MNIST

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

batch_size = clf['batch_size']

if 0 and dbg19:
    if 1:#op5
        overrid_train_data_s = [r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_5aug0\train\tens']
        # test_data_op4
        overrid_test_data_s = [r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\test_data\tens']
    else:#op4
        # train_data_op4
        overrid_train_data_s = [r'E:\AE\Projects\iterative-semiotics-networks\data\runs.aug0\run0\ATRb\tens']
        # test_data_op4
        overrid_test_data_s = [r'E:\AE\Projects\iterative-semiotics-networks\data\runs.aug0\run0\ATEb\tens']
elif 0:
    overrid_train_data_s_op4 = [
        r'G:\Downloads\op2op3.aug0\part\run0\ATRb\tens',
    ]
    overrid_test_data_s_op4 = [
        r'G:\Downloads\op2op3.aug0\part\run0\ATEb\tens',
    ]
    overrid_train_data_s_op5 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist\rmnist_5-9\rmnist_5-9_aug0\rmnist_5aug0\train\tens',
        ]
    overrid_test_data_s_op5 = []
    for i in range(1):
        overrid_test_data_s_op5 += [r'G:\Downloads\test_data\test_data\tens']
elif 0: #vanilla
    if is_win:
        overrid_train_data_s_op4 = [
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run0\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run1\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run2\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run3\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run4\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run5\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run6\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run7\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run8\ATRb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run9\ATRb\tens',
        ]
        overrid_test_data_s_op4 = [
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run0\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run1\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run2\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run3\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run4\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run5\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run6\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run7\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run8\ATEb\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.simhconn=0\run9\ATEb\tens',
        ]
        overrid_train_data_s_op5 = [
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_5aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_6aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_7aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_8aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_9aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_10aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_20aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_30aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_40aug0\train\tens',
            r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_50aug0\train\tens',
            ]
        overrid_test_data_s_op5 = []
        for i in range(len(overrid_train_data_s_op5)):
            overrid_test_data_s_op5 += [r'E:\AE\Projects\iterative-semiotics-networks\data\test_data\tens']
    else:#if/else is_win:
        overrid_train_data_s_op4 = [
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run0/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run1/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run2/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run3/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run4/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run5/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run6/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run7/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run8/ATRb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run9/ATRb/tens',
        ]
        overrid_test_data_s_op4 = [
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run0/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run1/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run2/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run3/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run4/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run5/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run6/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run7/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run8/ATEb/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run9/ATEb/tens',
        ]
        overrid_train_data_s_op5 = [
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_5aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_6aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_7aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_8aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_9aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_10aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_20aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_30aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_40aug0/train/tens',
            r'/home/research/projects2/iterative-semiotics-networks/data/MNIST/rmnist_50aug0/train/tens',
            ]
        overrid_test_data_s_op5 = []
        for i in range(len(overrid_train_data_s_op5)):
            overrid_test_data_s_op5 += [r'/home/research/projects2/iterative-semiotics-networks/data/test_data/tens']
    #if/else is_win:
elif 1: #hconn (aug0, h2=0) BEST stochastic
    overrid_train_data_s_op4 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run0\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run1\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run2\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run3\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run4\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run5\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run6\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run7\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run8\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run9\ATRb\tens',
    ]
    overrid_test_data_s_op4 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run0\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run1\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run2\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run3\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run4\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run5\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run6\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run7\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run8\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.ATRbATEb\run9\ATEb\tens',
    ]
    overrid_train_data_s_op5 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_5aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_6aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_7aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_8aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_9aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_10aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_20aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_30aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_40aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_50aug0\train\tens',
        ]
    overrid_test_data_s_op5 = []
    for i in range(len(overrid_train_data_s_op5)):
        overrid_test_data_s_op5 += [r'E:\AE\Projects\iterative-semiotics-networks\data\test_data\tens']

elif 0: #h2
    overrid_train_data_s_op4 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run0\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run1\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run2\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run3\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run4\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run5\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run6\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run7\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run8\ATRb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run9\ATRb\tens',
    ]
    overrid_test_data_s_op4 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run0\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run1\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run2\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run3\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run4\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run5\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run6\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run7\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run8\ATEb\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\result_data.aug0.h2\run9\ATEb\tens',
    ]
    overrid_train_data_s_op5 = [
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_5aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_6aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_7aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_8aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_9aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_10aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_20aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_30aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_40aug0\train\tens',
        r'E:\AE\Projects\iterative-semiotics-networks\data\MNIST\rmnist.aug0\rmnist_50aug0\train\tens',
        ]
    overrid_test_data_s_op5 = []
    for i in range(len(overrid_train_data_s_op5)):
        overrid_test_data_s_op5 += [r'E:\AE\Projects\iterative-semiotics-networks\data\test_data\tens']

if make_examples: #R8
    overrid_train_data_s_op4 = [overrid_train_data_s_op4[3]]
    overrid_test_data_s_op4 = [overrid_test_data_s_op4[3]]
    overrid_train_data_s_op5 = [overrid_train_data_s_op5[3]]
    overrid_test_data_s_op5 = [overrid_test_data_s_op5[3]]
if 0 and dbg23:
    overrid_train_data_s_op4 = [overrid_train_data_s_op4[-2], overrid_train_data_s_op4[-1]]
    overrid_test_data_s_op4 = [overrid_test_data_s_op4[-2], overrid_test_data_s_op4[-1]]
    overrid_train_data_s_op5 = [overrid_train_data_s_op5[-2], overrid_train_data_s_op5[-1]]
    overrid_test_data_s_op5 = [overrid_test_data_s_op5[-2], overrid_test_data_s_op5[-1]]

assert len(overrid_train_data_s_op4) == len(overrid_test_data_s_op4)
assert len(overrid_test_data_s_op4) == len(overrid_train_data_s_op5)
assert len(overrid_train_data_s_op5) == len(overrid_test_data_s_op5)

def train(train_loader, mlp, optimizer, clf, num_epochs):

    if dbg22:
        return mlp

    X_val = []
    y_val = []

    for epoch in range(num_epochs):
        if epoch > 0:
            print('\r', end='', flush=True)
        flush = (epoch==(num_epochs-1))
        end = '\n' if (epoch==(num_epochs-1)) else ''
        print('[epoch='+str(epoch)+' of '+str(num_epochs)+']', end=end, flush=flush)
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            images = images.reshape(-1, 28 * 28).float()
            outputs = mlp(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training loss every 100 batches
            if i == 0 and clf['verbose']:
                print('epoch=' + str(epoch) + ' Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i,\
                        len(train_loader),loss.item()) + ' i=' + str(i))

    return mlp

#def train(train_loader, clf):


def test2(nR, mlp_op4, X_test_op4, y_test_op4, mlp_op5, X_test_op5, y_test_op5, num_epochs, res_dict,
        filenames_op4 = [], filenames_op5 = []):
    # filenames_op4, filenames_op4 are for make_examples

    assert np.array_equal(y_test_op4, y_test_op5)

    if dbg22:
        return

    y_test = y_test_op4

    # Test the model
    with torch.no_grad():
        # Forward pass
        outputs_op4 = mlp_op4(X_test_op4)
        _, predicted_op4 = torch.max(outputs_op4.data, 1)

        # Compute accuracy
        total_op4 = y_test.size(0)
        correct_op4 = (predicted_op4 == y_test).sum().item()
        accuracy_op4 = correct_op4 / total_op4

        # Forward pass
        outputs_op5 = mlp_op5(X_test_op5)
        _, predicted_op5 = torch.max(outputs_op5.data, 1)

        # Compute accuracy
        total_op5 = y_test.size(0)
        correct_op5 = (predicted_op5 == y_test).sum().item()
        accuracy_op5 = correct_op5 / total_op5

        print('num_epochs = {:d} nR = {:d} Overall: Accuracy on test set _op4: {:.2f}%  on test set _op5 {:.2f}%'.format(\
                num_epochs, nR, accuracy_op4*100, accuracy_op5*100))

        if make_examples:
            # filenames_op4
            if 0:
                m = 445
                im = X_test_op4[m].numpy()
                im.shape=(28,28,1)
                d0a(im)
                print('...'+filenames_op4[m][-40::])
                #Out[7]: '/home/research/projects2/iterative-semiotics-networks/data/runs.vanilla/run3/ATEb/tens/0/nimabs=183.nb=183.nim=0.pt'
            tmp=10
            sys.exit('normal ternimation in make_examples mode')

        res=dict()
        res['num_epochs'] = num_epochs
        res['nR'] = nR
        res['nlab'] = 'overall'
        res['acc_op4'] = accuracy_op4*100
        res_dict[(num_epochs, nR, 'overall', 'op4')] = res

        res=dict()
        res['num_epochs'] = num_epochs
        res['nR'] = nR
        res['nlab'] = 'overall'
        res['acc_op5'] = accuracy_op5*100
        res_dict[(num_epochs, nR, 'overall', 'op5')] = res

        for nlab in range(10):
            predicted_op4_nlab = predicted_op4[y_test == nlab]
            total_op4_lab = np.argwhere(y_test == nlab).shape[1]
            correct_op4_lab = (predicted_op4 == y_test)[y_test == nlab].sum().item()
            accuracy_op4_lab = correct_op4_lab / total_op4_lab

            predicted_op5_nlab = predicted_op5[y_test == nlab]
            total_op5_lab = np.argwhere(y_test == nlab).shape[1]
            correct_op5_lab = (predicted_op5 == y_test)[y_test == nlab].sum().item()
            accuracy_op5_lab = correct_op5_lab / total_op5_lab

            print('num_epochs = {:d} nR = {:d} Label {:d}: Accuracy on test set _op4: {:.2f}%  on test set _op5 {:.2f}%'.\
                    format(num_epochs, nR, nlab, accuracy_op4_lab*100, accuracy_op5_lab*100))

            res=dict()
            res['num_epochs'] = num_epochs
            res['nR'] = nR
            res['nlab'] = nlab#res['overall'] = nlab
            res['acc_op4'] = accuracy_op4_lab*100
            res_dict[(num_epochs, nR, nlab, 'op4')] = res

            res=dict()
            res['num_epochs'] = num_epochs
            res['nR'] = nR
            res['nlab'] = nlab#res['overall'] = nlab
            res['acc_op5'] = accuracy_op5_lab*100
            res_dict[(num_epochs, nR, nlab, 'op5')] = res

            tmp=10


        #for nlab in range(10):

        tmp=10

        np.save('res_dict.npy', res_dict)
        #res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()


    return res_dict

#def test2(mlp_op4, X_test_op4, y_test_op4, mlp_op5, X_test_op5, y_test_op5)

def get_vals(res_dict):
    num_epochs_s = []
    nR_s = []
    for key in res_dict.keys():
        num_epochs_s.append(key[0])
        nR_s.append(key[1])
    num_epochs_s = np.unique(num_epochs_s).tolist()
    nR_s = np.unique(nR_s).tolist()
    print('num_epochs_s = '+ ' ' +str(num_epochs_s))
    print('nR_s = '+ ' ' +str(nR_s))
    return num_epochs_s, nR_s

def get_vals_100runs(res_dict):

    if 0:
        for n, res in enumerate(res_dict):
            print(res_dict[res])
            #{'num_epochs': 15, 'nR': 5, 'nlab': 'overall', 'acc_op4': 73.5, 'seed2': 2000}

        for n, res in enumerate(res_dict):
            if res[2] != 'overall':
                print(res_dict[res])
                #{'num_epochs': 3, 'nR': 9, 'nlab': 9, 'acc_op5': 87.5}

    num_epochs_s = []
    nR_s = []
    nlab_s = []
    acc_s = []
    seed2_s = []
    for n, res in enumerate(res_dict):
        if res[2] != 'overall':
            print('Warning: res = ' +  str(res) + ' res_dict[res] = ' + str(res_dict[res]))
            continue

        '''
        res
        Out[8]: (200, 0, 'overall', 'op4', 0)
        res_dict[res]
        Out[9]: {'num_epochs': 200, 'nR': 0, 'nlab': 'overall', 'acc_op4': 68.2, 'seed2': 684}            
        '''
        num_epochs_s.append(res_dict[res]['num_epochs'])
        nR_s.append(res_dict[res]['nR'])
        assert res_dict[res]['nlab'] == 'overall'
        nlab_s.append(res_dict[res]['nlab'])
        if 'acc_op4' in res_dict[res]:
            acc_s.append(res_dict[res]['acc_op4'])
        elif 'acc_op5' in res_dict[res]:
            acc_s.append(res_dict[res]['acc_op5'])
        seed2_s.append(res_dict[res]['seed2'])

    #for n, res in enumerate(res_dict):
    num_epochs_s = np.unique(num_epochs_s) #array([  3,   6,  15,  25,  50, 100, 200])
    nR_s = np.unique(nR_s)#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    nlab_s = np.unique(nlab_s)#array(['overall'], dtype='<U7')
    acc_s = np.unique(acc_s)# array(['op4', 'op5'], dtype='<U3')
    seed2_s = np.unique(seed2_s)#array([ 0,  1,  2, ...  51, 52, 53]) #FIX
    return num_epochs_s,nR_s,nlab_s,acc_s,seed2_s


def train_test():

    global seed
    res_dict = dict()

    num_epochs_s = [200, 100, 50, 25, 15, 6, 3]
    if dbg23:
        num_epochs_s = [5, 10, 15]
    if make_examples:
        if dbg25:
            num_epochs_s = [6]
        else:
            num_epochs_s = [200]

    for num_epochs in num_epochs_s:

        if dbg19:
            if num_epochs not in [15]:
                continue

        for nR, (overrid_train_data_op4, overrid_train_data_op5) in enumerate(zip(overrid_train_data_s_op4, overrid_train_data_s_op5)):
            if dbg19:
                if nR not in [0, 5, 6]:
                    continue

            print('----- num_epochs = ' +str(num_epochs) + ' ----- nR = ' +str(nR) + ' ---------')
            print('overrid_train_data_op4='+overrid_train_data_op4)
            overrid_test_data_op4 = overrid_test_data_s_op4[nR]
            print('overrid_test_data_op4='+overrid_test_data_op4)

            of_train = 1
            np.random.seed(seed=seed)
            torch.manual_seed(seed)
            train_loader_op4 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_train_data_op4, shuffle = True)

            if overrid_test_data_op4:
                of_train = 0
                #shufling was done while creating overrid_test_data_op4  qq1
                test_loader_op4 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_test_data_op4, \
                        shuffle = False)
                if make_examples:
                    for images, labels in test_loader_op4:
                        filenames_op4 = [sample[0] for sample in test_loader_op4.dataset.samples]
                        print(filenames_op4)
                        print (labels)
                    #if not reinit, the above enum influences the classifier cratedand snange the results
                    np.random.seed(seed=seed)
                    torch.manual_seed(seed)
                else:
                    filenames_op4 = []
                X_test = []
                y_test = []
                for data in test_loader_op4:
                    inputs, targets = data
                    X_test.append(inputs)
                    y_test.append(targets)
                X_test = torch.cat(X_test, dim=0)
            else:
                test_loader_op4 = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            X_test_op4 = X_test.data.reshape(-1, 28 * 28).float()
            y_test_op4 = torch.cat(y_test, dim=0)

            mlp_op4 = MLPClassifier(input_size, hidden_sizes, output_size)

            optimizer_op4 = optim.Adam(mlp_op4.parameters(), lr=clf['learning_rate_init'], \
                    betas=(clf['beta_1'], clf['beta_2']),eps=clf['epsilon'])

            mlp_op4 = train(train_loader_op4, mlp_op4, optimizer_op4, clf, num_epochs)

            # ------------

            print('overrid_train_data_op5='+overrid_train_data_op5)
            overrid_test_data_op5 = overrid_test_data_s_op5[nR]
            print('overrid_test_data_op5='+overrid_test_data_op5)

            of_train = 1
            np.random.seed(seed=seed)
            torch.manual_seed(seed)
            train_loader_op5 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_train_data_op5, \
                    shuffle = True)

            if overrid_test_data_op5:
                of_train = 0
                #shuling was done while creating overrid_test_data_op5 qq2
                test_loader_op5 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_test_data_op5,\
                        shuffle = False )
                if make_examples:
                    for images, labels in test_loader_op5:
                        filenames_op5 = [sample[0] for sample in test_loader_op5.dataset.samples]
                        print(filenames_op5)
                        print (labels)
                    #if not reinit, the above enum influences the classifier cratedand snange the results
                    np.random.seed(seed=seed)
                    torch.manual_seed(seed)
                else:
                    filenames_op5 = []
                X_test = []
                y_test = []
                for data in test_loader_op5:
                    inputs, targets = data
                    X_test.append(inputs)
                    y_test.append(targets)
                X_test = torch.cat(X_test, dim=0)
            else:
                test_loader_op5 = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            X_test_op5 = X_test.data.reshape(-1, 28 * 28).float()
            y_test_op5 = torch.cat(y_test, dim=0)

            mlp_op5 = MLPClassifier(input_size, hidden_sizes, output_size)

            optimizer_op5 = optim.Adam(mlp_op5.parameters(), lr=clf['learning_rate_init'], \
                    betas=(clf['beta_1'], clf['beta_2']),eps=clf['epsilon'])

            assert np.array_equal(y_test_op4,y_test_op5)

            mlp_op5 = train(train_loader_op5, mlp_op5, optimizer_op5, clf, num_epochs)

            # ---

            res_dict = test2(nR, mlp_op4, X_test_op4, y_test_op4, mlp_op5, X_test_op5, y_test_op5, num_epochs, res_dict,\
                    filenames_op4 = filenames_op4, filenames_op5 = filenames_op5)

            tmp=10

        #for nR, overrid_train_data in enumerate(overrid_train_data_s):
    #for num_epochs in [200, 100, 50, 25, 15, 6, 3]:
    tmp=10
#def train_test():

class PrintTee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
        self.file_handle = open(self.file, 'w')

    def write(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        self.stdout.write(message)
        self.file_handle.write(message)

    def flush(self):
        self.stdout.flush()
        self.file_handle.flush()

    def close(self):
        self.stdout.close()
        self.file_handle.close()

if __name__ == "__main__":


# Open the file in write mode
    file = 'output.txt'

    # Create an instance of the PrintTee class
    print_tee = PrintTee(file)

    # Redirect the sys.stdout to the PrintTee instance
    sys.stdout = print_tee

    # Your code goes here
    print("This will be redirected to the file and displayed on the screen")

    train_test()

    # Close the file
    print_tee.close()

    #res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()
    #num_epochs_s, nR_s = get_vals(res_dict)

    tmp=10


#if __name__ == "__main__":

'''

res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()
num_epochs_s, nR_s = get_vals(res_dict)

np.save('res_dict.npy', res_dict)
#res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()

res_dict[(num_epochs, nR, 'overall', 'op4')] = res
res_dict[(num_epochs, nR, 'overall', 'op5')] = res
res_dict[(num_epochs, nR, nlab, 'op4')] = res
res_dict[(num_epochs, nR, nlab, 'op5')] = res


'''