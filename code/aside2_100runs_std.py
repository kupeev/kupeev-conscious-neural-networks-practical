# aside2_100runs_std.py

N_RUNS = 100

dbg26 = 0 #short
dbg27 = 0 #development of aside2 #2
dbg28 = 0 #bug serach


import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datetime

from conn import get_mnist_data_dataloader
from general import is_empty, d0a, Str, i2d, t2i, i2t, struct

if dbg26:
    N_RUNS = 2

is_win = (sys.platform[:3] == "win")
if is_win:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # otherwise
    #   PyTorch no longer supports this GPU because it is too old
    #       https://discuss.pytorch.org/t/solved-pytorch-no-longer-supports-this-gpu-because-it-is-too-old/15444/6
    device = 'cpu'
else:
    device = 'cuda:0'

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

if 1: #hconn (aug0, h2=0) BEST stochastic
    overrid_train_data_s_op4 = [
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run0/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run1/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run2/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run3/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run4/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run5/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run6/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run7/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run8/ATRb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run9/ATRb/tens',
    ]
    overrid_test_data_s_op4 = [
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run0/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run1/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run2/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run3/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run4/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run5/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run6/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run7/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run8/ATEb/tens',
        r'/home/research/iterative-semiotics-networks/data/result_data.aug0.ATRbATEb/run9/ATEb/tens',
    ]
    overrid_train_data_s_op5 = [
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_5aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_6aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_7aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_8aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_9aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_10aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_20aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_30aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_40aug0/train/tens',
        r'/home/research/iterative-semiotics-networks/data/rmnist.aug0/rmnist_50aug0/train/tens',
        ]
    overrid_test_data_s_op5 = []
    for i in range(len(overrid_train_data_s_op5)):
        overrid_test_data_s_op5 += [r'/home/research/iterative-semiotics-networks/data/test_data/tens']


if 0 and dbg28:
    overrid_train_data_s_op4 = [overrid_train_data_s_op4[-2], overrid_train_data_s_op4[-1]]
    overrid_test_data_s_op4 = [overrid_test_data_s_op4[-2], overrid_test_data_s_op4[-1]]
    overrid_train_data_s_op5 = [overrid_train_data_s_op5[-2], overrid_train_data_s_op5[-1]]
    overrid_test_data_s_op5 = [overrid_test_data_s_op5[-2], overrid_test_data_s_op5[-1]]

assert len(overrid_train_data_s_op4) == len(overrid_test_data_s_op4)
assert len(overrid_test_data_s_op4) == len(overrid_train_data_s_op5)
assert len(overrid_train_data_s_op5) == len(overrid_test_data_s_op5)

def train(train_loader, mlp, optimizer, clf, num_epochs):

    if dbg27:
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
            images = images.to(device)
            labels = labels.to(device)
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


def test2_100runs(nR, mlp_op4, X_test_op4, y_test_op4, mlp_op5, X_test_op5, y_test_op5, num_epochs, \
        res_dict, seed2, n_run, filenames_op4 = [], filenames_op5 = []):
    # filenames_op4, filenames_op4 are for make_examples

    assert np.array_equal(y_test_op4.cpu().numpy(), y_test_op5.cpu().numpy())

    if dbg27:
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

        res=dict()
        res['num_epochs'] = num_epochs
        res['nR'] = nR
        res['nlab'] = 'overall'
        res['acc_op4'] = accuracy_op4*100
        res['seed2'] = seed2
        res_dict[(num_epochs, nR, 'overall', 'op4', n_run)] = res

        res=dict()
        res['num_epochs'] = num_epochs
        res['nR'] = nR
        res['nlab'] = 'overall'
        res['acc_op5'] = accuracy_op5*100
        res['seed2'] = seed2
        res_dict[(num_epochs, nR, 'overall', 'op5', n_run)] = res

        np.save('res_dict.npy', res_dict)
        #res_dict = np.load('res_dict.npy', allow_pickle=True).tolist()


    return res_dict

#def test2_100runs(mlp_op4, X_test_op4, y_test_op4, mlp_op5, X_test_op5, y_test_op5)

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



def train_test():

    global seed
    res_dict = dict()

    num_epochs_s = [200, 100, 50, 25, 15, 6, 3]
    if dbg28:
        num_epochs_s = [5, 10, 15]

    n_seed2_s = N_RUNS * len(num_epochs_s) * len(overrid_train_data_s_op4)
    seed2_s = np.random.randint(0, 4096, n_seed2_s)
    n_seed2 = 0

    for n_run in range(N_RUNS):

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print('----------- n_run = ' + str(n_run) + ' of ' + str(N_RUNS) + ' ------------- ' + current_time)
        for num_epochs in num_epochs_s:
            if dbg26:
                if num_epochs not in [15]:
                    continue

            for nR, (overrid_train_data_op4, overrid_train_data_op5) in enumerate(zip(overrid_train_data_s_op4, overrid_train_data_s_op5)):
                if dbg26:
                    if nR not in [0, 5, 6]:
                        continue

                print('----- num_epochs = ' +str(num_epochs) + ' ----- nR = ' +str(nR) + ' ---------')
                print('overrid_train_data_op4='+overrid_train_data_op4)
                overrid_test_data_op4 = overrid_test_data_s_op4[nR]
                print('overrid_test_data_op4='+overrid_test_data_op4)

                of_train = 1
                seed2  = seed2_s[n_seed2]
                n_seed2 += 1
                np.random.seed(seed=seed2)
                torch.manual_seed(seed2)
                train_loader_op4 = get_mnist_data_dataloader(None, of_train, \
                        overriden_location=overrid_train_data_op4, \
                        shuffle = True)

                if overrid_test_data_op4:
                    of_train = 0
                    #shufling was done while creating overrid_test_data_op4  qq1
                    test_loader_op4 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_test_data_op4, \
                            shuffle = False)
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
                X_test_op4 = X_test_op4.to(device)
                y_test_op4 = y_test_op4.to(device)
                mlp_op4 = MLPClassifier(input_size, hidden_sizes, output_size).to(device)

                optimizer_op4 = optim.Adam(mlp_op4.parameters(), lr=clf['learning_rate_init'], \
                        betas=(clf['beta_1'], clf['beta_2']),eps=clf['epsilon'])

                mlp_op4 = train(train_loader_op4, mlp_op4, optimizer_op4, clf, num_epochs)

                # ------------

                print('overrid_train_data_op5='+overrid_train_data_op5)
                overrid_test_data_op5 = overrid_test_data_s_op5[nR]
                print('overrid_test_data_op5='+overrid_test_data_op5)

                of_train = 1
                np.random.seed(seed=seed2)
                torch.manual_seed(seed2)
                train_loader_op5 = get_mnist_data_dataloader(None, of_train, \
                        overriden_location=overrid_train_data_op5, \
                        shuffle = True)

                if overrid_test_data_op5:
                    of_train = 0
                    #shuling was done while creating overrid_test_data_op5 qq2
                    test_loader_op5 = get_mnist_data_dataloader(None, of_train, overriden_location=overrid_test_data_op5,\
                            shuffle = False )
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
                X_test_op5 = X_test_op5.to(device)
                y_test_op5 = y_test_op5.to(device)

                mlp_op5 = MLPClassifier(input_size, hidden_sizes, output_size).to(device)

                optimizer_op5 = optim.Adam(mlp_op5.parameters(), lr=clf['learning_rate_init'], \
                        betas=(clf['beta_1'], clf['beta_2']),eps=clf['epsilon'])

                assert np.array_equal(y_test_op4.cpu().numpy(), y_test_op5.cpu().numpy())

                mlp_op5 = train(train_loader_op5, mlp_op5, optimizer_op5, clf, num_epochs)

                # ---

                res_dict = test2_100runs(nR, mlp_op4, X_test_op4, y_test_op4, \
                        mlp_op5, X_test_op5, y_test_op5, num_epochs, res_dict,\
                        seed2, n_run, filenames_op4 = filenames_op4, filenames_op5 = filenames_op5)

                tmp=10

            #for nR, overrid_train_data in enumerate(overrid_train_data_s):
        #for num_epochs in [200, 100, 50, 25, 15, 6, 3]:
    #for n_run in range(N_RUNS):

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