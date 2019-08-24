import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as k_backend
from random_forest import WaveletsForestRegressor
from keras_mnist import read_data
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from models import Generator, Discriminator, FeatureExtractor

batch_size = 20000
# Load the data
def load_dataset():
    data_path = '/Users/nadav.nagel/Desktop/Studying/ShayDekel/PyTorch-SRGAN/output/'

    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    return train_loader

# x_train, x_test, y_train, y_test, input_shape = read_data()


def calc_smoothness(x, y):
    print('dim is :{}'.format(x.shape))
    wfr = WaveletsForestRegressor(regressor='random_forest', criterion='mse', depth=9, trees=5)
    wfr.fit(x, y)
    alpha, n_wavelets, errors = wfr.evaluate_smoothness(m=1000)
    return alpha


def plot_vec(x=0, y=None, title='', xaxis='', yaxis='', epoch=1):
    if x == 0:
        x = range(1, len(y) + 1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()
    plt.savefig('smothness_epoch{}.png'.format(epoch))



def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def main():

    model = Discriminator()
    model.load_state_dict(
        torch.load('/Users/nadav.nagel/Desktop/Studying/ShayDekel/from_vm/discriminator_epoch_98_after_dist_and_gen.pth',
                   map_location='cpu'))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = load_dataset()
    alpha_vec = np.zeros((len(model._modules.items()), ))

    for i, data in enumerate(train_loader):
        x, y = data
        y = one_hot(y, 2)

        for j in range(batch_size):
            x[j] = normalize(x[j])

        for idx, layer in enumerate(model._modules.items()):
            layer_output = x
            print('Calculating smoothness parameters for layer '+str(idx)+'.')
            for idx_j, layer_j in enumerate(model._modules.items()):
                layer_output = layer_j[1](layer_output)
                if idx_j == idx:
                    layer_output = layer_output.detach().numpy()
                    break

            alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).T, y.detach().numpy())
            print('For Layer {}, alpha is: {}'.format(idx, alpha_vec[idx]))
    plot_vec(y=alpha_vec, title='Smoothness over layers', xaxis='Alpha', yaxis='#Layer')


if '__main__' == __name__:
    main()
