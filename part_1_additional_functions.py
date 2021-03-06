import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
from optimizer_config import LossFuncTypes

def get_mnist_data(mnist_data_fp):
    input_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Lambda(lambda cur_x: cur_x / (cur_x ** 2).sum(dim=(1, 2), keepdim=True).sqrt())
    ])
    # load dataset
    train_set = datasets.MNIST(root=mnist_data_fp,
                               train=True,
                               download=True,
                               transform=input_transforms)

    test_set = datasets.MNIST(root=mnist_data_fp,
                              train=False,
                              download=True,
                              transform=input_transforms)

    # combine datasets
    combined_dataset = [train_set, test_set]
    data_set = torch.utils.data.ConcatDataset(combined_dataset)
    return data_set


# TODO: maybe
class Optimizer:
    def __init__(self, w, hyper_params):
        self.w = w
        self.lr = hyper_params.learning_rate
        self.reg = hyper_params.reg
        self.K = hyper_params.k
        self.loss = self._set_loss_funciton(hyper_params)

    @staticmethod
    def _set_loss_funciton(hyper_params):
        if hyper_params.loss_function_type == LossFuncTypes.square_loss:
            loss = MeanSquareError()
        elif hyper_params.loss_function_type == LossFuncTypes.hinge_loss:
            loss = HingeLoss()
        else:
            loss = None
        return loss

    def step(self, t, y, X):
        """
        :param t: prediction
        :param y: label
        :param X: input
        :return:
        """
        l = self.loss.calc_loss(t, y) + self.reg * (self.w ** 2).sum()
        w = self.w - self.lr * (self.loss.grad(t, y, X) + self.reg * 2 * self.w)
        if self.K is not None:
            # verify if should be elementwise
            w = w.clamp(min=-self.K, max=self.K)
        self.w = w
        return w, l


# TODO: maybe
class HingeLoss:
    def __init__(self):
        return

    @staticmethod
    def calc_loss(t, y):
        """
        :param t: prediction
        :param y: label
        :return: loss
        """

        return torch.max(torch.zeros_like(y), 1 - y * t).mean()

    @staticmethod
    def grad(t, y, X):
        """
        :param t: [m,1]
        :param y: [m,1]
        :param X: [m,d]
        :return: grad
        """
        m = X.shape[0]
        X[t * y > 1, :] = 0
        return -1 / m * X.T @ y.type_as(X)


class MeanSquareError:
    def __init__(self):
        return

    @staticmethod
    def calc_loss(t, y):
        """
        :param t: prediction
        :param y: label
        :return: loss
        """
        return (t-y).square().mean()

    @staticmethod
    def grad(t, y, X):
        """
        :param t: [m,1]
        :param y: [m,1]
        :param X: [m,d]
        :return: grad
        """
        return 2*(t-y) @ X / X.shape[0]


class LinearModel(nn.Module):
    def __init__(self, dim):
        super(LinearModel, self).__init__()
        self.w = nn.Parameter(torch.randn(dim))

    def forward(self, X):
        # X is [m,d]
        # m - batch size, d - sample size
        return X @ self


def calc_zero_one_loss(t, y):
    """
    :param t: prediction
    :param y: label
    :return: loss
    """
    correct = len(np.where(abs(t - y) <= 1)[0])
    return correct / t.shape[0]