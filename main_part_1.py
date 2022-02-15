import numpy as np
import matplotlib.pyplot as plt
import torch

from part_1_additional_functions import *
from experiment_A import experiment_A
import time


def main():
    #binary_problems = ['odd_even', 'is_big_from_5', 'is_in_my_bd_date']
    #optimization = ['gd', 'constrained_gd', 'regularized_gd', 'sgd']

    mnist_data_fp = './mnist'
    mnist_data_set = get_mnist_data(mnist_data_fp)

    ''' theirs calc alpha'''
    # #TODO: change
    #
    # m = len(mnist_data_set)
    # d = 28 * 28
    # reg =0# 1 / d ** 0 / 5
    # dl = torch.utils.data.DataLoader(mnist_data_set, batch_size=m, shuffle=True)
    # X = next(iter(dl))[0]
    # X = X.view(-1, d)
    # B = 2 * X.T @ X / m + reg * torch.eye(d)
    # eig = torch.eig(B)[0]
    # eig.min(), eig.max()
    #
    # '''calc alpha'''
    # sum_of_grad_2 = torch.zeros([784,784])
    # mnist_len = len(mnist_data_set)
    # for ii, cur_data in enumerate(mnist_data_set):
    #     print(ii)
    #     cur_x = cur_data[0]
    #     cur_y = cur_data[1]
    #
    #     flatten_x = torch.unsqueeze(torch.flatten(cur_x),1)
    #     x_x_T = 2*torch.matmul(flatten_x, torch.transpose(flatten_x, 0, 1))
    #     #TODO - outer matmul, suppose to be a matrix. and than taking the min eigan value
    #     sum_of_grad_2 += x_x_T
    # a = torch.eig(sum_of_grad_2)
    # mean_sum = sum_of_grad_2/mnist_len
    # a=3
    # b = torch.eig(mean_sum)


    a=3

    experiment_A(mnist_data_set)
    # experiment_B(mnist_data_set, binary_problems, optimization)

    return


if __name__ == "__main__":
    main()
