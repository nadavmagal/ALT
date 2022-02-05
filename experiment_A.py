import numpy as np
import matplotlib.pyplot as plt
from part_1_additional_functions import *
import time

NUM_OF_ITERATION = 10
NUM_OF_EPOCHES = 1000
LR = 0.1

def experiment_A(mnist_data_set, binary_problems, optimization):
    for cur_binary_problem in binary_problems:
        for cur_optimization in optimization:
            for ii in range(NUM_OF_ITERATION):
                run_single_experiment(mnist_data_set, cur_binary_problem, cur_optimization)


def run_single_experiment(mnist_data_set, binary_problem_name, optimization_name):
    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels)  # initialization
    opt = Optimizer(w, lr=LR)

    data_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=70000, shuffle=True)

    for e in range(NUM_OF_EPOCHES):
        for samples, labels in data_loader:
            samples = samples.view(-1, num_of_pixels)
            labels = labels % 2
            labels[labels < 1] = -1
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            print('loss = {}'.format(loss))
