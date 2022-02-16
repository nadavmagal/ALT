import numpy as np
import matplotlib.pyplot as plt
import torch
from theoretical_parameters_calculation import calculate_theoretical_hyperparameters
from part_1_additional_functions import *
from experiment_A import experiment_A
import time

CALCULATE_THEORETICAL_HYPER_PARAMS = False


def main_part_1():
    #binary_problems = ['odd_even', 'is_big_from_5', 'is_in_my_bd_date']
    #optimization = ['gd', 'constrained_gd', 'regularized_gd', 'sgd']

    mnist_data_fp = './mnist'
    mnist_data_set = get_mnist_data(mnist_data_fp)

    if CALCULATE_THEORETICAL_HYPER_PARAMS:
        calculate_theoretical_hyperparameters(mnist_data_set)

    experiment_A(mnist_data_set)
    # experiment_B(mnist_data_set, binary_problems, optimization)

    return


if __name__ == "__main__":
    main_part_1()
