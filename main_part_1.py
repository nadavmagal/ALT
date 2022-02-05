import numpy as np
import matplotlib.pyplot as plt
from part_1_additional_functions import *
from experiment_A import experiment_A
import time


def main():
    binary_problems = ['odd_even', 'is_big_from_5', 'is_in_my_bd_date']
    optimization = ['gd', 'constrained_gd', 'regularized_gd', 'sgd']

    mnist_data_fp = './mnist'
    mnist_data_set = get_mnist_data(mnist_data_fp)

    experiment_A(mnist_data_set, binary_problems, optimization)

    return


if __name__ == "__main__":
    main()
