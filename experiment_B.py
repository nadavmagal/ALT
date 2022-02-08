import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from ALT import optimizer_config
from part_1_additional_functions import *
import os
from datetime import datetime
from experiment_A import init_rgd_optimizer

NUM_OF_ITERATION = 10
NUM_OF_EPOCHES = 1000


def experiment_B(mnist_data_set, binary_problems, optimization):

    curr_time_str = datetime.now().strftime("%m.%d.%Y.%H.%M")
    output_dir = './output'
    output_curr_time_dir = os.path.join(output_dir, curr_time_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_curr_time_dir)

    for cur_binary_problem in binary_problems:
        print(f'========= starting work on binary problem: {cur_binary_problem} =============')
        for cur_optimization in optimization:
            print(f'--------------- optimization: {cur_optimization} ---------------')
            losses_per_optimiation_method = np.zeros((NUM_OF_ITERATION, NUM_OF_EPOCHES))
            for ii in range(NUM_OF_ITERATION):
                print(f'--> iteration number {ii + 1}:')
                train_set, test_set = torch.utils.data.random_split(mnist_data_set, [60000, 10000])
                losses, w = run_single_experiment(train_set, test_set, cur_binary_problem, cur_optimization)
                losses_per_optimiation_method[ii, :] = np.array(losses)


def run_single_experiment(train_set, test_set, binary_problem_name, optimization_name):
    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels)  # initialization
    losses = []

    tagging_method = optimizer_config.binary_type_to_function_dic[binary_problem_name](binary_problem_name)
    if optimization_name == optimizer_config.OptimizerOptions.RegularizedGD:
        # run different hyper params for RGD (q3)
        opt, batch_size = init_rgd_optimizer(rgd_hyper_params_idx, w)
    else:
        opt, batch_size = init_optimizer(optimization_name, w)
    opt, batch_size = which_opt(optimization_name, w)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    for e in range(NUM_OF_EPOCHES):
        for samples, labels in train_loader:
            samples = samples.view(-1, num_of_pixels)
            labels = tagging_method(labels)
            labels[labels < 1] = -1
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            losses.append(float(loss))
            print('loss = {}'.format(loss))

        # todo: add a test evaluation for each epoch

    return losses, w

