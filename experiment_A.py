import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from part_1_additional_functions import *
import os
from datetime import datetime

NUM_OF_ITERATION = 10
NUM_OF_EPOCHES = 10
LR = 0.1


def experiment_A(mnist_data_set, binary_problems, optimization):

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
                losses, w = run_single_experiment(train_set, cur_binary_problem, cur_optimization)
                losses_per_optimiation_method[ii, :] = np.array(losses)

            average_losses = np.mean(losses_per_optimiation_method, axis=0)
            save_fig_path = os.path.join(output_curr_time_dir, f'opt_{cur_optimization}_losses.png')
            plot_losses(average_losses, save_fig_path)
    a=0


def tag_odd_even(labels):
    labels = labels % 2
    return labels


def tag_is_big_from_5(labels):
    labels[labels >= 5] = 1
    labels[labels < 5] = 0
    return labels


def tag_bd_date(labels):
    labels[(labels == 3) | (labels == 1) | (labels == 9) | (labels == 0)] = 1
    labels[labels != 1] = 0
    return labels


def run_single_experiment(mnist_data_set, binary_problem_name, optimization_name):
    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels)  # initialization
    losses = []

    tagging_method = which_tagging_method(binary_problem_name)
    opt, batch_size = which_opt(optimization_name, w)

    data_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=batch_size, shuffle=True)

    for e in range(NUM_OF_EPOCHES):
        for samples, labels in data_loader:
            samples = samples.view(-1, num_of_pixels)
            labels = tagging_method(labels)
            labels[labels < 1] = -1
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            losses.append(float(loss))
            print('loss = {}'.format(loss))

    return losses, w


def which_opt(optimization_name, w):
    if optimization_name == 'gd':
        opt = Optimizer(w, lr=LR)
        batch_size = 70000
    elif optimization_name == 'constrained_gd':
        # todo: make sure k=1 is alright and maybe change it
        opt = Optimizer(w, lr=LR, K=1)
        batch_size = 70000
    elif optimization_name == 'regularized_gd':
        opt = Optimizer(w, lr=LR, reg=0.035)
        batch_size = 70000
    elif optimization_name == 'sgd':
        opt = Optimizer(w, lr=LR)
        batch_size = 32
    else:
        print(f'optimization name invalid - {optimization_name}\n'
              f'The options are - [gd, constrained_gd, regularized_gd, sgd]')
        return None
    return opt, batch_size


def which_tagging_method(binary_problem_name):
    if binary_problem_name == 'odd_even':
        tagging_method = tag_odd_even
    elif binary_problem_name == 'is_big_from_5':
        tagging_method = tag_is_big_from_5
    elif binary_problem_name == 'is_in_my_bd_date':
        tagging_method = tag_bd_date
    return tagging_method


def plot_losses(losses, save_fig_path):
    plt.figure()
    plt.plot(np.arange(1, len(losses)+1), losses)
    # plt.show(block=False)
    plt.savefig(save_fig_path)
