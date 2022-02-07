import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from part_1_additional_functions import *
import os
from datetime import datetime
import optimizer_config

NUM_OF_ITERATION = 10
NUM_OF_EPOCHES = 10


def experiment_A(mnist_data_set):

    curr_time_str = datetime.now().strftime("%m.%d.%Y.%H.%M")
    output_dir = 'output'
    output_curr_time_dir = os.path.join(output_dir, curr_time_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_curr_time_dir)

    for cur_binary_problem in optimizer_config.BinaryProblem:
        print(f'========= starting work on binary problem: {cur_binary_problem.name} =============')
        for cur_optimization in optimizer_config.OptimizerOptions:
            hyper_params_options = 1
            if cur_optimization == optimizer_config.OptimizerOptions.RegularizedGD:
                hyper_params_options = len(optimizer_config.RGD_different_params_list)
            for hyper_params_idx in range(hyper_params_options):
                print(f'--------------- optimization: {cur_optimization.name} ---------------')
                losses_per_optimiation_method = np.zeros((NUM_OF_ITERATION, NUM_OF_EPOCHES))
                for ii in range(NUM_OF_ITERATION):
                    print(f'--> iteration number {ii + 1}:')
                    train_set, test_set = torch.utils.data.random_split(mnist_data_set, [60000, 10000])
                    losses, w = run_single_experiment(train_set, cur_binary_problem, cur_optimization, hyper_params_idx)
                    losses_per_optimiation_method[ii, :] = np.array(losses)

                average_losses = np.mean(losses_per_optimiation_method, axis=0)
                save_fig_path = os.path.join(output_curr_time_dir, f'opt_{cur_optimization.name}_losses.png')
                plot_losses(average_losses, save_fig_path)


def run_single_experiment(mnist_data_set, binary_problem_name, optimization_name, rgd_hyper_params):
    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels)  # initialization
    losses = []  # resulted loss in each iteration

    tagging_method = optimizer_config.binary_type_to_function_dic[binary_problem_name]
    if optimization_name == optimizer_config.OptimizerOptions.RegularizedGD:
        # run different hyper params for RGD (q3)
        print(f'--------------- Hyper Params Option: {rgd_hyper_params} ---------------')
        opt, batch_size = init_rgd_optimizer(optimizer_config.RGD_different_params_list[rgd_hyper_params], w)
    else:
        opt, batch_size = init_optimizer(optimization_name, w)

    data_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=batch_size, shuffle=True)

    for e in range(NUM_OF_EPOCHES):
        steps_loss = []
        for samples, labels in data_loader:
            samples = samples.view(-1, num_of_pixels)
            labels = tagging_method(labels)
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            print('loss = {}'.format(loss))
            steps_loss.append(loss)
        losses.append(np.mean(steps_loss))

    return losses, w


def init_optimizer(optimization_name, w):
    hyper_params = optimizer_config.GD_type_to_params_dic[optimization_name]
    opt = Optimizer(w, hyper_params)
    return opt, hyper_params.batch_size


def init_rgd_optimizer(hyper_params, w):
    opt = Optimizer(w, hyper_params)
    return opt, hyper_params.batch_size


def plot_losses(losses, save_fig_path):
    plt.figure()
    plt.plot(np.arange(1, len(losses)+1), losses)
    # plt.show(block=False)
    plt.savefig(save_fig_path)
