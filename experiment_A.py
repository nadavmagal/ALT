import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from part_1_additional_functions import *
import os
from datetime import datetime
from optimizer_config import BinaryProblem, OptimizerOptions, GD_type_to_params_dic, binary_type_to_function_dic

NUM_OF_ITERATION = 3
NUM_OF_EPOCHS = 700


def experiment_A(mnist_data_set):
    curr_time_str = datetime.now().strftime("%m.%d.%Y.%H.%M")
    output_dir = 'output'
    output_curr_time_dir = os.path.join(output_dir, curr_time_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_curr_time_dir)

    for cur_binary_problem in BinaryProblem:
        binary_problem_path = os.path.join(output_curr_time_dir, cur_binary_problem.name)
        os.makedirs(binary_problem_path)
        print(f'========= starting work on binary problem: {cur_binary_problem.name} =============')
        for cur_optimization in OptimizerOptions:
            cur_experiment_path = os.path.join(binary_problem_path, cur_optimization.name)
            os.makedirs(cur_experiment_path)
            print(f'--------------- optimization: {cur_optimization.name} ---------------')
            len_of_hyper_params_options = len(GD_type_to_params_dic[cur_optimization])
            for hyper_params_idx in range(len_of_hyper_params_options):
                losses_per_optimization_method = np.zeros((NUM_OF_ITERATION, NUM_OF_EPOCHS))
                test_losses_per_optimization_method = np.zeros((NUM_OF_ITERATION, NUM_OF_EPOCHS))
                test_accuracy_per_optimization_method = np.zeros((NUM_OF_ITERATION, NUM_OF_EPOCHS))
                if cur_optimization == OptimizerOptions.RegularizedGD:
                    print(f'----- Hyper Params Option: {hyper_params_idx + 1} -----')
                for ii in range(NUM_OF_ITERATION):
                    print(f'--> iteration number {ii + 1}:')
                    losses, test_losses, test_accuracies = run_single_experiment(mnist_data_set, cur_binary_problem, cur_optimization, hyper_params_idx)
                    losses_per_optimization_method[ii, :] = np.array(losses)
                    test_losses_per_optimization_method[ii, :] = np.array(test_losses)
                    test_accuracy_per_optimization_method[ii, :] = np.array(test_accuracies)

                average_losses = np.mean(losses_per_optimization_method, axis=0)
                save_fig_path = os.path.join(cur_experiment_path, f'opt_{cur_optimization.name}_{hyper_params_idx}_losses.png')
                plot_losses(average_losses, save_fig_path)

                test_average_losses = np.mean(test_losses_per_optimization_method, axis=0)
                test_save_fig_path = os.path.join(cur_experiment_path, f'opt_{cur_optimization.name}_{hyper_params_idx}_losses_test.png')
                plot_losses(test_average_losses, test_save_fig_path, best_val=np.min(test_average_losses))

                test_average_acc = np.mean(test_accuracy_per_optimization_method, axis=0)
                test_acc_save_fig_path = os.path.join(cur_experiment_path, f'opt_{cur_optimization.name}_{hyper_params_idx}_acc_test.png')
                plot_losses(test_average_acc, test_acc_save_fig_path, best_val=np.max(test_average_acc))

                save_losses_arrays(average_losses, test_average_losses, test_average_acc, cur_experiment_path, f'{cur_optimization.name}_{hyper_params_idx}')


def run_single_experiment(mnist_data_set, binary_problem_name, optimization_name, rgd_hyper_params_idx):
    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels)  # initialization
    train_losses = []  # resulted loss in each iteration
    test_losses = []
    test_accuracies = []

    tagging_method = binary_type_to_function_dic[binary_problem_name]
    opt, batch_size = init_optimizer(optimization_name, rgd_hyper_params_idx, w)
    train_set, val_set, test_set = torch.utils.data.random_split(mnist_data_set, [30000, 25000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    for e in range(NUM_OF_EPOCHS):
        print(f'------- Starting epoch={e} -------')
        steps_loss = []
        for samples, labels in train_loader:
            samples = samples.view(-1, num_of_pixels)
            labels = tagging_method(labels)
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            steps_loss.append(loss)
        train_losses.append(np.mean(steps_loss))

        steps_test_losses = []
        steps_test_acc = []
        for test_samples, test_labels in test_loader:
            test_samples = test_samples.view(-1, num_of_pixels)
            test_labels = tagging_method(test_labels)
            test_outputs = test_samples @ w
            test_loss = opt.loss.calc_loss(test_outputs, test_labels) + opt.reg * (opt.w ** 2).sum()
            test_acc = calc_zero_one_loss(test_outputs, test_labels)
            steps_test_losses.append(test_loss)
            steps_test_acc.append(test_acc)
        test_losses.append(np.mean(steps_test_losses))
        test_accuracies.append(np.mean(steps_test_acc))

        print(f'train loss = {np.mean(steps_loss)} | validation acc = {np.mean(steps_test_acc)}')

    return train_losses, test_losses, test_accuracies


def init_optimizer(optimization_name, hyper_params_idx, w):
    hyper_params = GD_type_to_params_dic[optimization_name][hyper_params_idx]
    opt = Optimizer(w, hyper_params)
    return opt, hyper_params.batch_size


def plot_losses(losses, save_fig_path, best_val=None):
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses, color='green')
    if best_val:
        plt.title(f'best value is {np.round(best_val, 5)}')
    plt.grid()
    plt.savefig(save_fig_path)


def save_losses_arrays(average_losses, test_average_losses, test_average_acc, output_dir, suffix):
    train_save_path = os.path.join(output_dir, f'opt_{suffix}_losses_train')
    np.save(train_save_path, average_losses)
    test_save_path = os.path.join(output_dir, f'opt_{suffix}_losses_test')
    np.save(test_save_path, test_average_losses)
    test_acc_save_path = os.path.join(output_dir, f'opt_{suffix}_acc_test')
    np.save(test_acc_save_path, test_average_acc)
