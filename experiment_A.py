import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from part_1_additional_functions import *
import os
from datetime import datetime
from optimizer_config import BinaryProblem, OptimizerOptions, GD_type_to_params_dic, binary_type_to_function_dic



def experiment_A(mnist_data_set):
    curr_time_str = datetime.now().strftime("%m.%d.%Y.%H.%M")
    output_dir = 'output'
    output_curr_time_dir = os.path.join(output_dir, curr_time_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_curr_time_dir, exist_ok=True)

    for cur_binary_problem in BinaryProblem:
        print(f'========= starting work on binary problem: {cur_binary_problem.name} =============')
        for cur_optimization in OptimizerOptions:
            print(f'--------------- optimization: {cur_optimization.name} ---------------')
            len_of_hyper_params_options = len(GD_type_to_params_dic[cur_optimization])
            for hyper_params_idx in range(len_of_hyper_params_options):
                cur_hyper_params = GD_type_to_params_dic[cur_optimization][hyper_params_idx]
                losses_per_optimization_method = np.zeros((cur_hyper_params.num_of_iteration, cur_hyper_params.num_of_epochs))
                test_losses_per_optimization_method = np.zeros((cur_hyper_params.num_of_iteration, cur_hyper_params.num_of_epochs))
                test_accuracy_per_optimization_method = np.zeros((cur_hyper_params.num_of_iteration, cur_hyper_params.num_of_epochs))
                subset_mnist_data_set = torch.utils.data.Subset(mnist_data_set, range(0,cur_hyper_params.data_set_size))
                if cur_optimization == OptimizerOptions.RegularizedGD:
                    print(f'----- Hyper Params Option: {hyper_params_idx + 1} -----')
                for ii in range(cur_hyper_params.num_of_iteration):
                    print(f'--> iteration number {ii + 1}:')
                    losses, test_losses, test_accuracies = run_single_experiment(subset_mnist_data_set, cur_binary_problem, cur_optimization, hyper_params_idx, cur_hyper_params)
                    losses_per_optimization_method[ii, :] = np.array(losses)
                    test_losses_per_optimization_method[ii, :] = np.array(test_losses)
                    test_accuracy_per_optimization_method[ii, :] = np.array(test_accuracies)

                average_losses = np.mean(losses_per_optimization_method, axis=0)
                save_fig_path = os.path.join(output_curr_time_dir, f'opt_{cur_optimization.name}_{hyper_params_idx}_losses.png')
                plot_losses(average_losses, save_fig_path)

                test_average_losses = np.mean(test_losses_per_optimization_method, axis=0)
                test_save_fig_path = os.path.join(output_curr_time_dir, f'opt_{cur_optimization.name}_{hyper_params_idx}_losses_test.png')
                plot_losses(test_average_losses, test_save_fig_path, best_val=np.min(test_average_losses))

                test_average_acc = np.mean(test_accuracy_per_optimization_method, axis=0)
                test_acc_save_fig_path = os.path.join(output_curr_time_dir, f'opt_{cur_optimization.name}_{hyper_params_idx}_acc_test.png')
                plot_losses(test_average_acc, test_acc_save_fig_path, best_val=np.max(test_average_acc))

                save_losses_arrays(average_losses, test_average_losses, test_average_acc, output_curr_time_dir, f'{cur_optimization.name}_{hyper_params_idx}')


def run_single_experiment(mnist_data_set, binary_problem_name, optimization_name, rgd_hyper_params_idx, cur_hyper_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'available processor: {device}')

    num_of_pixels = 28 * 28

    w = torch.randn(num_of_pixels, device=device)  # initialization
    train_losses = []  # resulted loss in each iteration
    test_losses = []
    test_accuracies = []

    tagging_method = binary_type_to_function_dic[binary_problem_name]
    opt, batch_size = init_optimizer(optimization_name, rgd_hyper_params_idx, w)
    data_set_size = len(mnist_data_set)
    test_size = int(data_set_size * cur_hyper_params.test_percentage)
    train_size = data_set_size - test_size
    train_set, test_set = torch.utils.data.random_split(mnist_data_set, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    for e in range(cur_hyper_params.num_of_epochs):
        steps_loss = []
        for samples, labels in train_loader:
            samples = samples.view(-1, num_of_pixels).to(device)
            labels = tagging_method(labels).to(device)
            outputs = samples @ w
            w, loss = opt.step(outputs, labels, samples)
            print('loss = {}'.format(loss.cpu()))
            steps_loss.append(loss.cpu())
        train_losses.append(np.mean(steps_loss))

        steps_test_losses = []
        steps_test_acc = []
        for test_samples, test_labels in test_loader:
            test_samples = test_samples.view(-1, num_of_pixels).to(device)
            test_labels = tagging_method(test_labels).to(device)
            test_outputs = test_samples @ w
            test_loss = opt.loss.calc_loss(test_outputs, test_labels) + opt.reg * (opt.w ** 2).sum()
            test_acc = calc_zero_one_loss(test_outputs.cpu(), test_labels.cpu())
            steps_test_losses.append(test_loss.cpu())
            steps_test_acc.append(test_acc)
        test_losses.append(torch.mean(torch.stack(steps_test_losses)))
        test_accuracies.append(np.mean(steps_test_acc))
        print(f'acc = {np.mean(steps_test_acc)}')

    return train_losses, test_losses, test_accuracies


def init_optimizer(optimization_name, hyper_params_idx, w):
    hyper_params = GD_type_to_params_dic[optimization_name][hyper_params_idx]
    opt = Optimizer(w, hyper_params)
    return opt, hyper_params.batch_size


def plot_losses(losses, save_fig_path, best_val=None):
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    if best_val:
        plt.title(f'best value is {best_val}')
    plt.savefig(save_fig_path)


def save_losses_arrays(average_losses, test_average_losses, test_average_acc, output_curr_time_dir, suffix):
    train_save_path = os.path.join(output_curr_time_dir, f'opt_{suffix}_losses_train')
    np.save(train_save_path, average_losses)
    test_save_path = os.path.join(output_curr_time_dir, f'opt_{suffix}_losses_test')
    np.save(test_save_path, test_average_losses)
    test_acc_save_path = os.path.join(output_curr_time_dir, f'opt_{suffix}_acc_test')
    np.save(test_acc_save_path, test_average_acc)
