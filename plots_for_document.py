import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    result_dir = r'/home/nadav/Downloads/tal_results/'
    problem_to_plot = ['IS_DIVIDED_BY_THREE', 'BIGGER_THAN_5', 'ODD_EVEN']
    optimization_to_plot = ['GD', 'ConstrainedGD', 'RegularizedGD', 'SGD']
    opt_name_for_legend = {
        'GD': 'gradient decent',
        'ConstrainedGD': 'constrained gradient decent',
        'RegularizedGD': 'regularized gradient decent',
        'SGD': 'stochastic gradient decent',
    }
    problem_name_for_legend = {
        'IS_DIVIDED_BY_THREE': 'is divided by 3',
        'ODD_EVEN': 'is odd or even',
        'BIGGER_THAN_5': 'is bigger than 5',
    }

    for cur_problem in problem_to_plot:
        plt.figure()
        for cur_optimization_to_plot in optimization_to_plot:
            cur_problem_dir = os.path.join(result_dir, cur_problem, cur_optimization_to_plot)
            train_file_name = [cur_name for cur_name in os.listdir(cur_problem_dir) if 'osses_train.npy' in cur_name][0]
            cur_train_loss = np.load(os.path.join(cur_problem_dir, train_file_name))
            if cur_optimization_to_plot == 'RegularizedGD':
                start = 22
            else:
                start = 0
            plt.plot(cur_train_loss[start:500+start], label=opt_name_for_legend[cur_optimization_to_plot])
        plt.legend()
        plt.title(f'problem: "{problem_name_for_legend[cur_problem]}"')
        plt.ylim([0, 4])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.show(block=False)

    return


if __name__ == "__main__":
    main()
