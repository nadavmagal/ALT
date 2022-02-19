import os
import matplotlib.pyplot as plt
import numpy as np

experiment_a_united_res_path = r'/home/tal/Personal/School/advanced_learning/project/ALT/output/united'
optimizations_to_take = os.listdir(os.path.join(experiment_a_united_res_path, 'BIGGER_THAN_5'))
regularized_experiment_folder = r'/home/tal/Personal/School/advanced_learning/project/ALT/output/all_RGD_results'
colors_to_plot = ['green', 'cyan', 'magenta', 'black', 'magenta']

for binary_problem in os.listdir(experiment_a_united_res_path):
    plt.figure()
    plt.title(binary_problem)
    for ii, opt in enumerate(optimizations_to_take):
        opt_loss_array = np.load(os.path.join(experiment_a_united_res_path, binary_problem, opt, f'opt_{opt}_0_losses_train.npy'))
        plt.plot(np.arange(1, len(opt_loss_array) + 1), opt_loss_array, color=colors_to_plot[ii], label=opt)
    plt.grid()
    plt.legend()
    plt.show(block=False)

RGD_VALS_DICT = ['0.04', '0.05', '0.02', '0.035']
plt.figure()
for ii in range(4):
    opt_loss_array = np.load(os.path.join(regularized_experiment_folder, f'opt_RegularizedGD_3_{ii}_losses_train.npy'))
    plt.plot(np.arange(1, len(opt_loss_array) + 1)[:300], opt_loss_array[:300], color=colors_to_plot[ii], label=RGD_VALS_DICT[ii])
plt.grid()
plt.legend()
plt.title('ODD_EVEN - RGD training loss')
plt.show(block=False)
a=0