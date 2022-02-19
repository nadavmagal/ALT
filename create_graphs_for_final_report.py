import os
import matplotlib.pyplot as plt
import numpy as np

experiment_a_united_res_path = r'/home/tal/Personal/School/advanced_learning/project/ALT/output/united'
optimizations_to_take = os.listdir(os.path.join(experiment_a_united_res_path, 'BIGGER_THAN_5'))
regularized_experiment_folder = os.path.join(experiment_a_united_res_path, 'ODD_EVEN', 'united_RGD')
colors_to_plot = ['green', 'blue', 'yellow', 'black', 'magenta']

for binary_problem in os.listdir(experiment_a_united_res_path):
    plt.figure()
    plt.title(binary_problem)
    for ii, opt in enumerate(optimizations_to_take):
        opt_loss_array = np.load(os.path.join(experiment_a_united_res_path, binary_problem, opt, f'opt_{opt}_0_losses_train.npy'))
        plt.plot(np.arange(1, len(opt_loss_array) + 1), opt_loss_array, color=colors_to_plot[ii], label=opt)
    plt.grid()
    plt.legend()
    plt.show(block=False)

plt.figure()
for ii in range(4):
    opt_loss_array = np.load(os.path.join(regularized_experiment_folder, f'opt_RegularizedGD_3_{ii}_losses_train.npy'))
    plt.plot(np.arange(1, len(opt_loss_array) + 1), opt_loss_array, color=colors_to_plot[ii], label=str(ii))
plt.grid()
plt.legend()
plt.show(block=False)
a=0