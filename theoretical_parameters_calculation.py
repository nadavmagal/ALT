import torch

def calculate_theoretical_hyperparameters(mnist_data_set):
    alpha, beta = calculate_alpha_and_beta(mnist_data_set)

    ''' GD '''
    gd_eta = 1/beta
    print(f'{gd_eta=}')

    '''  '''

    return

def calculate_alpha_and_beta(mnist_data_set):
    image_dim = 28 * 28
    dataset_length = len(mnist_data_set)
    data_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=dataset_length, shuffle=True)
    X = next(iter(data_loader))[0]
    X = X.view(-1, image_dim)
    B = 2 * X.T @ X / dataset_length
    eig_values = torch.eig(B)[0]
    alpha, beta = eig_values.min(), eig_values.max()
    return alpha, beta


