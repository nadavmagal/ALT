import torch

def calculate_theoretical_hyperparameters(mnist_data_set):
    ''' split data to train and test '''
    train_set, test_set = torch.utils.data.random_split(mnist_data_set, [60000, 10000])

    alpha, beta = calculate_alpha_and_beta(train_set)
    print(f'alpha={alpha}')
    print(f'beta={beta}')

    ''' GD '''
    gd_eta = 1/beta
    print(f'gd_eta={gd_eta}')

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


