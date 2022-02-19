from enum import Enum
from binary_problems import tag_odd_even, tag_bd_date, tag_is_big_from_5


class OptimizerOptions(Enum):
    GD = 0,
    # ConstrainedGD = 1,
    # RegularizedGD = 2,
    # SGD = 3,
    # RegularizedGD_3 = 4,
    # GD_part_3 = 5


class LossFuncTypes(Enum):
    square_loss = 0
    hinge_loss = 1


class OptimizerHyperParams:
    def __init__(self, gd_type, lr, k=None, reg=0.0, loss_function_type=LossFuncTypes.square_loss, num_of_epochs=500,
                 data_set_size=70000, num_of_iteration=10):
        self.opt_type = gd_type
        self.learning_rate = lr
        self.k = k
        self.reg = reg

        self.batch_size = data_set_size
        self.loss_function_type = loss_function_type
        self.num_of_epochs = num_of_epochs
        self.data_set_size = data_set_size
        self.num_of_iteration = num_of_iteration
        self.test_percentage = 1 / 7


GD_type_to_params_dic = {
    # OptimizerOptions.GD: [OptimizerHyperParams(OptimizerOptions.GD, 0.01)],
    # OptimizerOptions.ConstrainedGD: [OptimizerHyperParams(OptimizerOptions.ConstrainedGD, 0.01, k=1)],
    # OptimizerOptions.RegularizedGD: [OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.01, reg=0.035)],
    # OptimizerOptions.SGD: [OptimizerHyperParams(OptimizerOptions.SGD, 0.01)],
    # OptimizerOptions.RegularizedGD_3: [OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.05, reg=0.035),
    #                                    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.001, reg=0.1),
    #                                    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.1, reg=0.06)],
    OptimizerOptions.GD: [
        OptimizerHyperParams(OptimizerOptions.GD, 0.9, loss_function_type=LossFuncTypes.hinge_loss, num_of_epochs=500,
                             data_set_size=30000, num_of_iteration=1),
        OptimizerHyperParams(OptimizerOptions.GD, 0.5, loss_function_type=LossFuncTypes.hinge_loss, num_of_epochs=500,
                             data_set_size=30000, num_of_iteration=1),
        OptimizerHyperParams(OptimizerOptions.GD, 0.1, loss_function_type=LossFuncTypes.hinge_loss, num_of_epochs=500,
                             data_set_size=30000, num_of_iteration=1),
        OptimizerHyperParams(OptimizerOptions.GD, 0.05, loss_function_type=LossFuncTypes.hinge_loss, num_of_epochs=500,
                             data_set_size=30000, num_of_iteration=1),
        OptimizerHyperParams(OptimizerOptions.GD, 0.01, loss_function_type=LossFuncTypes.hinge_loss, num_of_epochs=500,
                             data_set_size=30000, num_of_iteration=1)]
}


class BinaryProblem(Enum):
    ODD_EVEN = 0,
    # BIGGER_THAN_5 = 1,
    # IS_MY_BDAY = 2


binary_type_to_function_dic = {
    BinaryProblem.ODD_EVEN: tag_odd_even,
    # BinaryProblem.BIGGER_THAN_5: tag_is_big_from_5,
    # BinaryProblem.IS_MY_BDAY: tag_bd_date,
}