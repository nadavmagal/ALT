from enum import Enum
from binary_problems import tag_odd_even, tag_divide_by_3, tag_is_big_from_4


class OptimizerOptions(Enum):
    GD = 0,
    ConstrainedGD = 1,
    RegularizedGD = 2,
    SGD = 3,
    RegularizedGD_3 = 4,
    GD_part_3 = 5


class LossFuncTypes(Enum):
    square_loss = 0
    hinge_loss = 1


class OptimizerHyperParams:
    def __init__(self, gd_type, lr, k=None, reg=0.0, loss_function_type=LossFuncTypes.square_loss):
        self.opt_type = gd_type
        self.learning_rate = lr
        self.k = k
        self.reg = reg
        if gd_type == OptimizerOptions.SGD:
            self.batch_size = 32
        else:
            self.batch_size = 70000
        self.loss_function_type = loss_function_type


GD_type_to_params_dic = {
    OptimizerOptions.GD: [OptimizerHyperParams(OptimizerOptions.GD, 0.7)],
    OptimizerOptions.ConstrainedGD: [OptimizerHyperParams(OptimizerOptions.ConstrainedGD, 0.05, k=1)],
    OptimizerOptions.RegularizedGD: [OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.61, reg=0.035)],
    OptimizerOptions.SGD: [OptimizerHyperParams(OptimizerOptions.SGD, 0.96)],
    OptimizerOptions.RegularizedGD_3: [OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.7, reg=0.035),
                                       OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.5, reg=0.035),
                                       OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.5, reg=0.06)],
    OptimizerOptions.GD_part_3: [
        OptimizerHyperParams(OptimizerOptions.GD, 0.01, loss_function_type=LossFuncTypes.hinge_loss),
        OptimizerHyperParams(OptimizerOptions.GD, 0.01, loss_function_type=LossFuncTypes.hinge_loss),
        OptimizerHyperParams(OptimizerOptions.GD, 0.01, loss_function_type=LossFuncTypes.hinge_loss)]
}


class BinaryProblem(Enum):
    ODD_EVEN = 0,
    BIGGER_THAN_4 = 1,
    DIVIDE_BY_3 = 2


binary_type_to_function_dic = {
    BinaryProblem.ODD_EVEN: tag_odd_even,
    BinaryProblem.BIGGER_THAN_4: tag_is_big_from_4,
    BinaryProblem.DIVIDE_BY_3: tag_divide_by_3,
}
