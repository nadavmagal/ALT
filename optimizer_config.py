from enum import Enum
import binary_problems


class OptimizerOptions(Enum):
    GD = 0,
    ConstrainedGD = 1,
    RegularizedGD = 2,
    SGD = 3,


class OptimizerHyperParams:
    def __init__(self, gd_type, lr, k=None, reg=0.0):
        self.type = type
        self.learning_rate = lr
        self.k = k
        self.reg = reg
        if gd_type == OptimizerOptions.SGD:
            self.batch_size = 32
        else:
            self.batch_size = 70000


GD_type_to_params_dic = {
    OptimizerOptions.GD: OptimizerHyperParams(OptimizerOptions.GD, 0.1),
    OptimizerOptions.ConstrainedGD: OptimizerHyperParams(OptimizerOptions.ConstrainedGD, 0.1, k=1),
    OptimizerOptions.RegularizedGD: OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.1, reg=0.035),
    OptimizerOptions.SGD: OptimizerHyperParams(OptimizerOptions.SGD, 0.1)
}


class BinaryProblem(Enum):
    ODD_EVEN = 0,
    BIGGER_THAN_5 = 1,
    IS_MY_BDAY = 2


binary_type_to_function_dic = {
    BinaryProblem.ODD_EVEN: binary_problems.tag_odd_even,
    BinaryProblem.BIGGER_THAN_5: binary_problems.tag_is_big_from_5,
    BinaryProblem.IS_MY_BDAY: binary_problems.tag_bd_date,
}

# TODO: change this according to q3
RGD_different_params_list = [
    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.1, reg=0.035),
    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.2, reg=0.035),
    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.3, reg=0.035),
    OptimizerHyperParams(OptimizerOptions.RegularizedGD, 0.1, reg=0.06),
]