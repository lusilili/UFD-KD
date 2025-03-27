import mmcls.engine.optimizers.layer_decay_optim_wrapper_constructor
from mmcls.registry import OPTIM_WRAPPER_CONSTRUCTORS

print("Registered constructors:", OPTIM_WRAPPER_CONSTRUCTORS.module_dict.keys())
