#=================================================
# UTILS
# Making People Smile
#=================================================
#Stephania Valdivia Diaz
#Fundamentos de IA
#=================================================
"""
Author: Sebastian Raschka
https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L17

Python implementation: CPython
Python version       : 3.8.8
IPython version      : 7.21.0

torch: 1.8.1

"""

import random
import torch
import numpy as np
import os


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
    
    
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)