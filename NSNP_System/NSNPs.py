# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt




# the trasfer function of NSNP,  tanh is used here.
def transferFunc(x, belta=1, flag='-01'): 
    if flag == '-01':
        return np.tanh(x)
    else:
        return 1 / (1 + np.exp(-belta * x))


