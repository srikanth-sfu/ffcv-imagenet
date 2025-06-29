import numpy as np
from itertools import product
import os

alpha_range = np.linspace(1.1, 1.3, 8) + 0.3 # e.g. [1.1, 1.2, 1.3]
beta_range = np.linspace(0.85, 1.15, 5) + 0.3  # e.g. [1.05, 1.1, 1.15]


candidates = []
for alpha, beta in product(alpha_range, beta_range):
    candidates.append((alpha, beta))


for candidate in candidates:
    os.system('bash train_imagenet.sh %.02f %.02f'%(candidate[0], candidate[1])) 
