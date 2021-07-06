import numpy as np 
import pickle as pkl

fname = "$PWD/data/Advection_1D_k1.0_step0.pkl"
file = open(fname, "rb")
pde = pkl.load(file)
file.close()

print(pde)

