import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--inr', type=str, default='sinsin.npy')
args = parser.parse_args()

inr = np.load(args.inr)
plt.imshow(inr)
plt.show()