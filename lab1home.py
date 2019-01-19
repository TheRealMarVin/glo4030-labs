import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
# %matplotlib inline

from deeplib.datasets import load_mnist, load_cifar10, train_valid_loaders
from sklearn.metrics import accuracy_score
from deeplib.net import MnistNet, CifarNet
from deeplib.history import History
from deeplib.visualization import plot_images

from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SequentialSampler