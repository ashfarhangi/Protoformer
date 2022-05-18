# =============================================================================
# Misc. Utilities 
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from collections import defaultdict
from textwrap import wrap
from joblib import load, dump
import pickle
from tqdm import tqdm
import transformers
import datetime
import matplotlib.pylab as pylab
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import energy_distance
from fastdist import fastdist
RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED) 
device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
