from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import LabelEncoder
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
df_imdb = pd.read_csv('.//data//IMDB.csv')
# df_imdb = df_imdb.sample(2000)
df_imdb.reset_index(drop=True,inplace=True)
sns.countplot(df_imdb.sentiment)
plt.ylabel('Samples')
plt.xlabel('IMDB Movie Sentiments')
plt.show()
# sns.countplot(df_embeddings.predicted_raw_difference)
df = df_imdb
df_profile = df_imdb
df_profile.columns = ['number','doc', 'labels_original']
df_profile['labels'] = df_profile['labels_original']

le = LabelEncoder()
df_profile['labels']= le.fit_transform(df_profile['labels'])

# X = df_profile.review
X = df_profile.doc
y = df_profile.labels
# z = df_profile.user_name
X_train,X_test,y_train,y_test= train_test_split(X,y,stratify=y,test_size=0.2, random_state=47)
print('number of training samples:', len(X_train))
print('number of test samples:', len(X_test))
train_df = pd.DataFrame({'doc':X_train,
                         'labels':y_train})
test_df = pd.DataFrame({'doc':X_test,
                         'labels':y_test})
train_df.reset_index(drop=True,inplace=True)
test_df.reset_index(drop=True,inplace=True)