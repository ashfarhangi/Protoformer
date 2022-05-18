# =============================================================================
# Trainer module 
# =============================================================================
import model,metric,utils
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
def train(epoch,model):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    # progress bar
    train_per_epoch = num_of_batches_per_epoch
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, 
                      num_epochs=EPOCHS, width=8, 
                      always_stateful=False)

    for idx,data in enumerate(training_loader, 0):
        # copy tensors to gpu
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        # get output and calculate loss.
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += acc_cal(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
      
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()
        kbar.update(idx, values=[("train_loss", tr_loss/(idx+1))])

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    # Comment them out for faster training
    test_acc,test_loss,predicted_labels,true_labels, predicted_raw= valid(model, testing_loader)
    print(" - ")
    print("test accuracy:",round(test_acc,2))
    print("test loss:",round(test_loss,2))
    history['train_acc'].append(epoch_accu)
    history['train_loss'].append(epoch_loss)
    history['test_acc_while_training'].append(test_acc)
    history['test_loss_while_training'].append(test_loss)

    # print(f"Training Loss Epoch: {epoch_loss}")
    # print(f"Training Accuracy Epoch: {epoch_accu}")

    return
# function to predict output.
def valid(model, testing_loader):
    predicted_raw = []
    predicted_labels = []
    true_labels = []
    nb_tr_steps = 0
    tr_loss =0
    nb_tr_examples=0
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            # copy tensors to gpu.
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            # calculate loss
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            predicted_raw += outputs
            predicted_labels += big_idx
            true_labels += targets
            n_correct += acc_cal(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    history['val_acc'].append(epoch_accu)
    history['val_loss'].append(epoch_loss)    
    return epoch_accu,epoch_loss,predicted_labels,true_labels,predicted_raw

def test_model(model, testing_loader):
    predicted_labels = []
    true_labels = []
    nb_tr_steps = 0
    tr_loss =0
    nb_tr_examples=0
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            # copy tensors to gpu.
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()

            # calculate loss
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            torch.max
            predicted_labels += big_idx
            true_labels += targets
            
            n_correct += acc_cal(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    return epoch_accu,epoch_loss,predicted_labels,true_labels        

from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
writer = SummaryWriter()
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
for epoch in range(EPOCHS):
    train(epoch,baseline_model)
    print('\n')

test_acc,test_loss,predicted_labels,true_labels, predicted_raw= valid(baseline_model, testing_loader)
predicted_labels = [i.item() for i in predicted_labels]
true_labels = [i.item() for i in true_labels]
predicted_raw = [i.cpu().detach().numpy() for i in predicted_raw]

accuracy_score(true_labels,predicted_labels)
print(classification_report(true_labels,predicted_labels))
cm = confusion_matrix(true_labels,predicted_labels)
cm.diagonal()/cm.sum(axis=1)

test_df['predicted'] = predicted_labels
test_df['predicted_raw'] = predicted_raw
test_df['wrong'] = 0
for j in range(len(test_df)):
  test_df.loc[j,'predict_c_0'] = predicted_raw[j][0]
  test_df.loc[j,'predict_c_1'] = predicted_raw[j][1]
  if (test_df.loc[j,'labels']!=test_df.loc[j,'predicted']):
    test_df.loc[j,'wrong'] = 1

df_embeddings.reset_index(drop=True,inplace=True)
sns.countplot(df_embeddings.wrong)
accuracy_score(df_embeddings.predicted,df_embeddings.labels)


df_embeddings_copy['larget_logit'] = 0
df_embeddings_copy['confidence'] = 0

for j in range(len(df_embeddings_copy)):
  df_embeddings_copy.loc[j,'confidence'] = abs(abs(df_embeddings_copy.loc[j, 'predict_c_0']) - abs(df_embeddings_copy.loc[j, 'predict_c_1']))
  if df_embeddings_copy.loc[j, 'predict_c_0'] >= 0:
    df_embeddings_copy.loc[j,'larget_logit'] = df_embeddings_copy.loc[j, 'predict_c_0']
  if df_embeddings_copy.loc[j, 'predict_c_1'] >= 0:
    df_embeddings_copy.loc[j,'larget_logit'] = df_embeddings_copy.loc[j, 'predict_c_1']    


fig, ax = plt.subplots()
fig.canvas.draw()

ax = sns.distplot([df_embeddings_copy[df_embeddings_copy.wrong == 0].larget_logit_minmax]
                  ,label = 'Correct',color='black',kde=True,bins=16)
ax = sns.distplot([df_embeddings_copy[df_embeddings_copy.wrong == 1].larget_logit_minmax]
                  ,label = 'Wrong',
                  color='#FF1B1C',bins=8,kde=True)
plt.xlabel('Highest Logit')
labels = [item.get_text() for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)

plt.legend(loc=1)
plt.show()    


# https://github.com/talboger/fastdist
def distance_matrix(df_embeddings):
  dim = len(df_embeddings) # - 1500
  s_ij = np.zeros([dim,dim])
  e_ij = np.zeros([dim,dim])
  l1norm_ij = np.zeros([dim,dim])
  l2norm_ij = np.zeros([dim,dim])
  l3norm_ij = np.zeros([dim,dim])
  for j in tqdm(range(dim)):
    for z in range(j,dim):
      c = fastdist.cosine_matrix_to_matrix(df_embeddings.embedding[j], df_embeddings.embedding[z]) 
  return s_ij,e_ij,l1norm_ij,l2norm_ij,l3norm_ij
s_ij,e_ij,l1norm_ij,l2norm_ij,l3norm_ij = distance_matrix(df_embeddings)
mask = np.tril(np.ones_like(s_ij, dtype=bool))
np.fill_diagonal(mask,False)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
s_ij_rank = s_ij.reshape(-1)
s_ij_rank.transpose()
df_similarity_rank = pd.DataFrame(s_ij_rank)
df_similarity_rank = df_similarity_rank.replace(0,np.nan).dropna()
df_similarity_rank.reset_index(drop=True,inplace=True)
df_similarity_rank.describe()
SIM_BASE = df_similarity_rank.sort_values(0).quantile(0.2).values

df_similarity = pd.DataFrame(s_ij)
df_similarity = df_similarity.replace(0,np.nan)
df_similarity.describe()
df_similarity
df_density = df_similarity

for j in tqdm(range(len(df_density))):
  for z in range(j,len(df_density.loc[j])):
    df_density.loc[j,z] = np.sign(df_density.loc[j,z] - SIM_BASE)
df_embeddings['density'] = 0
df_embeddings['average_sim_all_peers'] = 0
for j in range(len(df_density)):
  df_embeddings.loc[j,'density'] = array_density[j].sum()
  df_embeddings.loc[j,'average_sim_all_peers'] = array_similarity[j].mean()

df_embeddings_sorted = df_embeddings.sort_values(['density', 'average_sim_all_peers'],ascending=[False,False])
df_embeddings_sorted.reset_index(drop=True,inplace=True)
# 0,1 > 1 | 2,3,4,5 > 2
# imdb@
# For Twitter
# cp_list = [[],[],[],[],[],[],[],[]]
# For IMDB
cp_list = [[],[]]

for x in range(len(news_groups)):
  proto_list = []
  begin = 0
  z = 0
  p_num = 0
  proto_list.append(df_embeddings_sorted.loc[0,'sample_id'])
  df = df_embeddings_sorted[df_embeddings_sorted.labels == x]
  df.reset_index(drop=True,inplace=True)
  for j in range(1,len(df)):
    if (z<=2**(p_num+1)):
      z = z + 1
    if (z == 2**(p_num+1)):
      z = 0
      end = j
      p = df.loc[begin,'sample_id']
      for b in range(begin, end):
          i_sel = df.loc[b,'sample_id']
          if (np.mean(array_similarity[i_sel][proto_list]) < np.mean(array_similarity[p][proto_list])):
            p = i_sel
      proto_list.append(p)
      begin = j
      p_num = p_num + 1
  cp_list[x] = proto_list[1:]
print(cp_list)  

df_embeddings_sorted = df_embeddings.sort_values(['density', 'average_sim_all_peers'],ascending=[True,False])
df_embeddings_sorted.reset_index(drop=True,inplace=True)

ap_list = [[],[]]
for x in range(len(news_groups)):
  proto_list = []
  begin = 0
  z = 0
  p_num = 0
  df = df_embeddings_sorted[df_embeddings_sorted.labels == x]
  df.reset_index(drop=True,inplace=True)
  for j in range(1,len(df)):
    if (z<=2**(p_num+1)):
      z = z + 1
    if (z == 2**(p_num+1)):
      z = 0
      end = j
      p = df.loc[begin,'sample_id']
      for b in range(begin, end):
          i_sel = df.loc[b,'sample_id']
          if (np.mean(array_similarity[i_sel][proto_list]) < np.mean(array_similarity[p][proto_list])):
            p = i_sel
      proto_list.append(p)
      begin = j
      p_num = p_num + 1
  ap_list[x] = proto_list
print(ap_list)

df_embeddings['class_prototype'] = 0 
for x in range(len(cp_list)):
  for j in cp_list[x]:
    df_embeddings.loc[j,'class_prototype'] = 1


df_embeddings['anomaly_prototype'] = 0 
for x in range(len(ap_list)):
  for j in ap_list[x]:
    df_embeddings.loc[j,'anomaly_prototype'] = 1    


df_embeddings['larget_logit'] = 0
df_embeddings['confidence'] = 0

for j in range(len(df_embeddings)):
  df_embeddings.loc[j,'confidence'] = abs(abs(df_embeddings.loc[j, 'predict_c_0']) - abs(df_embeddings.loc[j, 'predict_c_1']))
  if df_embeddings.loc[j, 'predict_c_0'] >= 0:
    df_embeddings.loc[j,'larget_logit'] = df_embeddings.loc[j, 'predict_c_0']
  if df_embeddings.loc[j, 'predict_c_1'] >= 0:
    df_embeddings.loc[j,'larget_logit'] = df_embeddings.loc[j, 'predict_c_1']  
    
df = df_embeddings[df_embeddings.average_sim_all_peers!=0]
df.reset_index(drop=True,inplace=True)
sns.scatterplot(df.density,df.average_sim_all_peers,s=80,label='Embeddings',c=df.confidence, cmap='gray')
plt.title('Scatter plot of embeddings density for class'+' 0')
plt.xlabel('Proximity')
plt.ylabel('Similarity')
plt.legend()
plt.show()    