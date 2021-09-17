

# Parent dir: 
# %cd /content/drive/MyDrive/0Thiland Coordination

import wget
import pickle
import pandas as pd

import numpy as np
import geopandas as gpd

import pandas as pd
import numpy as np
import pickle
import os
import tarfile
import xarray
from matplotlib import pyplot as plt
from sklearn import  metrics
from scipy.cluster.hierarchy import dendrogram

import rioxarray as xr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.clustering_utils import *


"""## shapefile data"""

new_all_cities_file = os.path.join("GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0", "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
new_all_cities_gdf = gpd.read_file(new_all_cities_file)
new_all_cities_gdf_filterd_2=new_all_cities_gdf
new_all_cities_gdf_filterd_2.drop_duplicates('eFUA_name',inplace=True)
new_all_cities_gdf_filterd_2.sort_values('FUA_p_2015',ascending=False,inplace=True)


"""## try without vae [Regular Clustering ]"""

df_nondvi=pd.read_csv('/content/drive/MyDrive/0Thiland Coordination/CSV Datasets/Soc_econ_data_4paper.csv') # data
df_nondvi.set_index('eFUA_name',inplace=True)
X=df_nondvi.values
rows,cols=X.shape

X_scaled=((df_nondvi-df_nondvi.mean())/df_nondvi.std()).values
X_scaled.mean(axis=0)

from sklearn.cluster import AgglomerativeClustering,KMeans

ag=AgglomerativeClustering(n_clusters=8)
ag.fit(X_scaled)

cls_scr_sc={}
cls_scr_r={}

for n_c in range(3,10):
  ag=AgglomerativeClustering(n_clusters=n_c,linkage='ward')
  ag.fit(X_scaled)
  r=evaluate_clustering(X_scaled,ag.labels_)
  print(r)
  x,y=np.unique(ag.labels_,return_counts=True)
  print(y)
  cls_scr_sc[n_c]=r[-1]
  cls_scr_r[n_c]=r[0]



ts=TSNE(2)
X_2d=ts.fit_transform(X_scaled)
plt.scatter(X_2d[:,0],X_2d[:,1])

idx_name,name_idx=make_idx_name_dicts(df_nondvi)

# viz clusters on 2D space
ag_model,clus_names,clus,=agg_clustering(X_scaled,4)
viz_clusters(X_2d,clus_names,idx_name,name_idx)



"""# Try AEs

## Try A Shallow Autoencoder
"""

import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, Input, BatchNormalization


"""## Try VAE : Linear output layer"""

from src.VAE_model import *

df_nondvi=pd.read_csv('/content/drive/MyDrive/0Thiland Coordination/CSV Datasets/Soc_econ_data_4paper.csv')
df_nondvi.set_index('eFUA_name',inplace=True)
X=df_nondvi.values
rows,cols=X.shape

X_scaled=((df_nondvi-df_nondvi.mean())/df_nondvi.std()).values
X_scaled.mean(axis=0)

latent_dim=9


#training 

training =False # change to train 

if training:
  encoder=encoder_model(cols,latent_dim)
  decoder=decoder_model(latent_dim)
  vae=vae_model(cols,encoder,decoder)

  vae.compile(optimizer='adam',metrics=['mse'])

  hist=vae.fit(X_scaled,X_scaled,epochs=100)

else:
  #load Model
  encoder=keras.models.load_model('/content/drive/MyDrive/0Thiland Coordination/AE Models/vae_model_linear_soc-econ-v2.h5',custom_objects={'KL_loss':KL_loss,'Sample':Sample})


# try loaidng the model:
# vae=keras.models.load_model('/content/drive/MyDrive/0Thiland Coordination/AE Models/vae_model_linear_300_novi.h5',custom_objects=Sample)

X_enc,_,_=encoder(X_scaled)


# 2D Space

ts=TSNE(2)
X_2d=ts.fit_transform(X_enc)

plt.scatter(X_2d[:,0],X_2d[:,1])


cls_scr_sc={}
cls_scr_r={}
for n_c in range(3,10):
  ag=AgglomerativeClustering(n_clusters=n_c,linkage='ward')
  ag.fit(X_enc)
  r=evaluate_clustering(X_enc,ag.labels_)
  print(r)
  x,y=np.unique(ag.labels_,return_counts=True)
  print(y)
  cls_scr_sc[n_c]=r[-1]
  cls_scr_r[n_c]=r[0]


# vae.save('/content/drive/MyDrive/0Thiland Coordination/AE Models/vae_model_linear_soc-econ-v2.h5')
# encoder.save('/content/drive/MyDrive/0Thiland Coordination/AE Models/vae_model_linear_soc-econ-v2.h5')

# visualize clusters on 2D
ag_model,clus_names,clus,=agg_clustering(X_enc,6)

viz_clusters(X_2d,clus_names,idx_name,name_idx)

