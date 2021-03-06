
# Parent Direcotry
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

from sklearn.cluster import  AgglomerativeClustering, KMeans

from src.clustering_utils import *



## Shape file
new_all_cities_file = os.path.join("GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0", "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
new_all_cities_gdf = gpd.read_file(new_all_cities_file)
new_all_cities_gdf_filterd_2=new_all_cities_gdf
new_all_cities_gdf_filterd_2.drop_duplicates('eFUA_name',inplace=True)
new_all_cities_gdf_filterd_2.sort_values('FUA_p_2015',ascending=False,inplace=True)


"""# Clustering"""

df_nondvi=pd.read_csv('/content/drive/MyDrive/0Thiland Coordination/CSV Datasets/Soc_econ_data_4paper.csv')
df_nondvi.set_index('eFUA_name',inplace=True)
X=df_nondvi.values
rows,cols=X.shape

X_scaled=((df_nondvi-df_nondvi.mean())/df_nondvi.std()).values
X_scaled.mean(axis=0)

# read outleier and non outlier data
all_data_no=pd.read_csv('/content/drive/MyDrive/0Thiland Coordination/CSV Datasets/all_data_noni_nooutliers_.csv')
all_data_o=pd.read_csv('/content/drive/MyDrive/0Thiland Coordination/CSV Datasets/all_data_noni_outliers_.csv')
all_data_no.set_index('eFUA_name',inplace=True)
all_data_o.set_index('eFUA_name',inplace=True)

X=all_data_no.values
rows,cols=X.shape

X_scaled=((all_data_no-all_data_no.mean())/all_data_no.std()).values
print(X_scaled.mean(axis=0))

"""## Regular AC"""
## Tru Agglomerative Clustering on the Outlier Data


print(X_scaled.std(0))



from sklearn.cluster import AgglomerativeClustering,KMeans,AffinityPropagation

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

# scores
plt.scatter(cls_scr_sc.keys(),cls_scr_sc.values())
plt.plot(list(cls_scr_sc.keys()),list(cls_scr_sc.values()))

# scores

# 2D Space
from sklearn.manifold import TSNE
ts=TSNE(2)
X_2d2=ts.fit_transform(X_scaled)
plt.scatter(X_2d2[:,0],X_2d2[:,1])

# cluster mapping to index and vice versa
idx_name, name_idx=make_idx_name_dicts(all_data_no)
ag_model,clus_names,clus,=agg_clustering(X_scaled[:,:-1],idx_name,6)
viz_clusters(X_2d2,clus_names,idx_name,name_idx)


"""## Try VAE + Linear """

## Try VAE on non outlier data
""""
### VAE Arch
"""
from src.VAE_model import *
df_all_cl=pickle.load(open('/content/drive/MyDrive/0Thiland Coordination/pickle_files/nondvi_withcity_lights.pkl','rb'))
latent_dim=9
rows,cols=X_scaled.shape

training=False # change if you wish to train the model

if training:
  encoder=encoder_model(cols,latent_dim)
  decoder=decoder_model(latent_dim,cols)
  vae=vae_model(cols,encoder,decoder)


  vae.compile(optimizer='adam',metrics=['mse'])

  hist=vae.fit(X_scaled,X_scaled,epochs=60)

## load model
else:
  encoder=keras.models.load_model('/content/drive/MyDrive/0Thiland Coordination/AE Models/encoder_nooutliers__novi_best.h5',custom_objects={'KL_loss':KL_loss,'Sample':Sample})

"""### VAE Testing and Results Exploration"""

X_enc,_,_=encoder(X_scaled[:,:-1])



# 2D Space
ts=TSNE(2)
X_2d=ts.fit_transform(X_enc)
plt.scatter(X_2d[:,0],X_2d[:,1])

# Clustering Scores
cls_scr_sc={}
cls_scr_r={}
for n_c in range(3,10):
  ag=AgglomerativeClustering(n_clusters=n_c,linkage='ward')
  ag.fit(X_enc)
  r=evaluate_clustering(X_enc,ag.labels_)
  print(f"scores {r}")
  x,y=np.unique(ag.labels_,return_counts=True)
  print(f"dist {y}")
  cls_scr_sc[n_c]=r[-1]
  cls_scr_r[n_c]=r[0]

idx_name,name_idx=make_idx_name_dicts(all_data_no)




# viz clusters on 2D space
ag_model,clus_names,clus,=agg_clustering(X_enc,idx_name,8)

viz_clusters(X_2d,clus_names,idx_name,name_idx)

# Cluster Analysis and Sampling
from src.cluster_analysis_utils import *

plt_ranges_of_clusters(all_data_no,clus_names)

# repeat without VAE
ag_model2,clus_names2,clus2,=agg_clustering(X_scaled,idx_name,6)
plt_ranges_of_clusters(all_data_no,clus_names2)

X_o=all_data_o.iloc[:,:].values
X_o=(X_o-X_o.mean(0))/X_o.std(0)

# repeat for outliers
for n_c in range(2,10):
  ag=AgglomerativeClustering(n_clusters=n_c,linkage='ward')
  ag.fit(X_o)
  print(evaluate_clustering(X_o,ag.labels_))
  x,y=np.unique(ag.labels_,return_counts=True)
  print(y)



# repeat without VAE, with outliers

idx_name_o,name_idx_o=make_idx_name_dicts(all_data_o)
ag_model_o,clus_names_o,clus_o,=agg_clustering(X_o,idx_name_o,2)
plt_ranges_of_clusters(all_data_o,clus_names_o)




# draw samples.
print(sample_from_clus(all_data_o,clus_names_o,5))

print(sample_from_clus(all_data_no,clus_names,2))

print(sample_from_clus(all_data_no,clus_names2,2))