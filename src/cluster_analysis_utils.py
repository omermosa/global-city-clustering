# plt Cluster ranges for each variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plt_ranges_of_clusters(df,clus_names):
  pop_ls=['X1990_Pop','X2015_Pop']
  area_ls=['X1990_area','X2015_area']

  area_r=['2000_area_r','2015_area_r']
  pop_ar=['X2015_area','X2015_Pop']
  ar_arr=['X2015_area','2015_area_r']


  plt_names={'population':pop_ls,'area':area_ls,'Area ratio':area_r,'Pop and Area': pop_ar,'area area_ratio':ar_arr}

  for name in plt_names:
    for k in clus_names:
      _idx=clus_names[k]
      dt=df.loc[_idx,plt_names[name]].values
      plt.scatter(dt[:,0],dt[:,1])
    plt.title(name)
    plt.show()


def sample_from_clus(df,clus_names,n=10):
  dfss=[]
  for k in clus_names:
    n_c=np.random.choice(clus_names[k],n)
    dfss.append(df.loc[n_c].iloc[:,:])
  
  return pd.concat(dfss)