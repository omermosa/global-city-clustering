import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plt_ranges_of_clusters(df,clus_names):
    """
    plt ranges of variables for each cluster

    Args:
        df: dataframe of data
        clus_names: dict of cluster, city pairs
    
    returns : None
    """
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
    """
    draw sample from the clusters using cluster, city name mapping

    Args:
        df: dataframe
        clus_names: dict of cluster, city pairs
        n: number of samples
    
    returns:
        dfss: dataframe of samples
    """
    dfss=[]
    for k in clus_names:
        n_c=np.random.choice(clus_names[k],n)
        dfss.append(df.loc[n_c].iloc[:,:])
    
    return pd.concat(dfss)


def get_cluster_of(name,clus_dict):
    """
    given city, cluster number mapping, get the cluster of the given city name
    """
    for k in clus_dict:
        if name in clus_dict[k]:
            return k

