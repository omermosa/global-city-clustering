
"""## Utils

"""
from sklearn.cluster import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram


def make_idx_name_dicts(df):
    """
    get the city name to index and index to city name dicts for the data

    Args:
        df: dataframe
    
    returns:
        idx_name: index: name dict
        name_idx: name: index dict
    """
    idx_name={i:name for i,name in enumerate(df.index)}
    name_idx={name:i for i,name in enumerate(df.index)}
    return idx_name, name_idx


def cluster_with_kmeans(df,idx_name,n=5):
    """
    cluster using kmeans algo
    Args:
        df: data
        idx_name: index,name dict
        n: num of clusters
    
    Returns:
        cluster_dict_names: city_names to cluster number dict
        l: labels (list of cluster numbers for the data)
    
    """
    kmeans = KMeans(n_clusters=n,max_iter=10000,n_init=20).fit(df)

    clusters_dict={}
    l=kmeans.labels_
    for i,c in enumerate(l):
        if c in clusters_dict:
            clusters_dict[c].append(i)
        else:
            clusters_dict[c]=[i,]

    clusters_dict_names={}
    for k in clusters_dict:
        clusters_dict_names[k]=[]
        for city_idx in clusters_dict[k]:
            clusters_dict_names[k].append(idx_name[city_idx])
    
    return clusters_dict_names,l


def agg_clustering(df,idx_name,n=5,link='ward'):

    """
    Cluster using Agglomerative Clustering

    Args:
        df: data
        idx_name:index,city_name dict
        n= num of clusters
        link: linkage of the AgglomerativeClustering
    
    Returns:
        AC: agg_clustering object
        clusters_dict_names: city_names to cluster number dict
        l: labels (list of cluster numbers for the data)


    """
    AC=AgglomerativeClustering(n_clusters=n,linkage=link).fit(df)
    clusters_dict={}
    l=AC.labels_
    for i,c in enumerate(l):
        if c in clusters_dict:
            clusters_dict[c].append(i)
        else:
            clusters_dict[c]=[i,]

    clusters_dict_names={}
    for k in clusters_dict:
        clusters_dict_names[k]=[]
        for city_idx in clusters_dict[k]:
            clusters_dict_names[k].append(idx_name[city_idx])
    
    return AC,clusters_dict_names,l


def viz_clusters(df,clusters_dict_names,idx_name,name_idx,inc_names=False):

    """
    visualize the clusters on the 2D space

    Args:
        df: 2D space given by the TSNE

        clusters_dict_names: city_names to cluster number dict
        name_idx: name: index dict
        idx_name: index:city_name dict
        inc_names: boolean to decide whether to put city names on the plot
    
    Returns:
        None

    """
    plt.figure(figsize=(10,10))  
    for c in clusters_dict_names.keys():
        cities=clusters_dict_names[c]
        inds=[]
        for city in cities:
            inds.append(name_idx[city])
        

        
        pts=df[inds]
        plt.scatter(pts[:,0],pts[:,1])
        _avg=np.mean(pts,axis=0)
        plt.annotate(c,(_avg[0],_avg[1]))


    v=[clusters_dict_names[k][0] for k in clusters_dict_names.keys()]  
    plt.legend(v)  
    plt.show()

    if inc_names:
        plt.figure(figsize=(30,20))
        for c in clusters_dict_names.keys():
            cities=clusters_dict_names[c]
        inds=[]
        for city in cities:
            inds.append(name_idx[city])
        

        
        pts=df[inds]
        plt.scatter(pts[:,0],pts[:,1])
        for ind in inds:
            plt.text(df[ind,0],df[ind,1],idx_name[ind],fontsize=15)
        
        plt.show()

    

def plot_dendrogram(model, **kwargs):
    """
    plot dendrogram of a given Agg clustering Model
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def evaluate_clustering(df,labels):
    """
    get SC (silhouette_score) and CHI (calinski_harabasz_score) for the clustering

    Args:
        df:data
        labels: list of cluster number for each  data point
    
    returns: 
        SC, CHI scores
    """
    return silhouette_score(df, labels, metric='euclidean'),calinski_harabasz_score(df, labels)

def get_clus_dist(dict_clus):
    """
    plot the number of data point per cluster   (cluster Distribution)
    
    Args:
        dict_clust: cluster  to city_names  dict

    Returns:
        dictionary of cluster_number: count
    """
    c_num={}
    n=len(dict_clus)
    for k in dict_clus:
        c_num[k]=len(dict_clus[k])
    
    plt.bar(list(range(n)),c_num)
    return c_num

