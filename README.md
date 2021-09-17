# global-city-clustering

This repo is the implementation of the paper: Decoding Cities: Global Clustering Approach using Machine Learning.

This paper proposes a novel approach to classify 9000 urban areas using top-notch comparable satellite data into clusters with the objective of contributing to the development of sustainable pathways for each group. We propose a Machine Learning model throught the implementation of different components to improve the classification results. Our model is composed of 3-components: first an Outlier Detector, then a Variational Autoencoder and finally Agglomerative Clustering. Our results show that the complete model with the 3 components is more robust and yield better results in terms of creating delimeted clusters with high Calinski-Harabasz Index scores, than any other possible combination of components. 

The code is mainly to cluster city data into different groups based on featurs such as population size, area, built-up area, etc.

The jupyter notebooks provided are to test the effects of each of the components of the proposed pipeline, please look at them for more details about how to use the code and produce plots and results.

The dataset itself can not be shared currently for legal reasons.
