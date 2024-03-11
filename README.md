# Clustering – Faster! - Using GenAI
Clustering is the task of dividing the unlabeled data or data points into different clusters such that similar data points fall in the same cluster than those which differ from the others. The aim of the clustering process is to segregate groups with similar traits and assign them into clusters.

Sentence-Transformers can be used in different ways to perform clustering of small or large set of sentences.

SBERT Fast Clustering algorithm could be used for clustering large datasets (50k sentences in less than 5 seconds). In a large list of sentences it searches for local communities (A local community is a set of highly similar sentences).

The threshold of cosine-similarity could be configured for which we consider two sentences as similar. Also, we can specify the minimal size for a local community. This allows us to get either large coarse-grained clusters or small fine-grained clusters.

Refer: https://threadwaiting.com/clustering-faster/
