from sentence_transformers import SentenceTransformer, util
import time
import torch

#Hugging Face Model
model_path = r"..\Models\all-mpnet-base-v2"

#Dataset Path
dataset_path = r"..\Datasets\clustering_dataset.txt" 

# Model for computing sentence embeddings.
model = SentenceTransformer(model_path)

# Get all unique sentences from the file
corpus_sentences = []
with open(dataset_path, encoding="ascii") as fIn:
    corpus_sentences = set(fIn.readlines())

corpus_sentences = list(corpus_sentences)

print("Encode the corpus. This might take a while")

#use GPU if avialable and drivers are installed
device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)

corpus_embeddings = model.encode(
    corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True,device=device
)

print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(
    corpus_embeddings, min_community_size=25, threshold=0.75
)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])
