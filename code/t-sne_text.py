import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import os

# Create an argument parser
parser = argparse.ArgumentParser(description='Visualize text embeddings using t-SNE')
parser.add_argument('--text_emb_dir', type=str, required=True, help='Directory containing the text embeddings')
parser.add_argument('--cid_emb_dir', type=str, required=True, help='Directory containing the CID embeddings')
args = parser.parse_args()

# Load the text embeddings and CIDs
text_embeddings_train = np.load(os.path.join(args.text_emb_dir, 'text_embeddings_train.npy'))
cids_train = np.load(os.path.join(args.cid_emb_dir, 'cids_train.npy'))


# Load the text embeddings and CIDs
#text_embeddings_train = np.load('/Users/sherry/Downloads/598dhl/Model_embeddings/2024_4_8_epoch40_sample100_mlp1/text_embeddings_train.npy')
#cids_train = np.load('/Users/sherry/Downloads/598dhl/Model_embeddings/2024_4_8_epoch40_sample100_mlp1/cids_train.npy')

# Select the top 20 embeddings and their corresponding CIDs
top_n = 100
selected_embeddings = text_embeddings_train[:top_n]
selected_cids = cids_train[:top_n]

# Apply t-SNE to reduce the dimensionality of the selected embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(selected_embeddings)

# Create a colormap with a unique color for each CID
unique_cids = np.unique(selected_cids)
num_cids = len(unique_cids)
cmap = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, num_cids)))

# Create a dictionary to map each CID to its corresponding color
cid_color_map = {cid: cmap(i) for i, cid in enumerate(unique_cids)}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create a scatter plot for each CID with its corresponding color
for cid in unique_cids:
    mask = selected_cids == cid
    ax.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], c=[cid_color_map[cid]], label=str(cid))
    
    # Annotate each point with its CID label
    for i in range(len(selected_cids)):
        if selected_cids[i] == cid:
            if str(cid) == '24817':
                ax.annotate(str(cid), (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                            textcoords="offset points", xytext=(5, 5), ha='right', va='bottom', fontsize=8, color='red')
            elif str(cid) == '18854':
                ax.annotate(str(cid), (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                            textcoords="offset points", xytext=(-5, -5), ha='right', va='top', fontsize=8, color='blue')
            else:
                ax.annotate(str(cid), (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                            textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)

# Set the axis labels and title
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('t-SNE Visualization of Top 100 Text Embeddings')

# Display the plot
plt.tight_layout()
plt.show()


