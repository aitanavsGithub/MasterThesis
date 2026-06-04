import numpy as np
import os
import sklearn.preprocessing
from umap import umap_ as umap
import wavemap_paper
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib import pyplot as plt
import random
from wavemap_paper.helper_functions import RAND_STATE, set_rand_state, plot_inverse_mapping
import networkx as nx
import cylouvain
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mplcursors

# Load the data, optionally equalise number of waveforms, stack
def load_and_stack_waveforms(pathlist,labellist = None, equalise=True, stack = True):

    """
    Load waveforms from a list of paths, optionally equalise the number of waveforms across datasets and stack them.
    Parameters:
    - pathlist: list of paths to the waveform datasets (csv files)
    - labellist: list of labels for each dataset
    - equalise: whether to equalise the number of waveforms across datasets
    - stack: whether to stack the datasets  
    
    Returns:
    - all_waveforms: list of waveform arrays (if stack=False) or a single array of all waveforms (if stack=True)
    - dataset_labels: list of labels for each waveform (if stack=True) or None (if stack=False)
    """

    # Load the data
    waveforms = []
    lengths = []

    for path in pathlist:
        wf = np.loadtxt(path, delimiter=',', dtype=float)
        waveforms.append(wf)
        lengths.append(wf.shape[0])

    # If given, equalise the number of waveforms across datasets
    if equalise:
        n_min = min(lengths)

        indices = [random.sample(range(length), n_min) for length in lengths]
        waveforms = [wf[indices[i], :] for i, wf in enumerate(waveforms)]

    # If given, stack the datasets
    if stack:
        all_waveforms = np.vstack(waveforms)
        dataset_labels = [label
            for label, wf in zip(labellist, waveforms)
            for _ in range(len(wf))]
    else:
        all_waveforms = waveforms
        dataset_labels = None

    return all_waveforms, dataset_labels

# Run PCA
def run_pca(waveforms, n_exp_var = 0.9, plot=False):
    """
    
    Run PCA on the given waveforms and optionally plot the explained variance and the first two principal components.
    Parameters:
    - waveforms: array of waveform data
    - n_exp_var: threshold for cumulative explained variance to determine the number of components to retain
    - plot: whether to plot the results
    Returns:
    - X_pca: array of principal components
    - pca: fitted PCA object
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(waveforms)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_   # Explained variance ratio
    cumulative_variance = np.cumsum(explained_variance)   # Cumulative explained variance

    # Determine the number of components to retain based on the explained variance threshold
    n_components = np.argmax(cumulative_variance >= n_exp_var) + 1

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        # Elbow plot
        axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,marker='o')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Elbow Plot')
        axes[0].set_xlim((0, 15))
        axes[0].grid(True)

        # PC1 vs PC2
        axes[1].scatter(X_pca[:, 0],X_pca[:, 1],s=10,alpha=0.7)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('PC1 vs PC2')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    return X_pca, pca, n_components

# Run UMAP
def run_umap(X, labels, n_nb=15, min_d=0.1, n_comps=2, metric='euclidean', random_state=RAND_STATE):
    """
    Run UMAP on the given data.
    Parameters:
    - X: array of data to be embedded
    - labels: list of labels for each data point
    - n_nb: number of neighbors for UMAP
    - min_d: minimum distance for UMAP
    - n_comps: number of dimensions for UMAP embedding
    - metric: distance metric for UMAP
    - random_state: random state for reproducibility
    Returns:
    - umap_df: DataFrame containing UMAP embeddings and associated metadata
    """

    reducer = umap.UMAP(random_state=random_state, n_neighbors=n_nb, n_components=n_comps, min_dist=min_d, metric=metric)
    mapper = reducer.fit(X)
    G = nx.from_scipy_sparse_array(mapper.graph_)
    clustering = cylouvain.best_partition(G,resolution=3)
    clustering_solution = list(clustering.values())

    embedding = reducer.fit_transform(X)

    # automatic column names
    col_names = [f'umap_{i+1}' for i in range(n_comps)]

    umap_df = pd.DataFrame(embedding,columns=col_names)
    umap_df['waveform'] = list(X)
    umap_df['cluster_id'] = clustering_solution
    umap_df['dataset'] = labels

    # colours
    cmap = plt.get_cmap("turbo")
    colors = cmap(np.linspace(0,1,len(set(clustering_solution))))
    umap_df['cluster_color'] = [colors[i]for i in clustering_solution]
    return umap_df, clustering_solution


# Plot UMAP embedding with an interactive plot!
def interactive_umap_plot(umap_df, cmap_name="tab10"):
    """
    Create an interactive UMAP plot with dynamic dataset labels.

    Parameters
    ----------
    umap_df : pandas.DataFrame
        Must contain:
        - 'umap_1'
        - 'umap_2'
        - 'dataset'
        - 'waveform'

    cmap_name : str
        Matplotlib colormap name
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -----------------------------------
    # Dynamically create label mappings
    # -----------------------------------
    unique_labels = sorted(umap_df['dataset'].unique())

    dataset_map = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    cmap = plt.get_cmap(cmap_name)

    # normalize colors across labels
    n_labels = max(len(unique_labels) - 1, 1)

    dataset_colors = {
        label: cmap(idx / n_labels)
        for label, idx in dataset_map.items()
    }

    # numeric values for scatter coloring
    color_values = umap_df['dataset'].map(dataset_map)

    # -----------------------------------
    # 1) UMAP scatter
    # -----------------------------------
    sc = axes[0].scatter(
        umap_df['umap_1'],
        umap_df['umap_2'],
        c=color_values,
        cmap=cmap_name,
        s=20,
        alpha=0.5,
        picker=True
    )

    # dynamic legend
    handles = [
        plt.Line2D(
            [],
            [],
            marker='o',
            linestyle='',
            color=dataset_colors[label],
            label=label,
            markersize=8
        )
        for label in unique_labels
    ]

    axes[0].legend(handles=handles)
    axes[0].set_title("UMAP")

    # -----------------------------------
    # 2) Background waveforms
    # -----------------------------------
    for i, wf in enumerate(umap_df['waveform']):

        label = umap_df['dataset'].iloc[i]

        axes[1].plot(
            wf,
            color=dataset_colors[label],
            alpha=0.05,
            linewidth=1
        )

    axes[1].set_title("Waveforms")

    # -----------------------------------
    # Highlighted waveform
    # -----------------------------------
    highlight_line, = axes[1].plot(
        [],
        [],
        linewidth=3
    )

    # -----------------------------------
    # Interactive click behaviour
    # -----------------------------------
    cursor = mplcursors.cursor([sc], hover=False)

    @cursor.connect("add")
    def on_click(sel):

        idx = sel.index
        wf = umap_df['waveform'].iloc[idx]
        label = umap_df['dataset'].iloc[idx]

        # update waveform
        highlight_line.set_data(
            np.arange(len(wf)),
            wf
        )

        # matching color
        highlight_line.set_color(
            dataset_colors[label]
        )

        # optional annotation text
        sel.annotation.set_text(
            f"{label}\nIndex: {idx}"
        )

        # rescale waveform axis
        axes[1].relim()
        axes[1].autoscale_view()

        fig.canvas.draw_idle()

    plt.tight_layout()
    plt.show()
