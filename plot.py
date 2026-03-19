import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('average_waveforms_filtered.csv', header=None)

# Number of waveforms
num_waveforms = len(df)

# Subplots per figure: 10x10 = 100
subplots_per_figure = 100
nrows = 10
ncols = 10

# Number of figures needed
num_figures = int(np.ceil(num_waveforms / subplots_per_figure))

# Plot in multiple figures
for fig_num in range(num_figures):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    start_idx = fig_num * subplots_per_figure
    end_idx = min(start_idx + subplots_per_figure, num_waveforms)
    
    for i in range(start_idx, end_idx):
        waveform = df.iloc[i].values
        ax_idx = i - start_idx
        axes[ax_idx].plot(waveform)
        axes[ax_idx].axis('off')  # Turn off axes for cleaner view
    
    # Hide any unused subplots in this figure
    for i in range(end_idx - start_idx, subplots_per_figure):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Waveforms {start_idx+1} to {end_idx}')
    plt.tight_layout()
    plt.show()

plt.tight_layout()
plt.show()
