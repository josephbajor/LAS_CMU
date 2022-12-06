import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention(attention):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap="GnBu")
    plt.show()
