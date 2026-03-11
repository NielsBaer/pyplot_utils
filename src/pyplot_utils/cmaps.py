import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LScmap
from matplotlib.colors import ListedColormap as Lcmap
import numpy as np


def div_cmap(n, cmap_neg=plt.cm.Blues_r, cmap_pos=plt.cm.Reds):
    """generate a diverging colormap given two colormaps for negative and positive side. Always sets 0 to white.
        Inputs:
        n: number of levels for each side (total 2n+1 levels)
        cmap_neg: colormap for the negative part
        cmap_pos: colormap for the positive part
    """
    colors_neg = cmap_neg(np.linspace(0,1,n))
    colors_pos = cmap_pos(np.linspace(0,1,n))

    colors = np.vstack((colors_neg, np.ones((1,4)), colors_pos))#, np.zeros((1,4))))
    return Lcmap(colors)



