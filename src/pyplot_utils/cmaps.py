"""Utilities for handeling colormaps and colorbars, so far largely for countourf plots"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap as LScmap
from matplotlib.colors import ListedColormap as Lcmap
import numpy as np
import math


def div_cmap(n, cmap_neg=plt.cm.Blues, cmap_pos=plt.cm.Reds):
    """generate a diverging colormap given two colormaps for negative and positive side. Always sets 0 to white.
        Inputs:
        n: number of levels for each side (total 2n+1 levels)
        cmap_neg: colormap for the negative part
        cmap_pos: colormap for the positive part
    """
    colors_neg = cmap_neg.reversed()(np.linspace(1/n,1,n))
    colors_pos = cmap_pos(np.linspace(1/n,1,n))

    colors = np.vstack((colors_neg, np.ones((1,4)), colors_pos))
    #return LScmap.from_list('name', colors)
    return Lcmap(colors)

def get_cmap_w_levels(max_abs, n, cmap=plt.cm.magma_r, negative=False):
    """generate listed colormap for data with given maximum absolute value and levels for use in contourplots
        Inputs:
        max_abs: maximum absoulte value in the data to visualize
        n: number of levels
        cmap: colormap
        negative: Bool, if True, the levels will cover [-max_abs, 0], otherwise [0,-max_abs]
    """
    if negative: 
        colors= cmap(np.linspace(0,1,n))
        levs = np.linspace(-max_abs, 0, n)
    else:
        colors= cmap.reversed()(np.linspace(0,1,n+1))
        levs = np.linspace(0, max_abs, n+1)
    return (Lcmap(colors), levs)


def get_div_cmap_w_levels(max_abs, n, cmap_neg=plt.cm.Blues, cmap_pos=plt.cm.Reds):
    """generate a diverging listed colormap for data with given maximum absolute value. Also returns levels for use in contourplots
        Inputs:
        max_abs: maximum absoulte value in the data to visualize
        n: number of levels for each side (total 2n+1 levels)
        cmap_neg: colormap for the negative part
        cmap_pos: colormap for the positive part
    """
    levs = np.linspace(-max_abs, max_abs, 2*n+2)
    cmap = div_cmap(n, cmap_neg, cmap_pos)
    return (cmap, levs)


def plot_cbar(ax, title, cmap, levs, location='bottom', mappable=None, **kwargs):
    """plot a colorbar for contourf plot. The colorbar is independent of any actual plot, making it easy to use for multiple panels.
        Inputs:
        ax: axis for the colorbar
        title: title of the colorbar
        cmap: colormap of the colorbar
        levs: levels of the colorbar. Assumes is that these are levels from a contourf plot, meaning they are the points of transition between colors
        orientation: orientation of the colorbar. Either 'horizontal' (default) or 'vertical'
    """
    if location == 'left' or location == 'right':
        orientation = 'vertical'
    elif location == 'bottom' or location == 'top':
        orientation = 'horizontal'
    else:
        raise Exception("unknown location: "+location)
    print(mappable)
    cbar = ColorbarBase(ax, cmap=cmap, orientation=orientation, ticklocation=location, values=levs, **kwargs)
    print('title')
    print(title)
    cbar.set_label(title, fontsize=10)
    # for some reason the cbar doesnt draw its tick on its own, but this works...
    # maybe investigate this/open an issue 
   
def cbar_n_map(ax, max_abs, n, cmap_type='diverging', title='', location='bottom', cmap_neg=plt.cm.Blues, cmap_pos=plt.cm.Reds, round=None, auto_reduce_n=True, **cbar_kwargs):
    """plot a colorbar on given axis and return the corresponding colormap with levels.
        -------
        Inputs:
        ax: axis to plot colorbar onto
        max_abs: maximum absolute value
        n: levels for the colorbar (level for 0 not included). For diverging colormaps n is the number of levels for each side (2n+1 in total).
        type: type of the values. Options are 'diverging' (default), 'positive', 'negative'
        title: title of the colorbar
        cmap_neg: colormap for the negative part (ignored if type is 'positive')
        cmap_pos: colormap for the positive part (ignored if type is 'negative')
        round: float or None. If not None, ensures that all steps are multiples of this value
        auto_reduce_n: only has an effect if round is not none. If True, reduces n such that levels exceeding max_abs do not exist.
        -------
        Returns:
        cmap: colormap plotted on the colorbar to use in plots
        clevels: colorlevels to use in plots. These are values for the color transitions to use directly as levels in e.g. contourf plots
    """
    if round is not None:
        if cmap_type == 'diverging':
            # maximum value has to be integer multiple of 2n+1 and round (then the half-interval step around zero is also multiple of round)
            fac = (2*n+1)*round
        else:
            fac = n*round
        max_abs_new = math.ceil(max_abs/fac) * fac
        # adjust n and max_abs, if auto_reduce_n
        if auto_reduce_n:
            step = max_abs_new/(n+0.5)
            n_new = math.ceil((max_abs-0.5*step)/step)
            max_abs_new = (n_new + 0.5) * step
            n = n_new
        max_abs = max_abs_new

    if cmap_type == 'diverging':
        cmap, clevels = get_div_cmap_w_levels(max_abs, n, cmap_neg, cmap_pos)
    elif cmap_type == 'positive':
        cmap, clevels = get_cmap_w_levels(max_abs, n, cmap_pos)
    elif cmap_type == 'negative':
        cmap, clevels = get_cmap_w_levels(max_abs, n, cmap_neg, negative=True)
    else:
        raise Exception("unknown cmap_type: "+cmap_type)
    cbar_levs = (clevels[:-1] + clevels[1:]) / 2
    norm = mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)
    mappable = mpl.cm.ScalarMappable(norm, cmap)
    plot_cbar(ax, title, cmap, cbar_levs, location, norm=norm, **cbar_kwargs)
    return (cmap, clevels)


