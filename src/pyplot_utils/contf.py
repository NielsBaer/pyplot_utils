import matplotlib.pyplot as plt
import matplotlib.axes as mplax
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
from .cmaps import cbar_n_map
from cartopy.util import add_cyclic_point

def plot_contf(data:xr.DataArray, x, y, ax, cmap, clevels, stipple=None, stipple_dens=4, stipple_hatch='.', mask=None, mask_color='lightgray'):
    """plots a contourf plot on ax.
        -------
        Inputs:
        data: data to plot. Has to have x and y dimensions and all other dimensions length 1
        x: dimension of data to plot on the x axis
        y: dimension of data to plot on the y axis
        ax: axis to plot on
        cmap: colormap for the plot
        clevels: contourlevels (transition between colors)
        stipple: data to indicate where to stipple the plot to e.g. indicate significance. Areas to stipple should have value 1, other 0 or Nan and dimensions x and y as specified in arguments
        stipple_dens: density of stippling, has no effect if stipple is None
        stipple_hatch: type of hatch used for stippling, has no effect if stipple is None
        mask: data indicating where to mask the plot. Areas to mask should have value 1, other 0 or Nan. Has to have dimensions x and y.
        mask_color: color to use for masking, has no effect if mask is None
    """
    #map_data, lon = add_cyclic_point(data.to_numpy(), coord=data.lon)
    plot_data = data.to_numpy()
    plot_x = data[x]
    plot_y = data[y]
    ax.contourf(plot_x, plot_y, plot_data, levels=clevels, cmap=cmap)
    # mask and stipple if given arguments
    if stipple is not None:
        stip_data = stipple.to_numpy()
        stip_x = stipple[x]
        stip_y = stipple[y]
        ax.contourf(stip_x, stip_y, stip_data, levels=[0.5, 1.5], colors='none', hatches=[stipple_dens*stipple_hatch])
    if mask is not None:
        mask_data = mask.to_numpy()
        mask_x = mask[x]
        mask_y = mask[y]
        ax.contourf(mask_x, mask_y, mask_data, levels=[0.5, 1.5], colors=mask_color)


def zonal_contf(data:xr.DataArray, ax, cmap, clevels, y=None, stipple=None, stipple_dens=4, stipple_hatch='.', mask=None, mask_color='lightgray', reduce_lat=False):
    """plots a contourf plot on continets only.
        -------
        Inputs:
        data: data to plot. Has to have lat, y (if none given checks other dim of length greater 1 in data) dimensions and all other dimensions length 1 values on ocean should be Nan or zero (Nan usually gives clearer coastlines)
        ax: axis to plot on with according projection (recommend equalEarth)
        cmap: colormap for the plot
        clevels: contourlevels (transition between colors)
        stipple: data on same lat lon coordinates as data to indicate where to stipple the map to e.g. indicate significance. Areas to stipple should have value 1, other 0 or Nan
        stipple_dens: density of stippling, has no effect if stipple is None
        stipple_hatch: type of hatch used for stippling, has no effect if stipple is None
    """
    # find y
    if y is None:
        for dim in data.sizes.keys():
            if (dim != 'lat') and (data.sizes[dim] > 1):
                y = dim
                if dim == 'plev':
                    ax.yaxis.set_inverted(True)
                    ax.set_yscale("log")
                break

    plot_data = data.to_numpy()
    plot_lat = data.lat
    plot_y = data[y]
    ax.contourf(plot_lat, plot_y, plot_data, levels=clevels, cmap=cmap)

    if reduce_lat:
        ax.set_xlim([-50, 90])

    if stipple is not None:
        stip_data = stipple.to_numpy()
        stip_lat = stipple.lat
        stip_y = stipple[y]
        ax.contourf(stip_lat, stip_y, stip_data, levels=[0.5, 1.5], colors='none', hatches=[stipple_dens*stipple_hatch])
    if mask is not None:
        mask_data = mask.to_numpy()
        mask_lat = mask.lat
        mask_y = mask[y]
        ax.contourf(mask_lat, mask_y, mask_data, levels=[0.5, 1.5], colors=mask_color)


def get_lat_transform(projection=ccrs.EqualEarth()):
    """get a tuple of functions to pass to set_yscale of mpl axes to transform latitudes matching with given projection to use for zonal plots"""
    def to_map(lat):
        """transform latitude to y-vals"""
        dummy_lon = np.zeros_like(lat)
        proj = projection.transform_points(ccrs.PlateCarree(), dummy_lon, lat)
        return(np.take(proj, 1, axis=-1))
    def from_map(y):
        """transform y-vals to latitudes"""
        dummy_x = np.zeros_like(y)
        proj = ccrs.PlateCarree().transform_points(projection, dummy_x, y)
        return(np.take(proj, 1, axis=-1))
    return(to_map, from_map)


def contf_n_cbar(data:xr.DataArray | list[xr.DataArray] , ax:mplax.Axes | list[mplax.Axes], cbar_ax: mplax.Axes, n:int, mabs:float|None=None cont_only:bool=False, cbar_kwargs:dict={}, contf_kwargs:dict={}, stipple:None|xr.DataArray|list[xr.DataArray|None]=None, mask:None|xr.DataArray|list[xr.DataArray|None]=None, type='contf'):
    """plot a colorbar and contourf plots for multiple datasources showing the same variable.
        Inputs:
        data: data to plot. If this is a list, ax has to be a list of same length, and the contfs are plotted on the respective axes in ax.
        ax: axis/axes to plot the data on. If data contains multiple dataarrays, ax has to be a list with as many axes.
        cbar_ax: axis to plot the colorbar on
        n: number of levels for contourf plot. Note, that this turns into an upper limit if 'round' is passed in cbar_kwargs, and 'auto_reduce_n' is left True.
        mabs: maximum absolute value to plot
        cbar_kwargs: keyword arguments to be passed to cmaps.cbar_n_map. Can be used to e.g. change the colormap generated.
        contf_kwargs: keyword arguments to be passed to the contf plotting function used. Can e.g. be used to change stipple density.
        stipple: dataarray(s) showing where to stipple (1 for stipple, 0 or Nan for rest). If this is a list, it has to have the same length as data. If it is a single dataarray it is applied to all plots.
        mask: dataarray(s) showing where to mask (1 for stipple, 0 or Nan for rest). If this is a list, it has to have the same length as data. If it is a single dataarray it is applied to all plots.
        type: string indicating which plotting function to use, 'contf' (default) uses plot_contf for general contourf plots. 'zonal' uses zonal_contf
    """

    # if given single dataarray, put it in a list so that code below works universal
    if isinstance(data, xr.DataArray):
        data = [data]
    num_da = len(data)
    # check number of axes
    if isinstance(ax, mplax.Axes):
        ax = [ax]
    if len(ax) != num_da:
        raise Exception("number of axes given does not equal number of dataarrays!")
    # check stipple and convert to corresponding list
    if isinstance(stipple, xr.DataArray):
        stip_list = [stipple] * num_da
    elif stipple is None:
        stip_list = [None] * num_da
    else: 
        stip_list = stipple
    if len(stip_list) != num_da:
        raise Exception("number of stipples given does not equal number of dataarrays!")
    # same for mask
    if isinstance(mask, xr.DataArray):
        mask_list = [mask] * num_da
    elif mask is None:
        mask_list = [None] * num_da
    else: 
        mask_list = mask
    if len(mask_list) != num_da:
        raise Exception("number of masks given does not equal number of dataarrays!")
    # find maximum absolute value across datasets
    if mabs is None:
        mabs = np.float64(0)
        for darray in data:
            da_mabs = np.nanmax(np.abs(darray))
            mabs = np.nanmax([mabs, da_mabs])
    cmap, clevels = cbar_n_map(ax=cbar_ax, max_abs=mabs, n=n, **cbar_kwargs)
    for darray, stip_da, mask_da, axis in zip(data, stip_list, mask_list, ax):
        if type == 'contf':
            plot_contf(data=darray, ax=axis, cmap=cmap, clevels=clevels, stipple=stip_da, **contf_kwargs)
        elif type == 'zonal':
            zonal_contf(data=darray, ax=axis, cmap=cmap, clevels=clevels, stipple=stip_da, **contf_kwargs)
        else:
            raise Error("unkown argument for type: "+type)


