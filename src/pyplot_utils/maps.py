import matplotlib.pyplot as plt
import matplotlib.axes as mplax
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
from .cmaps import cbar_n_map
from cartopy.util import add_cyclic_point

def glob_cont_map(data:xr.DataArray, ax, cmap, clevels, stipple=None, stipple_dens=4, stipple_hatch='.', mask=None, mask_color='lightgray'):
    """plots a global contourf map.
        -------
        Inputs:
        data: data to plot. Has to have lat, lon dimensions and all other dimensions length 1
        ax: axis to plot on with according projection
        cmap: colormap for the plot
        clevels: contourlevels (transition between colors)
        stipple: data to indicate where to stipple the map to e.g. indicate significance. Areas to stipple should have value 1, other 0 or Nan
        stipple_dens: density of stippling, has no effect if stipple is None
        stipple_hatch: type of hatch used for stippling, has no effect if stipple is None
        mask: data indicating where to mask the map. Areas to mask should have value 1, other 0 or Nan
        mask_color: color to use for masking, has no effect if mask is None
    """
    map_data, lon = add_cyclic_point(data.to_numpy(), coord=data.lon)
    lat = data.lat
    ax.contourf(lon, lat, map_data, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    # mask and stipple if given arguments
    if stipple is not None:
        stip_dat, stip_lon = add_cyclic_point(stipple.to_numpy(), coord=stipple.lon)
        stip_lat = stipple.lat
        ax.contourf(stip_lon, stip_lat, stip_dat, levels=[0.5, 1.5], colors='none', hatches=[stipple_dens*stipple_hatch], transform=ccrs.PlateCarree())
    if mask is not None:
        mask_dat, mask_lon = add_cyclic_point(mask.to_numpy(), coord=mask.lon)
        mask_lat = mask.lat
        ax.contourf(mask_lon, mask_lat, mask_dat, levels=[0.5, 1.5], colors=mask_color, transform=ccrs.PlateCarree())


def cont_cont_map(data:xr.DataArray, ax, cmap, clevels, stipple=None, stipple_dens=4, stipple_hatch='.', mask=None, mask_color='lightgray', reduce_lon=True):
    """plots a contourf plot on continets only.
        -------
        Inputs:
        data: data to plot. Has to have lat, lon dimensions and all other dimensions length 1 values on ocean should be Nan or zero (Nan usually gives clearer coastlines)
        ax: axis to plot on with according projection (recommend equalEarth)
        cmap: colormap for the plot
        clevels: contourlevels (transition between colors)
        stipple: data on same lat lon coordinates as data to indicate where to stipple the map to e.g. indicate significance. Areas to stipple should have value 1, other 0 or Nan
        stipple_dens: density of stippling, has no effect if stipple is None
        stipple_hatch: type of hatch used for stippling, has no effect if stipple is None
        reduce_lon: if True reduces the extend of the axis to save space. This ommits Hawaii, Fiji, and Pacific Islands between, but saves a lot of space. Ony tested with EqualEarth projection.
    """
    map_data, lon = add_cyclic_point(data.to_numpy(), coord=data.lon)
    lat = data.lat
    ax.contourf(lon, lat, map_data, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree())
    ax.axis('off')
    ax.set_global()
    if reduce_lon:
        ax.set_extent([-140,170,-57,90], crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-179,179,-57,90], crs=ccrs.PlateCarree())
    ax.coastlines()
    if stipple is not None:
        stip_dat, stip_lon = add_cyclic_point(stipple.to_numpy(), coord=stipple.lon)
        stip_lat = stipple.lat
        ax.contourf(stip_lon, stip_lat, stip_dat, levels=[-0.5, 0.5, 1.5], colors='none', hatches=[stipple_dens*stipple_hatch], transform=ccrs.PlateCarree())
    if mask is not None:
        mask_dat, mask_lon = add_cyclic_point(mask.to_numpy(), coord=mask.lon)
        mask_lat = mask.lat
        ax.contourf(mask_lon, mask_lat, mask_dat, levels=[0.5, 1.5], colors=mask_color, transform=ccrs.PlateCarree())


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


def map_n_cbar(data:xr.DataArray | list[xr.DataArray] , ax:mplax.Axes | list[mplax.Axes], cbar_ax: mplax.Axes, n:int, cont_only:bool=False, cbar_kwargs:dict={}, map_kwargs:dict={}, stipple:None|xr.DataArray|list[xr.DataArray|None]=None, mask:None|xr.DataArray|list[xr.DataArray|None]=None):
    """plot a colorbar and maps for multiple datasources showing the same variable.
        Inputs:
        data: data to plot. If this is a list, ax has to be a list of same length, and the maps are plotted on the respective axes in ax.
        ax: axis/axes to plot the data on. If data contains multiple dataarrays, ax has to be a list with as many axes.
        cbar_ax: axis to plot the colorbar on
        n: number of levels for contourf plot. Note, that this turns into an upper limit if 'round' is passed in cbar_kwargs, and 'auto_reduce_n' is left True.
        cont_only: if True, maps are plotted with cont_cont_map, otherwise glob_cont_map. Use True for data only over land.
        cbar_kwargs: keyword arguments to be passed to cmaps.cbar_n_map. Can be used to e.g. change the colormap generated.
        map_kwargs: keyword arguments to be passed to the map plotting function used. Can e.g. be used to change stipple density.
        stipple: dataarray(s) showing where to stipple (1 for stipple, 0 or Nan for rest). If this is a list, it has to have the same length as data. If it is a single dataarray it is applied to all maps plotted.
        mask: dataarray(s) showing where to mask maps (1 for stipple, 0 or Nan for rest). If this is a list, it has to have the same length as data. If it is a single dataarray it is applied to all maps plotted.
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
    mabs = np.float64(0)
    for darray in data:
        da_mabs = np.nanmax(np.abs(darray))
        mabs = np.nanmax([mabs, da_mabs])
    cmap, clevels = cbar_n_map(ax=cbar_ax, max_abs=mabs, n=n, **cbar_kwargs)
    for darray, stip_da, mask_da, axis in zip(data, stip_list, mask_list, ax):
        if cont_only:
            cont_cont_map(data=darray, ax=axis, cmap=cmap, clevels=clevels, stipple=stip_da, **map_kwargs)
        else:
            glob_cont_map(data=darray, ax=axis, cmap=cmap, clevels=clevels, stipple=stip_da, **map_kwargs)


