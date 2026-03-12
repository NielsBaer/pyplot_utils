import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

def glob_cont_map(data:xr.DataArray, ax, cmap, clevels, title=None, stipple=None, stipple_dens=4, stipple_hatch='.', mask=None, mask_color='lightgray'):
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
        ax.contourf(mask_lon, mask_lat, mask_dat, levels=[0.5, 1.5], colors='lightgray', transform=ccrs.PlateCarree())


def cont_cont_map(data:xr.DataArray, ax, cmap, clevels, title=None, stipple=None, stipple_dens=4, stipple_hatch='.'):
    """plots a contourf plot on continets only.
        -------
        Inputs:
        data: data to plot. Has to have lat, lon dimensions and all other dimensions length 1 values on ocean should be Nan or zero (Nan usually gives clearer coastlines)
        ax: axis to plot on with according projection (recommend equalEarth or Eckert IV)
        cmap: colormap for the plot
        clevels: contourlevels (transition between colors)
        stipple: data on same lat lon coordinates as data to indicate where to stipple the map to e.g. indicate significance. Areas to stipple should have value 1, other 0 or Nan
        stipple_dens: density of stippling, has no effect if stipple is None
        stipple_hatch: type of hatch used for stippling, has no effect if stipple is None
    """
    map_data, lon = add_cyclic_point(data.to_numpy(), coord=data.lon)
    lat = data.lat
    ax.contourf(lon, lat, map_data, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree())
    ax.axis('off')
    ax.set_extent([-180,180,-57,90], crs=ccrs.PlateCarree())
    ax.coastlines()
    if stipple is not None:
        stip_dat, stip_lon = add_cyclic_point(stipple.to_numpy(), coord=stipple.lon)
        stip_lat = stipple.lat
        ax.contourf(stip_lon, stip_lat, stip_dat, levels=[-0.5, 0.5, 1.5], colors='none', hatches=[stipple_dens*stipple_hatch], transform=ccrs.PlateCarree())

