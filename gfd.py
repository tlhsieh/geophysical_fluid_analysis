import numpy as np

def get_fcor(lat):
    return 2*2*np.pi/86164*np.sin(lat/180*np.pi)

def get_beta(lat):
    return 2*2*np.pi/86164/6371e3*np.cos(lat/180*np.pi)
    
def geoaxes(axes):
    '''
    Axes settings for geographical maps
    
    Args:
        axes: array of axes or plt.gca()
    '''
    
    if not (type(axes) == np.ndarray):
        axes = [axes]
        
    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks(range(0, 361, 60))
        
from scipy.ndimage.filters import convolve1d

def xroll(array, fac=1, axis=-1):
    '''compute cyclic rolling mean'''
    
    kernal = np.ones((fac))/fac
    rolled = convolve1d(array, kernal, axis=axis, mode='wrap')
    return array - array + rolled # recover coords of the dataarray

def yroll(array, fac=1, axis=-2):
    kernal = np.ones((fac))/fac
    rolled = convolve1d(array, kernal, axis=axis, mode='nearest')
    return array - array + rolled # recover coords of the dataarray

def smooth(da, xfac=1, yfac=1):
    return yroll(xroll(da, xfac), yfac)

import xarray as xr

def ddx(da, dim='x'):
    upper = da.diff(dim, label='upper')/da[dim].diff(dim, label='upper')
    lower = da.diff(dim, label='lower')/da[dim].diff(dim, label='lower')
    return (upper + lower)*0.5

def nan2zero(da):
    nparray = da.values
    nparray[np.isnan(nparray)] = 0
    return xr.DataArray(nparray, coords=da.coords)

def get_land():
    '''return land data for HiRAM'''
    
    land_frac = xr.open_dataset('/home/hsiehtl/HiRAM_land_static.nc')['frac'][0]
    land_bool = land_frac.values/land_frac.values
    land_bool[np.isnan(land_bool)] = 0
    return xr.DataArray(land_bool, coords=[land_frac.grid_yt, land_frac.grid_xt], dims=['lat', 'lon'])