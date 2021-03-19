import numpy as np

def get_fcor(lat):
    return 2*2*np.pi/86164*np.sin(lat/180*np.pi)

def get_beta(lat):
    return 2*2*np.pi/86164/6371e3*np.cos(lat/180*np.pi)
    
def nearest_mean(x_in, y_in, z_in, x_out, y_out, n_nearest=6, decay=4):
    """Given z(x, y), find z(x', y'), in which x' and y' are not on the x, y grid
    
    Example:
        slp_track = nearest_mean(slp.lon, slp.lat, slp.values, lon_track, lat_track)

    Args:
        x_in (1d or 2d array)
        y_in (1d or 2d array)
        z_in (2d array)
        x_out (1d array)
        y_out (1d array)
        n_nearest (int): number of nearest input grid points to average over; the smaller the faster
        decay (int): exponent over 1/dist^2; the larger the more local

    Returns:
        z_out (1d array)
    """
    
    assert len(x_out) == len(y_out)
    
    small = 1e-9
    
    xx, yy = np.meshgrid(x_in, y_in)
    weights = [1/((xx - x_out[i])**2 + (yy - y_out[i])**2 + small)
               for i in range(len(x_out))] # weights proportional to 1/dist^2
    
    idx_large = [np.argsort(weight.flatten())[-n_nearest:] for weight in weights] # indices of the largest weights
    weights_large = [weights[i].flatten()[idx_large[i]] for i in range(len(idx_large))]
    
    z_out = [np.average(z_in.flatten()[idx_large[i]], weights=weights_large[i]**decay)
             for i in range(len(idx_large))]
    
    return np.array(z_out)

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
