import numpy as np

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

def fft(y, dt):
    """fft along the last axis
    
    Returns:
        amplitude, phase, omega vector
    """
    
    from scipy import fftpack
    nt = y.shape[-1]
    time = dt*nt
    
    domega = 2*np.pi/time
    omegavec = domega*np.arange(0, nt) # starting from 0
    
    Fy = fftpack.fft(y)/nt*2 # Fy is complex
    
    numUniquePts = int(np.ceil( (nt+1)/2 ))
    omegavec = omegavec[..., :numUniquePts]
    Fy = Fy[..., :numUniquePts] # remove half of the last axis
    Fy[..., 0] = Fy[..., 0]*0.5 # normalization of the first point
    
    return abs(Fy), np.angle(Fy), omegavec

def fft_prod(u, v, dx):
    """fft of the product of u, v along the last axis
    
    Returns:
        (1/2)Re(u v*)
    """
    amp_u, ph_u, kvec = fft(u, dx)
    amp_v, ph_v, kvec = fft(v, dx)

    return 1/2*amp_u*amp_v*np.cos(ph_u - ph_v), kvec

def area_weighted_mean(da, lat_name=None):
    if lat_name == None:
        lat = da[da.dims[-2]]
        print('lat name:', lat.name)
    else:
        lat = da[lat_name]
        
    area_weights = np.cos(np.deg2rad(lat))
    da_mean = np.nansum(da*area_weights)/np.nansum(xr.ones_like(da)*area_weights)
    return da_mean

def get_fcor(lat):
    return 2*2*np.pi/86164*np.sin(lat/180*np.pi)

def get_beta(lat):
    return 2*2*np.pi/86164/6371e3*np.cos(lat/180*np.pi)
    
def geoaxes(axes, land=True):
    """Axes settings for geographical maps
    
    Args:
        axes: array of axes or plt.gca()
    """
    
    if not (type(axes) == np.ndarray):
        axes = [axes]
        
    if land:
        coastlines = get_land()
        
    for i in range(len(axes)):
        if land:
            coastlines.plot.contour(ax=axes[i], colors='k', linewidths=1)
        if i == len(axes) - 1: # label the last plot
            axes[i].set_xlabel('Longitude')
        else:
            axes[i].set_xlabel('')
        axes[i].set_ylabel('Latitude')
        axes[i].set_xticks(range(0, 360+1, 60))
        axes[i].set_yticks(range(-90, 90+1, 15))
        
from scipy.ndimage.filters import convolve1d

def xroll(array, fac=1, axis=-1):
    """Compute cyclic rolling mean"""
    
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

def sym(da, dim='y'):
    """Symmetrize da across y = 0"""

    da_reverse = xr.DataArray(da.values, coords=[-da[dim]], dims=[dim])
    da_out = (da + da_reverse)/2
    return da_out

def antisym(da, dim='y'):
    """Anti-symmetrize da across y = 0"""

    da_reverse = xr.DataArray(-da.values, coords=[-da[dim]], dims=[dim])
    da_out = (da + da_reverse)/2
    return da_out

def latsel(da, latS, latN):
    lat_name = da.dims[-2]
    lat = da[lat_name].values
    
    ilatS = np.searchsorted(lat, latS)
    ilatN = np.searchsorted(lat, latN)
    
    return da.isel({lat_name: range(ilatS, ilatN)})

def lonsel(da, lonW, lonE):
    lon_name = da.dims[-1]
    lon = da[lon_name].values
    
    ilonW = np.searchsorted(lon, lonW)
    ilonE = np.searchsorted(lon, lonE)
    
    return da.isel({lon_name: range(ilonW, ilonE)})

def latlon(da, latS, latN, lonW, lonE):
    return latsel(lonsel(da, lonW, lonE), latS, latN)

def get_land(da_source=None):
    """For use in TC seed tracker with HiRAM land data"""
    
    if da_source is None:
        land_frac = xr.open_dataset('HiRAM_land_static.nc')['frac'][0]
    else:
        land_frac = da_source
        
    land_bool = land_frac.values/land_frac.values
    land_bool[np.isnan(land_bool)] = 0
    return xr.DataArray(land_bool, coords=[land_frac[land_frac.dims[-2]], land_frac[land_frac.dims[-1]]], dims=['lat', 'lon'])

def downsample(da, fac=2):
    return da[..., ::fac, ::fac]

from scipy.linalg import block_diag

def coarse_grain(data4d, factor=1):
    """Coarse grain the last two dimensions of input data by a factor >= 1
    
    Args:
        factor (int)
    """
    
    ndim = len(data4d.dims)
    
    yaxis = data4d[data4d.dims[-2]]
    xaxis = data4d[data4d.dims[-1]]    
    data = data4d.values
    
    # parameters needed for coarse-graining
    xlen = data.shape[-1]
    ylen = data.shape[-2]
    xquotient = xlen//factor
    yquotient = ylen//factor
    ysouth = (ylen - yquotient*factor)//2
    ynorth = ysouth + yquotient*factor
    
    # helper matrices
    onecol = np.ones((factor, 1))/factor # a column vector
    ones = (onecol,)*xquotient
    right = block_diag(*ones)
    onerow = np.ones((1, factor))/factor # a row vector
    ones = (onerow,)*yquotient
    left = block_diag(*ones)
    
    # do the work
    xcoarse = np.dot(xaxis.values, right)
    ycoarse = np.dot(left, yaxis.values[ysouth:ynorth]).flatten()
    
    if ndim == 4:
        taxis = data4d[data4d.dims[0]]
        zaxis = data4d[data4d.dims[1]]
        coarse = np.array([[np.dot( np.dot(left, data[it,iz,ysouth:ynorth,:]), right ) for iz in range(data.shape[1])] for it in range(data.shape[0])])
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[taxis,zaxis,ycoarse,xcoarse], name=data4d.name)
    elif ndim == 3:
        taxis = data4d[data4d.dims[0]]
        coarse = np.array([np.dot( np.dot(left, data[it,ysouth:ynorth,:]), right ) for it in range(data.shape[0])])
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[taxis,ycoarse,xcoarse], name=data4d.name)
    elif ndim == 2:
        coarse = np.dot( np.dot(left, data[ysouth:ynorth,:]), right )
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[ycoarse,xcoarse], name=data4d.name)
    else:
        print("check ndim")
        
    return da

def nan2zero(da):
    nparray = da.values
    nparray[np.isnan(nparray)] = 0
    return xr.DataArray(nparray, coords=da.coords)

from scipy.interpolate import interp2d

def xrinterp(da, target):
    """interpolation to target's grid"""
    
    da = nan2zero(da) # some data have nan on land, which breaks interp2d
    npinterp = interp2d(da[da.dims[-1]], da[da.dims[-2]], da)(target[target.dims[-1]], target[target.dims[-2]])
    da2 = xr.DataArray(npinterp, coords=[target[target.dims[-2]], target[target.dims[-1]]], dims=[target.dims[-2], target.dims[-1]], name=da.name)
    return da2