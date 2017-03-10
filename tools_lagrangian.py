import numpy as np
import pandas as pd
import xarray as xr
import bcolz
from floater.generators import FloatSet
from helper import bcolz_to_array, dataseries_to_array, region_to_json
from floater import hexgrid

# Constants R = radius of the earth
R = 6370.0e3

def latlon_2_xy(lon_deg,lat_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    y = R*lat
    x = R*np.cos(lat)*lon
    return x,y


def calc_flow_map(bcolz_dir, floatset):
    bc = bcolz.open(rootdir=bcolz_dir)
    df = bc.todataframe()
    times = df.time.unique()
    fs = floatset
    X1, Y1 = bcolz_to_array(times.max(), bc, fs, colname=['x', 'y'], drop_duplicates=True)
    X0, Y0 = bcolz_to_array(times.min(), bc, fs, colname=['x', 'y'], drop_duplicates=True)
    return (X0, Y0, X1, Y1)


def calc_l1_l2(bcolz_dir, floatset, grid = 'hex', calc_T_D=False):
    bc = bcolz.open(rootdir=bcolz_dir)
    df = bc.todataframe()
    times = df.time.unique()
    fs = floatset
    x1, y1 = bcolz_to_array(times.max(), bc, fs, colname=['x', 'y'], drop_duplicates=True)
    x0, y0 = bcolz_to_array(times.min(), bc, fs, colname=['x', 'y'], drop_duplicates=True)
    
    lav_index_npart = df.where(df.vort>-999).groupby('npart')['vort'].mean()
    lav = dataseries_to_array(lav_index_npart, fs)

    delta = 0.1
    xmax, xmin = df.x.max()-delta, df.x.min()+delta
    ymax, ymin = df.y.max()-delta, df.y.min()+delta
    mask = (df.x > xmax) | (df.x < xmin) | (df.y > ymax) | (df.y < ymin)

    avg_count_series = df[~mask].where(df.vort>-999).groupby('npart')['vort'].count()
    cnt = dataseries_to_array(avg_count_series, fs)
    lav_mask = (cnt<cnt.max())

    ha = hexgrid.HexArray(np.abs(lav).astype('f8'))
    neighbors = np.zeros((ha.N,6), dtype='i4')
    for n in xrange(ha.N):
        nbr = ha.neighbors(n)
        if len(nbr)==6:
            neighbors[n,:] = nbr
    # Find the neighbors
    x2m = neighbors[:,0]
    x3m = neighbors[:,1]
    x1p = neighbors[:,2]
    x2p = neighbors[:,3]
    x3p = neighbors[:,4]
    x1m = neighbors[:,5]
    # Convert from spherical to rectangular cartesian coordinates
    X0,Y0 = latlon_2_xy(x0,y0)
    X1,Y1 = latlon_2_xy(x1,y1)


    X0rav = np.abs(X0).ravel()
    Y0rav = np.abs(Y0).ravel()
    X1rav = np.abs(X1).ravel()
    Y1rav = np.abs(Y1).ravel()
    
    # Here we start the calculations for the Cauchy-Green stress tensor 
    C = np.zeros((ha.N, 2, 2))

    if grid=='interp':
        dxx = X0rav[x1p] - X0rav[x1m]
        dyy = (Y0rav[x2p]+Y0rav[x3p])/2.0 - (Y0rav[x2m]+Y0rav[x3m])/2.0
        gradF = np.zeros((ha.N,2,2))
        gradF[:,0,0] = (X1rav[x1p] - X1rav[x1m])/dxx
        gradF[:,0,1] = ((X1rav[x2p] + X1rav[x3p])/2.0 - (X1rav[x2m] + X1rav[x3m])/2.0) /dyy
        gradF[:,1,0] = (Y1rav[x1p] - Y1rav[x1m])/dyy
        gradF[:,1,1] = ((Y1rav[x2p] + Y1rav[x3p])/2.0 - (Y1rav[x2m] + Y1rav[x3m])/2.0) /dyy
        for i in range(ha.N):
            C[i] = np.dot(np.transpose(gradF[i]),gradF[i])
    else:
        dx1 = np.sqrt((X0rav[x1p] - X0rav[x1m])**2 +(Y0rav[x1p] - Y0rav[x1m])**2 )
        dx2 = np.sqrt((X0rav[x2p] - X0rav[x2m])**2 + (Y0rav[x2p] - Y0rav[x2m])**2)
        dx3 =-np.sqrt((X0rav[x3p] - X0rav[x3m])**2 + (Y0rav[x3p] - Y0rav[x3m])**2)
        gradF = np.zeros((ha.N,2,3))
        gradF[:,0,0] = (X1rav[x1p] - X1rav[x1m])/dx1
        gradF[:,0,1] = (X1rav[x2p] - X1rav[x2m])/dx2
        gradF[:,0,2] = (X1rav[x3p] - X1rav[x3m])/dx3
        gradF[:,1,0] = (Y1rav[x1p] - Y1rav[x1m])/dx1
        gradF[:,1,1] = (Y1rav[x2p] - Y1rav[x2m])/dx2
        gradF[:,1,2] = (Y1rav[x3p] - Y1rav[x3m])/dx3
        
        theta = np.arctan(fs.dy/fs.dx)
	# This is the Jacobian-Transformation matrix to convert from the hexagonal grid to regular grid
        #JT = np.array([[1.0,0.0],[1./2., np.sqrt(3.0)/2.0],[-1./2., np.sqrt(3.0)/2.0]])
        JT = np.array([[1.0,0.0],[np.cos(theta) , np.sin(theta)],[-np.cos(theta), np.sin(theta)]])

        C = np.zeros((ha.N, 2, 2))
        for i in range(ha.N):
            C[i] = np.dot(np.dot(np.transpose(JT),np.transpose(gradF[i])), np.dot(gradF[i],JT))
    
    Cnew = np.reshape(C, ((X0.shape[0],X0.shape[1],2,2)))[1:-1,1:-1]
    eigs1 = np.zeros((X0.shape[0]-2,X0.shape[1]-2))
    eigs2 = np.zeros((X0.shape[0]-2,X0.shape[1]-2))

    for i in range(X0.shape[0]-2):
        for j in range(X0.shape[1]-2):
            eigval,eigvec = np.linalg.eig(Cnew[i,j])
            eigs1[i,j] = eigval.max()
            eigs2[i,j] = eigval.min()
    
    lamb1 = np.zeros_like(X0)
    lamb1[1:-1,1:-1] = eigs1
    lamb2 = np.zeros_like(X0)
    lam2 = np.zeros_like(X0)
    lam2[1:-1,1:-1] = eigs2
    lamb2[1:-1,1:-1] = eigs1**-1
    lamb_mask = (lamb1<=1.)

    if calc_T_D:
        T = Cnew[:,:,0,0] + Cnew[:,:,1,1]
        D = Cnew[:,:,0,0]*Cnew[:,:,1,1] - Cnew[:,:,0,1]*Cnew[:,:,1,0]

        rt = 0.25*T**2 - D
        #if np.any(rt < 0):
        #    print 'There will be imaginary eigenvalues'

        
        lamb1 = np.zeros_like(X0)
        lamb2 = np.zeros_like(X0)

        # large eigenvalue and eigenvector
        lam1 = 0.5*T + np.sqrt(rt)
        lamb1[1:-1,1:-1] = lam1

    
        # these should all be greater than 1, mask otherwise
        lamb_mask = lamb1<=1.

        # small eigenvalue and eigenvector
        # (direct computation is numerically unstable)
        lam2b = 0.5*T - np.sqrt(rt)

        # instead, compute as inverse of lam2
        lam2 = lam1**-1
        lamb2[1:-1,1:-1] = lam2
    
    lamb1 = np.ma.masked_array(lamb1, lamb_mask)
    lamb2 = np.ma.masked_array(lamb2, lamb_mask)
    
    new_mask = np.ma.mask_or(lav_mask,lamb_mask)
    return lamb1, lamb2, new_mask

def ddt(X):
    dxdt = X.diff('time')/X['time'].diff('time')
    return dxdt

def rel_disp(bcolz_dir):
    bc = bcolz.open(rootdir=bcolz_dir, mode='r')
    df = bc.todataframe()
    # remove duplicates
    df = df.drop_duplicates(subset=['npart', 'time'])
    
    df_reindex = df.set_index(['time', 'npart'])
    ds = xr.Dataset(df_reindex)
    
    ds = ds.unstack('dim_0')
    # now the dataset is two dimensional

    # get rid of -999.0 mask value
    ds['vort'] = ds['vort'].where(ds['vort'] != -999.0)

    deltax= R*(2*np.pi)/360. * np.cos((ds.y+ds.y.isel(time=0))/2. *np.pi/180.0)* (ds.x - ds.x.isel(time=0))
    deltay= R*(2*np.pi)/360. * (ds.y - ds.y.isel(time=0))

    # Mean Drift
    M_x = (deltax).mean(dim='npart')
    M_y = (deltay).mean(dim='npart')

    Npart = ds.npart.shape[0]

    # Dispersion
    D_x = (Npart-1)**-1 * ((deltax - M_x)**2.).sum(dim='npart')
    D_y = (Npart-1)**-1 * ((deltay - M_y)**2.).sum(dim='npart')
    
    # Diffusivity (as the rate of change of Absolute Dispersion)
    K_x = 0.5 * ddt(D_x)
    K_y = 0.5 * ddt(D_y)
    return M_x, M_y, D_x, D_y, K_x, K_y, deltax, deltay


