import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def bcolz_to_array(time, bc, fs, colname='y', lock=None, drop_duplicates=False):
    """Convert bcolz data from a certain timestep to a 2d numpy array.
    For some reason, this is fast.
    
    Parameters
    ----------
    time : float
        the timestep value
    bc : bcolz ctable
        where the data lives
    fs : FloatSet
        needed to reshape the array properly
    """
    
    # convert to pandas dataframe
    if lock is not None:
        lock.acquire()
    df = pd.DataFrame(bc["(time==%g) & (npart>0)" % time])
    if drop_duplicates:
        df = df.drop_duplicates()
    if lock is not None:
        lock.release()
        
    # reindex the dataframe
    df = df.set_index('npart', drop=True, verify_integrity=True)
    
    # try to iterate series
    squeeze = False
    if isinstance(colname, basestring):
        colnames = [colname]
        squeeze = True
    else:
        colnames = [c for c in colname]
    
    res = [dataseries_to_array(df[c], fs) for c in colnames]
    if squeeze:
        return res[0]
    else:
        return res
    
def dataseries_to_array(ds, fs):
    # create index based on particle id
    npart = np.arange(1,(fs.Nx*fs.Ny)+1)
    assert ds.index.name == 'npart'
    ds = ds.reindex(npart)
    assert len(ds) == (fs.Nx*fs.Ny)
    return ds.values.reshape((fs.Ny,fs.Nx))

def ndarray_to_list(ar):
    """Convert numpy array to regular python list of floats.
    This is necessary to serialize to json."""
    return [float(v) for v in ar]
    
def zip_coordinates(x, y):
    """Convert two numpy arrays to coordinate pairs for
    geojson serialization."""
    return list(zip(ndarray_to_list(x), ndarray_to_list(y)))

def daynum_to_datetime(nday, refdate=datetime(1993,1,1)):
    """Convert the experiment day number to a datetime."""
    return refdate + timedelta(days=nday)
    
# convert to json
def region_to_json(reg, lav, x0, y0, x1, y1, area, df, fname_base):
    """Convert a convex region from hexgrid to json format.
    
    Parameters
    ----------
    reg : floater.hexgrid.HexArrayRegion
        The region to process
    lav : arraylike
        The lagrangian averaged vorticity for the region
    x0 : arraylike
        Ravelled array of initial x positions
    y0 : arraylike
        Ravelled array of initial y positions
    x1 : arraylike
        Ravelled array of final x positions
    y1 : arraylike
        Ravelled array of final y positions
    area : arraylike
        Ravelled array of parcel areas
    df : pandas.Dataframe
        Full particle data
    fname_base : str
        The prefix associated with this specific Lagrangian model run
    """
    
    # npart and index are off by 1
    npart = reg.first_point+1
    eddy_id = fname_base + '_%010d' % npart
    
    day0 = int(fname_base[5:9])
    day1 = int(fname_base[10:14])
    ndays = day1-day0
    date_start = daynum_to_datetime(day0)
    date_end = daynum_to_datetime(day1)
    
    idx = list(reg.members)
    x0_reg = x0[idx]
    y0_reg = y0[idx]
    x1_reg = x1[idx]
    y1_reg = y1[idx]
    region_area = float(area[idx].sum())
    # .sum() is WRONG! should be .mean()!
    region_lav = float(lav.ravel()[idx].mean())
    
    x0_center = float(x0[reg.first_point])
    y0_center = float(y0[reg.first_point])
    x1_center = float(x1[reg.first_point])
    y1_center = float(y1[reg.first_point])
    
    df_center_point = df[df.npart==(npart)].sort('time')
    traj_x = df_center_point['x'].values
    traj_y = df_center_point['y'].values
    
    # holds one eddy
    data = {
        '_id': eddy_id,
        # tells us we have a geojson object
        'type': 'FeatureSet',
        # indices for mongodb searching
        'loc_start':  [x0_center, y0_center],
        'loc_end': [x1_center, y1_center],
        'date_start': date_start,
        'date_end': date_end,
        'duration': ndays,
        'area': region_area,
        'lav': region_lav,
        'features': [
            {'type': 'Feature',
             'properties': {'name': 'start_center'},
             'geometry': {
                'type': 'Point',
                'coordinates': [x0_center, y0_center]}
            },
            {'type': 'Feature',
             'properties': {'name': 'end_center'},
             'geometry': {
                'type': 'Point',
                'coordinates': [x1_center, y1_center]}
            },
            {'type': 'Feature',
             'properties': {'name': 'trajectory'},
             'geometry': {
                'type': 'LineString',
                'coordinates': zip_coordinates(traj_x, traj_y)}
            },
            {'type': 'Feature',
             'properties': {'name': 'start_points'},
             'geometry': {
                'type': 'MultiPoint',
                'coordinates': zip_coordinates(x0_reg, y0_reg)}
            },
            {'type': 'Feature',
             'properties': {'name': 'end_points'},
             'geometry': {
                'type': 'MultiPoint',
                'coordinates': zip_coordinates(x1_reg, y1_reg)}
            },
        ]
    }
    return data
