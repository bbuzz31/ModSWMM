#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pull data from USGS data files.
Compare with SWMM.
"""

import BB
import os.path as op
import time

from ../components import swmmtoolbox as swmtbx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#0204293125, 0204295505, 0204297575, 02043016, 02043120, 02043155, 02043190, 02043200, 02043210, 02043269, 02043390, 02043500, 365423076051300

def swmmTS(out_file, param, loc=False, dates=[0, -1], plot=False):
    """
    Pull a time series from a swmm binary '.out' file
    Arguments:
        Path to .out
        Param (eg 'precip')
        Loc ID number (eg 10735)
        Dates can be integers or datetimes
        Plot True if you want to see plots
    Returns:
        Pandas dataframe with datetime index
    """
    # map parameters to type, individual code, units (for plot), system code
    param_map = {'precip' : ['subcatchment,', ',0', 'mm', ',1'],
                 'flow'   : ['link,', ',2', 'm/s'],
                 'flood'  : ['node,', ',5', 'cms', ',10'],
                 'inf'    : ['subcatchment,', ',3', 'mm/hr?', ',3'],
                 'evap'   : ['subcatchment,', ',2', 'mm/hr?', ',13'],
                 'run'    : ['subcatchment,', ',4', 'mm/hr?', ',4'],
                 'head'   : ['subcatchment,', ',6', 'm'],
                 'soil'   : ['subcatchment,', ',7', 'theta'],
                 'gwout'  : ['subcatchment,', ',5', '??', '6?'],

                 'pet'    : ['subcatchment,', '', 'mm/hr?', ',14'],

                }
    swmtbx.listdetail(out_file, 'link')
    if loc:
        t_series_raw = swmtbx.extract(out_file, param_map[param][0] + str(loc)
                                                     + param_map[param][1])
    else:
        t_series_raw = swmtbx.extract(out_file, 'system' + param_map[param][3]
                                                     + param_map[param][3])
    t_series = t_series_raw[dates[0]:dates[1]]

    if plot:
        t_series.plot()
        plt.ylabel (param_map[param][2])
        plt.show()

    return t_series

def precipgage(PATH_usgs, plot=False):
    """ Format and plot time series of usgs precip gage data """
    usgs_raw  = pd.read_csv(PATH_usgs, sep='\t', comment='#')
    keep = ['datetime', '83096_00045_00001']
    usgs_data = usgs_raw[keep]
    # skip over first row which has formats
    usgs_data = usgs_data.iloc[1:]
    usgs_data.set_index(pd.to_datetime(usgs_data.datetime), inplace=True)
    usgs_data.drop('datetime', axis=1, inplace=True)
    colname = ['Precip']
    usgs_data.columns = colname
    usgs_data['Precip']
    #BB.DIE()
    usgs_data['Precip'] = usgs_data['Precip'].apply(lambda x: float(x) * 25.4)
    if plot:
        usgs_data.Precip.plot()
        plt.ylabel ('mm')
        plt.show()
    return usgs_data

def flowdata(PATH_usgs, plot=False):
    usgs_raw  = pd.read_csv(PATH_usgs, sep='\t', comment='#')
    # drop quality controlled columns
    not_qc =  [col for col in usgs_raw.columns if not '_cd' in col]
    # drop temp/sal
    not_qual = ['datetime', '83107_00055_00003', '83110_00065_00003', '83111_00060_00003']
    # no 'site_no' : 'Site' ?
    # ft/s ; ft; cfs
    colnames = ['DateTime', 'Mean_Stream_Vel', 'Max_Gage_Height', 'Mean_Discharge']

    # clean up
    usgs_data = usgs_raw[not_qc]
    usgs_data = usgs_data[not_qual]
    usgs_data.columns=colnames
    usgs_data.set_index(pd.to_datetime(usgs_data.DateTime), inplace=True)
    usgs_data.drop('DateTime', axis=1, inplace=True)
    #print usgs_data.head()

    # convert to m/s
    usgs_data['Mean_Stream_Vel'] = usgs_data['Mean_Stream_Vel'].apply(lambda x: x * 0.3048)
    # convert to m
    usgs_data['Max_Gage_Height'] = usgs_data['Max_Gage_Height'].apply(lambda x: x * 0.3048)
    # convert to cms
    usgs_data['Mean_Discharge']  = usgs_data['Mean_Discharge'].apply(lambda x: x* 0.0283168)
    if plot:
        #print usgs_data.head()
        usgs_data.Mean_Stream_Vel.plot()
        #usgs_data.Mean_Discharge.plot()
        plt.ylabel('m/s')
        plt.show()
    return usgs_data

def head_data(path_usgs):
    """
    Parse USGS head file 62C_8
    Latitude  36°45'33.22", Longitude  76°03'19.79" NAD27
    Land-surface elevation 15.00 feet above NGVD29
    The depth of the well is 60.0 feet below land surface.
    The depth of the hole is 106 feet below land surface.
    No two adjacent days (to see if tidal influence)
    """
    usgs_raw = pd.read_csv(path_usgs, sep='\t', comment='#')
    #lev_va is water level feet below land surface
    keep     = ['lev_dt', 'lev_va']
    well_df  = usgs_raw[keep]
    well_df  = well_df.iloc[1:]
    well_df  = well_df.rename(columns={'lev_dt': 'Date', 'lev_va' : 'Water_level'})
    well_df.set_index(pd.to_datetime(well_df.Date))
    return well_df

# WNC_rt149 only has precipitation
# Dates don't match up
' precip is essentially useless '

gage      = ['WNC_indian_river', 'WNC_rt149']
PATH_root = op.join(op.expanduser('~'), 'Google_Drive', 'WNC', 'Misc_Data', 'usgs_stream_gages')
PATH_well = op.join (op.join(op.expanduser('~'), 'Google_Drive', 'WNC', 'Coupled', 'Data'))
PATH_swmm    = op.join(op.expanduser('~'), 'Google_Drive', 'WNC', 'Coupled',
                               time.strftime('%b'), 'SWMM', 'SLR-0.0_04-01.out')
dates = [datetime(2012, 01, 01), datetime(2012, 7, 25)]

print flowdata(op.join(PATH_root, gage[0]), plot=False).dropna().resample('A').mean()

# compare to the link in
flow      = swmmTS(PATH_swmm, 'flow', 46, plot=False).resample('A').mean()
print flow - (0.3048* 200 / (3600*24) * 25) # if this conversion is right it's better
#precipgage(op.join(PATH_root, gage[1]), plot=False)
#head_data(op.join(PATH_well, 'well_62C_8.txt'))
