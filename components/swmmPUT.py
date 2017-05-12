"""
SWMM Input File Module.  Contains the SWMM token classes.
Use the classes to create objects and write a .inp module that runs SWMM.
Access with 'from modwmm.swmmput import *'
"""

import os
import os.path as op
import time

from collections import OrderedDict
import csv

import numpy as np
import pandas as pd
from datetime import datetime

# set path for switching between OS
month        = time.strftime('%b')

class SwmmPd(object):
    def __init__(self, names=[], title=''):
        """ Name is list of Subcatch, Aquifer, or Outfall Names """
        # these only for the parent (title) class
        self.token     = '[TITLE]'
        self.headers   = [title]
        self.constants = [''] * len(self.headers)
        self.dict      = OrderedDict(zip(self.headers, self.constants))
        self.names     = names

    def make_df(self):
        # make dataframe
        init = np.resize(self.constants, [len(self.names), len(self.constants)])
        self.df = pd.DataFrame(init, index=self.names, columns=self.headers)
        self.df.index.rename(';;{}'.format('Name'), inplace=True)
        return self.df

    def final(self):
        """ Final format for printing dataframe type """
        try:
            self.df
        except:
            self.df = self.make_df()
        return {self.token : self.df}

    def final_dict(self):
        ''' Final format for printing dictionary type'''
        out_list = _swmput_list(self.dict)
        return {self.token : out_list}

    def __str__(self):
        return 'Instance of class: {}'.format(self.token)

### Dictionaries ###

class EvapInp(SwmmPd):
    def __init__(self):
        self.token = '[EVAPORATION]'
        self.headers = ['TEMPERATURE', 'DRY_ONLY']
        self.constants = ['', 'NO']
        self.dict = OrderedDict(zip(self.headers, self.constants))

class TempInp(SwmmPd):
    def __init__(self, climate_file):
        self.token = '[TEMPERATURE]'
        self.headers = ['FILE', 'WINDSPEED FILE', 'SNOWMELT', 'ADC IMPERVIOUS',
                        'ADC PERVIOUS']
        self.snowmelt = ['34', '0.5', '0.6', '0.0', '50.0', '0.0']
        self.adc_imperv = ['{} '.format('1.0') * 10]
        self.adc_perv = ['.10','0.35', '0.53', '0.66','0.75','0.82','0.87',
                         '0.92', '0.95', '0.98']
        self.constants = [climate_file, '', '\t'.join(self.snowmelt),
                       '\t'.join(self.adc_imperv), '\t'.join(self.adc_perv)]
        self.dict = OrderedDict(zip(self.headers, self.constants))

class ReportInp(SwmmPd):
    def __init__(self):
        self.token = '[REPORT]'
        self.headers = ['INPUT', 'CONTROLS', 'SUBCATCHMENTS', 'NODES', 'LINKS']
        self.constants = ['NO', 'NO', 'ALL', 'ALL', 'ALL']

        self.dict = OrderedDict(zip(self.headers, self.constants))

### DataFrames ###
class RainInp(SwmmPd):
    def __init__(self, rain_file, names=[0], **params):
        SwmmPd.__init__(self, names)

        self.token = '[RAINGAGES]'
        self.headers = ['Format', 'Interval', 'SCF', 'Source', 'Path',
                        'StationID', 'Units']
        self.defaults = ['INTENSITY', '1:00', '1.0', 'FILE', rain_file,
                         '446139', 'MM']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]

class OptInp(SwmmPd):
    def __init__(self,  **params):
        self.token = '[OPTIONS]'
        self.headers = ['FLOW_UNITS', 'INFILTRATION', 'FLOW_ROUTING', 'LINK_OFFSETS',
            'MIN_SLOPE', 'ALLOW_PONDING', 'SKIP_STEADY_STATE', 'IGNORE_SNOWMELT',
            'IGNORE_QUALITY', 'START_DATE', 'START_TIME', 'REPORT_START_DATE',
            'REPORT_START_TIME', 'END_DATE', 'END_TIME', 'SWEEP_START', 'SWEEP_END',
            'DRY_DAYS', 'REPORT_STEP', 'WET_STEP', 'DRY_STEP', 'ROUTING_STEP',
            'INERTIAl_DAMPING', 'NORMAL_FLOW_LIMITED', 'FORCE_MAIN_EQUATION',
            'VARIABLE_STEP', 'LENGTHENING_STEP', 'MIN_SURFAREA', 'MAX_TRIALS',
            'HEAD_TOLERANCE', 'SYS_FLOW_TOL', 'LAT_FLOW_TOL', 'MINIMUM_STEP', 'THREADS']

        self.defaults = ['CMS', 'HORTON', 'KINWAVE', 'DEPTH',
                         '0', 'YES','NO', 'YES',
                         'YES', '06/30/2011', '23:00:00', '06/30/2011',
                         '00:00:00', '12/31/2012', '00:00:00', '01/01', '12/31',
                         '0', '01:00:00', '01:00:00', '01:00:00', '01:00:00',
                         'PARTIAL', 'BOTH','H-W',
                         '1', '0', '1.14','8',
                         '0.0015', '5', '5', '0.5', '4']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.dict      = OrderedDict(zip(self.headers, self.constants))
        self.dict['REPORT_START_TIME'] = self.dict['START_TIME']
        self.dict['REPORT_START_DATE'] = self.dict['START_DATE']

class SubInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)
        self.token      = '[SUBCATCHMENTS]'
        self.headers    = ['Rain_Gage', 'Outlet', 'Area', 'perImperv', 'Width',
                           'perSlope', 'CurbLen', 'SnowPack']
        self.defaults   = [0, 2, 4, 41, 200, 4.93, 0, '']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class AreasInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)
        self.token      = '[SUBAREAS]'
        self.headers    = ['N-Imperv', 'N-Perv', 'S-Imperv', 'S-Perv', 'PctZero',
                           'RouteTo', 'PctRouted']
        self.defaults   = [0.011, 0.015, 0.05, 2.54, 0, 'OUTLET', 100]
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class InfInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token      = '[INFILTRATION]'
        self.headers    = ['MaxRate', 'MinRate', 'Decay', 'DryTime', 'MaxInfil']
        self.defaults   = [50, 5, 10, 7, 0]
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df = self.make_df()

class AqInp(SwmmPd):
    def __init__(self, names=['Surficial'], **params):
        SwmmPd.__init__(self, names)

        self.token = '[AQUIFERS]'
        self.headers = ['Por', 'WP', 'FC', 'Ksat', 'Kslope', 'Tslope',
                        'ETu', 'Ets', 'Seep', 'Ebot', 'Egw', 'Umc', 'ETupat']
        self.wiltdf   = 0.091
        self.defaults = [0.4, self.wiltdf, 0.3, 5, 20, 0.01,
                            1, 0, 0, 0, 0, self.wiltdf, '']

        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        # manually make Umc equivalent to WP
        self.constants[-2] = self.constants[1]

class GwInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token    = '[GROUNDWATER]'
        self.headers  = ['Aquifer ', ' Node', 'Esurf',
                         'a1', 'b1', 'a2', 'b2', 'a3',
                         'Dsw', 'Egwt', 'Ebot', 'Egw', 'Umc']
        self.aqname   = 'Surficial'
        self.defaults = [self.aqname, 13326, '?Z?',
                         0.0, 0, 0, 0, 0,
                         0, '*', 0, '*', '*']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class JnctInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token     = '[JUNCTIONS]'
        self.headers   = ['Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded']
        self.defaults  = ['*', '*', 0, 0, 0]
        self.constants = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df        = self.make_df()

class OutInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token      = '[OUTFALLS]'
        self.headers    = ['ELEvation', 'Type', 'Stage_Data', 'Gated', 'RouteTo']
        self.defaults   = [0, 'FREE', '', 'NO', '']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class StorInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token   = '[STORAGE]'
        self.headers = ['eleVATION', 'MAXDepth', 'INITDepth', 'Shape',
                        'A1', 'A2', 'A0', 'N/A', 'Fevap',
                        'PSI', 'KSAT', 'IMD']
        self.defaults = [-3, 4, 1, 'FUNCTIONAL',
                          0, 0, 40000, 0, 0.95,
                          '', 0.0114, '']

        # !! CURVE NAME/PARAMS HAS 3 ITEMS ASSOCIATED WITH IT; HERE SEPARATED
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df = self.make_df()

class LinkInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token = '[CONDUITS]'
        self.headers = ['From_Node', 'To_Node', 'Length', 'Roughness', 'InOffset',
                                              'OutOffset', 'InitFlow', 'MaxFlow']

        self.defaults = ['*', '*', 200, 0.015, 0, 0, 0.00, 0]

        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df = self.make_df()

class TsectsInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token    = '[XSECTIONS]'
        self.headers  = ['Shape', 'Depth', 'WIDTH', 'Side1', 'Side2', 'Barrels', 'Culvert']
        self.defaults = ['TRAPEZOIDAL', 1.5, 10, 0.5, 0.5, 1, '']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class InflowsInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)
        self.token    = '[DWF]'
        self.headers  = ['Constituent', 'Baseline', 'Pattern']
        self.defaults = ['FLOW', 1.0, '""']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df = self.make_df()

class CoordInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token = '[COORDINATES]'
        self.headers = ['X-Coord', 'Y-Coord']
        self.defaults = ['*', '*']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

class PolyInp(SwmmPd):
    def __init__(self, names=[], **params):
        SwmmPd.__init__(self, names)

        self.token = '[POLYGONS]'
        self.headers = ['X-Coord', 'Y-Coord']
        self.defaults = ['*', '*']
        self.constants  = [params.get(head, default) for head, default in
                                               zip(self.headers, self.defaults)]
        self.df         = self.make_df()

def _swmput_list(nested_dict):
    '''
    Pulls first or second nested layer of dictionary and aligns with outer key.
    Returns a list.
    Use for SWMM input file print format.
    '''
    sub_list = []
    for key, values in nested_dict.items():
        if isinstance(values, str):
            sub = [key] + [values]
            sub_list.append(sub)
        elif isinstance(values, dict):
            sub = [key] + [val for val in values.values()]
            sub_list.append(sub)
    return sub_list

def writedicts(filename, lst):
    """
    Writes a list of dictionaries (lst) to filename.
    Dictionary value can be a pandas dataf ram
    Keys get their own line.
    Values are tab seperated (for SWMM).
    """
    with open(filename, 'w') as fh:
        for heading in lst:
            for key, values in heading.items():
                fh.write('{}\n'.format(key))
                # DEPRECATED
                if isinstance(values, dict):
                    for k, v in values.items():
                        fh.write('{}\t{}\n'.format(k, v))
                # for data frames
                elif isinstance(values, pd.DataFrame):
                    values.to_csv(fh, sep="\t", quoting=csv.QUOTE_NONE)

                # DEPRECATED; list has been replaced with pd.DataFrame
                elif isinstance(values, list):
                    for each in values:
                        each = '\t'.join(each)
                        fh.write('{}\n'.format(each))
                else:
                    fh.write('{}\n'.format(values))
            fh.write( "\n" )

def __format_data(path_file, write=True):
    """
    Format a rain (COOP) or climate (GHCND) raw data file for SWMM
    Copy and paste the 'download' link into a text file:
    https://www.ncdc.noaa.gov/cdo-web/datasets (precip hourly / normals daily)
    see rain_climate.py for old functions
    """
    kind = op.basename(path_file).split('_')[0]
    year = op.basename(path_file).split('_')[1]
    raw = pd.read_csv (path_file)

    if kind == 'rain':
        # want station ID, date, and precip
        df        = raw.loc[:, ['STATION', 'DATE', 'HPCP']]
        # overwrite station ID
        df.STATION = df.get_value(0, 'STATION').split(':')[1]
        # convert to date time and format
        df.DATE    = pd.to_datetime(df.DATE)
        df.DATE    = df.DATE.apply(lambda x: datetime.strftime(x, '%Y %m %d %H %M'))
        #print df
        df.to_csv(op.join(op.dirname(path_file), '{}_{}.DAT'.format(kind, year)),
                        sep=' ', float_format='%.2f', index=False, header=False,
                        quoting=csv.QUOTE_NONE, escapechar=' ')

    elif kind == 'climate':
        # using headers and then skipping the first row raises copy warning
        df = raw.loc[:, ['STATION', 'DATE', 'TMAX', 'TMIN', 'AWND']]
        df.STATION = df.get_value(0, 'STATION')[-6:]
        df.DATE =  pd.to_datetime(df.DATE, format='%Y%m%d')
        df.DATE    =  df.DATE.apply(lambda x: datetime.strftime(x, '%Y %m %d'))

    if write:
        df.to_csv(op.join(op.dirname(path_file), '{}_{}.DAT'.format(kind, year)),
                        sep=' ', float_format='%.2f', index=False, header=False,
                        quoting=csv.QUOTE_NONE, escapechar=' ')
    else:
        print df.head(10)
        try:
             df.HPCP.astype(float)
             print df.HPCP.sum() - 6.35 - 0.25 - 3.05 - 0.25 - 1.27
        except:
            print 'Using Climate data'

#PATH_root = op.join('/', 'Volumes', 'BB_4TB', 'Backups')
#__format_data(op.join(PATH_root, 'Data', 'rain_full_raw.csv'), write=True)
