#!/usr/bin/env python

"""
SWMM Input (.inp) file creation

Usage:
  wncSWMM.py <path> <start> <days>[options]

Examples:
  Create SWMM input files in .../Feb/MF/ with 10 days:
  wncSWMM.py  /Users/bb/Google_Drive/WNC/Coupled/Feb 10 -t

Arguments:
  PATH        Path .../Coupled/[month]
  start       Start date of simulation (dd/mm/YYYYY)
  days        Total number of DAYS
              Actual Days SWMM Runs - SAME as MF kper - 1 less than COUPLED

Options:
  -p, --wilt=FLOAT    Wilting Point/UMC                 [default: 0.091]
  -v, --verbose=BOOL  Print SWMM Final Msg              [default: 1]
  -w, --write=BOOL    Write the .inp file               [default: 1]
  -h, --help          Print this message

Notes:
    Directory with csv files  must be at <path>/Data. Can be symlink.
    Data is prepared in swmmfPREP.py
    Engine is swmmput.py

    Wilting Point controls to a large extent GW EVAP

    UZFB files must all be present; run wncMF.py with 1 SS and 1 Trans Step
    Make sure only the current UZFB files are in the PATH_root/MF directory
    Full Grid
    Hourly Simulation

###
Author : Brett Buzzanga
Created: 2016-12-28
Updates:
    2017-01-16: Documentation and takes system date for filename
    2017-01-31: Refactoring, added soil moisture
    2017-02-02: Refactoring, index alignment
    2017-02-16: Refactoring, call as module
    2017-02-22: Added start year option, removed time step option
    2017-02-27: Changed start year to start date
To Do:
    Make renaming of rain/climate file more robust
    Should probably move to updates function
"""

from BB_PY import BB
from swmmPUT import *
import bcpl
import os
import os.path as op
#
import time
from datetime import datetime, timedelta
#
import numpy as np
import pandas as pd
#
from docopt import docopt
from schema import Schema, Use

def swmm_load(path_root):
    """ Load data for SWMM input file creation. """
    PATH_data            = op.join(path_root, 'Data')
    swmm_data = {}
    swmm_data['subs']    = pd.read_csv(op.join(PATH_data, 'SWMM_subs.csv'),
                                       index_col='Zone')
    swmm_data['jcts']    = pd.read_csv(op.join(PATH_data,'SWMM_junctions.csv'),
                                       index_col='Zone')
    swmm_data['links']   = pd.read_csv(op.join(PATH_data, 'SWMM_links.csv'),
                                       index_col='Zone')
    swmm_data['outs']    = pd.read_csv(op.join(PATH_data, 'SWMM_outfalls.csv'),
                                       index_col='Zone')
    swmm_data['stors']   = pd.read_csv(op.join(PATH_data, 'SWMM_storage.csv'),
                                       index_col='Zone')
    swmm_data['inflows'] = pd.read_csv(op.join(PATH_data, 'SWMM_inflows.csv'),
                                       index_col='Node')
    # combine for mapping purposes
    swmm_data['nodes'] = pd.concat([swmm_data['jcts'], swmm_data['outs'], swmm_data['stors']])
    return swmm_data

def mf_load(path_root, df_subcatchs, **params):
    """ Load steady state head and soil moisture data from uzfb files """
    # retrieve head and soil moisture from uzfb files (these only have subcatchs);

    mat_mf          = bcpl.mf_get_all(path_root, mf_step=0, **params) # 0 is steady state
    df_mf           = pd.DataFrame(mat_mf, index=mat_mf[:,4], columns=['ROW', 'COL',
                                    'HEAD', 'THETA', 'IDX']).drop('IDX', axis=1)
    try:
        df_mf.index = df_subcatchs.index
    except:
        raise SystemError('Incorrect amount of uzfb gage files. Run MF 0 (SS).')
    return df_mf

def swmm_objs(path_root, swmm_df, mf_df, **params):
    """ Create SWMM input file Sections. """
    # sub data
    SUBS     = swmm_df['subs'].index.tolist()
    # select cells by hand in arc, convert to csv, send to excel
    OUTS     = swmm_df['outs'].index.tolist()
    # create junction at each outfall to stream
    JNCTS    = swmm_df['jcts'].index.tolist()
    # combine all nodes for mapping purposes
    NODES    = swmm_df['nodes'].index.tolist()
    # create stream reaches in excel; one per constant head cell
    LINKS    = swmm_df['links'].index.tolist()
    # select storage units using python list in swmm_surface
    STOR     = swmm_df['stors'].index.tolist()
    # junction and storage units - lakes (aka chd nodes)
    INFLOWS  = swmm_df['inflows'].index.tolist()

    ##### ONLY UPDATING CONSTANTS
    Title   = "\n".join(['Brett Buzzanga', time.strftime('%A %B %d, %Y'),
                         'Input File Created: ' + time.strftime('%X'),
                         'SLR: {}'.format(params.get('slr'))])
    title    = SwmmPd(title=Title).final_dict()
    evap     = EvapInp().final_dict()
    temp     = TempInp(op.join(path_root, 'Data', 'climate_full.DAT')).final_dict()
    rain     = RainInp(op.join(path_root, 'Data', 'rain_full.DAT'), [0]).final()
    report   = ReportInp().final_dict()
    opts     = OptInp(**params).final_dict()
    aq       = AqInp(**params).final()
    outs     = OutInp(OUTS, **params).final()
    stors    = StorInp(STOR, **params).final()

    ##### WILL OVERWRITE COLUMNS
    sub      = SubInp(SUBS, **params)
    area     = AreasInp(SUBS, **params)
    infil    = InfInp(SUBS, **params)
    gw       = GwInp(SUBS, **params)
    juncts   = JnctInp(JNCTS, **params)
    streams  = LinkInp(LINKS, **params)
    tsects   = TsectsInp(LINKS, **params)
    inflows  = InflowsInp(INFLOWS, **params)
    coords   = CoordInp(NODES)
    polys    = PolyInp(SUBS)

    ### SUBCATCHMENTS
    #sub.update_col('Outlet',    swmm_df['subs'].Outlet, dtype=int)

    sub.df['Outlet']    = swmm_df['subs'].Outlet.astype(int)
    sub.df['perImperv'] = swmm_df['subs'].perImperv
    sub.df['perSlope']  = swmm_df['subs'].perSlope
    sub = sub.final()

    ### SUBAREAS
    area.df['N-Perv'] =  swmm_df['subs'].N_Perv
    area.df['S-Perv'] =  swmm_df['subs'].S_Perv
    area = area.final()

    ### INFILATRATION
    infil.df['MinRate'] = swmm_df['subs'].Min_Inf
    inf = infil.final()

    ### GROUNDWATER
    gw.df['Esurf'] = swmm_df['subs'].Esurf
    # From MODFLOW step 1 of Trans (SS)
    gw.df['Egw']   =  mf_df.HEAD
    gw.df['Umc']   = mf_df.THETA
    gw = gw.final()

    ### JUNCTIONS
    # put bottom of node to the bottom of the stream; assume land surface is top of stream
    juncts.df['Elevation'] = swmm_df['jcts']['invert_z'] - params.get('Depth', 0)
    juncts.df.MaxDepth = params.get('Depth', 0) - params.get('surf', 0)
    jncts = juncts.final()

    ### STREAMS
    # no init flow; it would double tidal inflow to nodes, which aint right
    #streams.df['InitFlow']  = swmm_df['links'].conv2cms * (0.3048 + params.get('slr', 0))
    streams.df['From_Node'] = swmm_df['links'].From_Node
    streams.df['To_Node']   = swmm_df['links'].To_Node
    streams.df['InOffset']  = swmm_df['links'].InOffset
    conduits = streams.final()

    ### STREAM GEOMETRY
    tsects.df.WIDTH = swmm_df['links'].Geom2_width
    geom = tsects.final()

    ### INFLOWS
    inflows.df['Baseline'] = swmm_df['inflows'].conv2cms * (0.3048 + params.get('slr'))
    #inflows.df.loc[STOR[:-2], 'Baseline'] = 200
    inflow = inflows.final()

    ### NODE COORDINATES
    coords.df['X-Coord'] =  swmm_df['nodes'].CENTROID_X
    coords.df['Y-Coord'] =  swmm_df['nodes'].CENTROID_Y
    xynodes = coords.final()

    ### POLY COORDINATES
    polys.df['X-Coord']  =  swmm_df['subs'].CENTROID_X
    polys.df['Y-Coord']  = swmm_df['subs'].CENTROID_Y
    xypolys = polys.final()
    return [title, opts, evap, temp, rain, sub, area, inf, aq, gw, jncts, outs,
                        stors, conduits, inflow, geom, report, xynodes, xypolys]

def swmm_writeinp(path_root, to_write, v=True, **params):
    '''
    Purpose:
        Write the swmm.inp file.
    Arguments:
        path_root = (+ /SWMM/) is location file will be written
        to_write  = list of swmmput objects
        swmm_days = SET FROM PARAMETERS (prob ignore following 3 lines)
                    ACTUAL amount of days (1 is be added automatically for SWMM)
                    One less than coupled step.
                    Equal to MF Kper (check). Set
    '''
    # form = '%m/%d/%Y'
    # start = datetime.strptime(params.get('START_DATE', '06/30/2011'), form)
    # end   = datetime.strptime(params.get('END_DATE', '12/31/2012'), form)
    # swmm_days = (end-start-timedelta(days=1))
    swmm_days = params.get('days', -1)

    opts  = [i for i, items in enumerate(to_write) if '[OPTIONS]' in items][0]
    tstep =[each for each in to_write[opts]['[OPTIONS]'] if 'REPORT_STEP' in each][0][1]

    if isinstance(to_write, list):
        inp_name      = '{}.inp'.format(params.get('name', 'missing_name'))
        inp_file      = op.join(path_root, 'SWMM', inp_name)
        writedicts(inp_file, to_write)
    else:
        raise TypeError('Must pass a list of swmmput objects')

    if v:
        print '  ***********************************', \
              '\n  SWMM Input File Created at: ', \
              '\n  ...',"/".join(BB.splitall(inp_file)[-4:]), \
              '\n  SWMM Time Step:', tstep, '(hr:min:sec)', \
              '\n  SWMM simulation length:', swmm_days, 'days', \
              '\n  ***********************************\n'

def main(path_root, write=True, verbose=True, **params):
    swmm_Dfs      = swmm_load(path_root)
    mf_Df         = mf_load(path_root, swmm_Dfs['subs'], **params)
    swmm_objects  = swmm_objs(path_root, swmm_Dfs, mf_Df, **params)

    if write:
        swmm_writeinp(path_root, swmm_objects, v=verbose, **params)

if __name__ == '__main__':

    arguments = docopt(__doc__)
    typecheck = Schema({'<path>'    : os.path.exists, '<days>'  : Use(int),
                        '<start>'   : Use(str),       '--wilt'  : Use(float),
                        '--verbose' : Use(int),       '--write' : Use(int)},
                        ignore_extra_keys=True)

    args   = typecheck.validate(arguments)
    form   = '%m/%d/%Y'
    name   = 'SLR-0.0_{}'.format(time.strftime('%m-%d'))
    s_date = datetime.strptime(args['<start>'], form)
    e_date = datetime.strftime(s_date + timedelta(days=args['<days>']+1), form)
    params = {'wp': args['--wilt'], 'START_DATE' : args['<start>'], 'END_DATE' : e_date}

    main (args['<path>'], name, args['--write'], args['--verbose'], **params)
