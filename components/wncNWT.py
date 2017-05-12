#!/usr/bin/env python
"""
Run MODFLOW2005 with FloPy Wrapper

Usage:
  wncMF.py <path> <kpers> <slr> [options]

Examples:
  Run a SS MF model with 1 stress periods and no SLR:
  wncMF.py /Users/bb/Google_Drive/WNC/Coupled/Feb 1 0

  Run a Transient MF model with 5 stress periods and 1m SLR
  wncMF.py . 5 1 -s 0

Arguments:
  path        Path .../Coupled/[month]
  kpers       Total number of MODFLOW Steps
  slr         Amount (m) to add to CHD (which is at 0.3048 m(

Options:
  -s, --steady=BOOL  Steady/Transient                  [default: 1]
  -v, --verbose=INT  Print MODFLOW messages(0:3)       [default: 1]
  -h, --help         Print this message

Notes:
  Directory with csv files  must be at <path>/Data. Can be symlink.
  Data is prepared in swmmfPREP.py

"""

import os
import os.path as op
import time

from subprocess import Popen
import flopy.modflow as flomf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from docopt import docopt
from schema import Schema, Use, Or

import shutil

def _chd_format(chd, layers=[0]):
    """
    Format list of lists (from table) for FloPy Wrapper.
    Adds layer for each active (value > 0 CHD cell. Corrects row/col to 0 index.
    Returns dictionary of stress period to values.
    If more stress periods than keys, the last key will be repeated.
    """
    chd_fmt = []
    for layer in layers:
        for each in chd:
        # if statement allows entire array (including non chd cells) to be given
            if each[2] > 0:
                chd_temp = [layer, each[0] -1, each[1] - 1, each[2], each[3]]
                chd_fmt.append(chd_temp)
    return {0:chd_fmt}

def _uzf_gage_format(listoflists):
    """
    Convert a tuple of row, column, IUZOPT to proper format for FloPy Wrapper.
    IUZOPT: 1 = print time, head, gwdis... 2 = 1 + rates, 3 = head, wter content
    Return Format:  {UNIT : [row, col, unit, iuzopt], UNIT2 : [...], ...}
    """
    from collections import OrderedDict
    gage_fmt = OrderedDict()
    for i, each in enumerate(listoflists):
        if len(each) > 1:
            #iftn = i + 20205
            iftn = i + 2001
            row = each[1]
            col = each[0]
            uzopt = 3
            gage_fmt[iftn] = [row, col, iftn, uzopt]
        # just summary if length is one
        elif len(each) == 1 and each[0] < 1:
            gage_fmt[-70] = [-70]
        else:
            print ('ERROR: UZF gages improperly formatted')
            return
    gage_fmt[-2000] = [-2000]
    #print gage_fmt
    return gage_fmt

def mf_init(path_root, coupled=False, **params):
    """ Paths to 'Data' dir. Time settings. Root slr_name. Exe locations"""
    ### initialize flopy modflow class
    NAME_mf  = '{}'.format(params.get('name', 'missing_name'))
    MODEL    = op.join(path_root, 'MF', NAME_mf)
    path_exe = params.get('path_exe')
    if coupled:
        MF  = flomf.Modflow(MODEL, exe_name=op.join(path_exe, 'NWT_BB'),
                                    version='mfnwt', silent=True, verbose=False,
                              external_path=op.join(path_root, 'MF', 'ext'))
    else:
        MF  = flomf.Modflow(MODEL, exe_name=op.join(path_exe, 'mfnwt'),
                                    version='mfnwt',
                              external_path=op.join(path_root, 'MF', 'ext'))
    ### load data from csvs
    data_mf        = {}
    data_mf['mf']  = pd.read_csv(op.join(path_root, 'Data', 'MF_GRID.csv'),
                                 index_col='FID')
    data_mf['chd'] = pd.read_csv(op.join(path_root, 'Data', 'MF_CHD.csv'),
                                 index_col='Zone')
    # for uzf gages
    data_mf['subs'] = pd.read_csv(op.join(path_root, 'Data', 'SWMM_subs.csv'),
                                 index_col='Zone')

    kpers = params.get('days', 1)
    #kpers = kpers*24
    if params['ss']:
        steady      = [True]  * kpers
    else:
        steady      = [False] * kpers;
        steady[0]   = True
    return [MF, data_mf, steady, kpers, coupled]

def mf_inps(init, **params):
    """ Initialize FloPy Classes
    Args:
        init from mf_init:
            Path where 'Data' dir is located
            # of Days
            steady state / transient
            kpers
            coupled
        possible params:
            strt, sy, vks, rates, thts, eps, extwc, extdp, slr
    """

    ### globals
    mf, mf_dfs, steady, kpers, coupled = init
    nrows    = int(max(mf_dfs['mf'].ROW));
    ncols    = int(max(mf_dfs['mf'].COLUMN_));
    last_col = len(mf_dfs['mf'].columns)
    nlays    = 7

    "-- NWT --"
    nwt      = flomf.ModflowNwt(mf, iprnwt=1)

    "-- DIS --"
    delrc    = 200
    ztop     = mf_dfs['mf']['MODEL_TOP'].values.reshape(nrows,ncols);
    botms    = np.stack([mf_dfs['mf'].ix[:, lay].values.reshape(nrows,ncols) for
                                     lay in range (last_col-7, last_col)])
    # prj = NAD_1983_2011_UTM_Zone_18N
    dis      = flomf.ModflowDis(mf, nlays, nrows, ncols, delr=delrc, delc=delrc,
                                botm=botms, top=ztop, nper=kpers, steady=steady,
                                xul=400232.64777, yul=4072436.50165, rotation=19,
                                proj4_str='EPSG:6347', start_datetime='06/30/2012')
    "-- BAS --"
    active = mf_dfs['mf']['IBND'].values.reshape(nrows,ncols)
    ibound = np.zeros((nlays, nrows, ncols), dtype=np.int32) + active
    bas    = flomf.ModflowBas(mf, ibound=ibound, strt=params.get('strt', 1), ichflg=True)

    "-- UPW --"
    hk = np.stack([mf_dfs['mf'].ix[:, lay].values.reshape(nrows,ncols)
                   for lay in range (last_col-14, last_col-7 )])
    vk = np.stack([mf_dfs['mf'].ix[:, lay].values.reshape(nrows,ncols)
                   for lay in range (last_col-21, last_col-14)])
    upw = flomf.ModflowUpw(mf, laytyp=np.ones(nlays), layavg=np.ones(nlays)*2,
                            hk=hk, vka=vk, sy=params.get('sy', 0.35))

    "-- UZF --"
    uzfbnd        = mf_dfs['mf']['UZF_IBND'].values.reshape(nrows,ncols)
    # list of 2 dimensional array; see finf array in uzf example notebook github
    Finf          = [mf_dfs['mf']['FINF'].values.reshape(nrows,ncols)]
    Pet           = [mf_dfs['mf']['ET'].values.reshape(nrows,ncols)]
    ss_rate       = params.get('ss_rate', 0.0002783601)
    # extend list to num of timesteps; overwritten in coupled simulation
    [Finf.append(Finf[0]) for   i in range (kpers-1)]
    # [Finf.append(0) for   i in range (kpers-1)]
    [Pet.append(Pet[0])   for   i in range (kpers-1)]
    # [Pet.append(0)   for   i in range (kpers-1)]


    ### Gage Settings
    sub_id_mf   = [sub_id - 10000 for sub_id in mf_dfs['subs'].index.tolist()]
    listofgages = mf_dfs['mf'].iloc[:,2:4][mf_dfs['mf'].OBJECTID.isin(sub_id_mf)].values
    gage_info   = _uzf_gage_format(listofgages)
    uzf = flomf.ModflowUzf1(mf, iuzfopt=1, ietflg=1, nuztop=1, iuzfbnd=uzfbnd,
                                                iuzfcb2=61, ntrail2=15,
                                                eps=params.get('eps', 5.0),
                                                vks=params.get('vks', 0.3),
                                                thts=params.get('thts', 0.40),
                                                surfdep=params.get('surf', 0.5),#
                                                nosurfleak=params.get('noleak', False),
                                                extdp=params.get('extdp', 2.5),
                                                extwc=params.get('extwc', 0.101),
                                        nuzgag=len(gage_info), uzgag=gage_info,
                                        finf=Finf, pet=Pet)
    "-- CHD --"
    chead    = mf_dfs['chd'][['ROW', 'COLUMN_', 'START', 'END']]
    # SLR ; df has 0.3048 stored as 0m rise
    chead['START'] += params.get('slr', 0)
    chead['END']   += params.get('slr', 0)
    chd_lst  = [list(row) for row in chead.values]
    chd_info = _chd_format(chd_lst, layers=range(nlays))
    chd      = flomf.ModflowChd(mf, stress_period_data=chd_info,
                                    options=['AUXILIARY IFACE NOPRINT MXACTC'])
    "-- OC --"
    # May have to alter spd[0] (stress period) to print other results.
    spd    = {(0, 0): ['save head', 'save drawdown', 'save budget']}
    ext    = ['oc', 'fhd', 'ddn', 'cbc']
    #units  = [14, 51, 57, 63, 65]
    # for formatted head file
    ihedfm = 0 #10
    fmt    = '(1X1PE13.5)'
    oc     = flomf.ModflowOc(mf, stress_period_data=spd, compact=True, #unitnumber=units,
                             ihedfm=ihedfm, chedfm=fmt, cddnfm=fmt, extension=ext)
    "-- PCG --"
    # pcg = flomf.ModflowPcg(mf, mxiter=60, iter1=20, hclose=0.25, rclose=0.25);

    "-- HOB --"
    # copied from file made by muse;
    # Were not 0 indexed in Hob, but should be in HeadObservation Package
    # changed to 0 index
    obs         = ['C.8', 'C.5', 'B.18', 'C.3', 'C.2', 'B.22']
    rows        = [26, 32, 67, 12, 12, 50]
    columns     = [20, 20, 19, 27, 27, 28]

    # timestep offset?
    roffs       = [0.2659268754837, 0.4356522229430, -0.3916411692253,
                   0.1791305761365, 0.1751442566724, 0.08775511517422]
    coffs       = [-0.169169106565, 0.103307718318, 0.4411342911341,
                   -0.4312416938972, -0.4298201227491, -0.259623546392]
    hobs        = [1.84, 1.48, 2.55, 1.95, 1.5, 0.97]

    # get final time and use for head obs comparing
    tmax = mf.dis.get_final_totim()
    hob_data = []
    for i, ob in enumerate(hobs):
        data = np.array([tmax, ob])
        hob_data.append(flomf.HeadObservation(mf, row=rows[i], column=columns[i],
                                roff=roffs[i], coff=coffs[i],
                                          obsname=obs[i], time_series_data=data))

    hob = flomf.ModflowHob(mf, iuhobsv=59, hobdry=-9999, obs_data=hob_data)
    return [nwt, dis, bas, upw, uzf, chd, oc, hob]

def mf_run(init, quiet=False):
    mf, mf_df, steady, kpers, coupled = init
    mf.write_name_file()
    if coupled:
        Popen([mf.exe_name, mf.namefile])
    else:
        if quiet == 1 or quiet == 3:
            quiet = False
        else:
            quiet = True
        ### silent doesn't work; i think flopy problem
        success, buff = mf.run_model(silent=True)
        if success and quiet:
            print '\n  ************************', \
                  '\n  MODFLOW (flopy) is done.'
            if kpers == 2:
                print '  Creating new SWMM .inp from MF SS gages'
            elif kpers > 2:
                trans = [step for step in steady if step is False]
                print '  Ran', str(len(trans)), 'transient steps.'
            print '  ************************'
        return success

def inp_plot(mf_objs, to_plot):
    """
    Plot MODFLOW input files. mf_objs generated by 'mf_inps'
    to_plot options: ['DIS', 'BAS6', 'LPF', 'UZF', 'CHD', 'OC', 'PCG', 'HOB']
    """
    # added ftype method to flopy.modflow.mfhob:
    obj_names = [mf_file.ftype() for mf_file in mf_objs]
    if not isinstance(to_plot, list): [to_plot]
    for obj in mf_objs:
        if obj.ftype() in to_plot:
            obj.plot()
    plt.show()

def write(mf_objs, remove=True):
    """ Write MODFLOW inp files. mf_objs generated by 'init'. Remove checks. """
    [mf_file.write_file() for mf_file in mf_objs]
    [os.remove(f) for f in os.listdir(os.getcwd()) if f.endswith('chk') ]
    return mf_objs

###########################################

def main_SS(path_root, quiet=False, **params):
    """ Run a SS simulation for generating SWMM inp (initial heads/wc) """
    params['days'] = 2
    mf_inits       = mf_init(path_root, coupled=False, **params)
    flopy_objs     = mf_inps(mf_inits, **params)
    write(flopy_objs)
    mf_run(mf_inits, quiet=quiet)

def main_TRS(path_root, quiet=False, **params):
    """ Run a Transient simulation, MODFLOW only """
    params['ss'] = False
    mf_inits     = mf_init(path_root, coupled=False, **params)
    flopy_objs   = mf_inps(mf_inits)
    write(flopy_objs)
    # for ext_file in os.listdir(op.join(os.getcwd(), 'ext')):
    #     if ext_file.startswith('FINF') or ext_file.startswith('PET'):
    #         # shutil.copy(ext_file, op.join(os.getcwd(),
    #         print ext_file

    mf_run(mf_inits, quiet=quiet)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    typecheck = Schema({'<kpers>'   : Use(int),  '<path>'  : os.path.exists,
                        '<slr>'     : Use(int),  '--verbose' : Use(int),
                        '--steady'  : Use(int)}, ignore_extra_keys=True)
    args = typecheck.validate(arguments)

    slr_name   = 'SLR-{}_{}'.format(args['<slr>'], time.strftime('%m-%d'))
    params = {'days' : args['<kpers>'], 'slr' : args['<slr>'], 'name': slr_name,
              'ss'   : args['--steady'],
              'path_exe' : op.join('/', 'opt', 'local', 'bin')}

    if args['--steady']:
        main_SS(args['<path>'], args['--verbose'], **params)
    else:
        main_TRS(args['<path>'], args['--verbose'], **params)
