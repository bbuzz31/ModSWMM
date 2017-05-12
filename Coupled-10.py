#!/usr/bin/env python
"""
Run SWMM and MODFLOW2005 sequentially

Usage:
  Coupled-8.py <kpers> [options]

Examples:
  Change **params dictionary within script.
  Run a year coupled simulation
  ./Coupled-8.py 366
  Create a swmm input file with 366 days
  ./Coupled-8.py 366 -c 0

Arguments:
  kpers       Total number of MODFLOW Steps / SWMM Days

Options:
  -c, --coupled=BOOL  Run a Coupled Simulation | SWMM input file   [default: 1]
  -v, --verbose=BOOL  Print Messages for debugging                 [default: 4]
  -h, --help          Print this message

Purpose:
    Create SWMM Input or
    Run MODFLOW and SWMM, sequentially

Notes:
  ** Set SLR runs, parameters and path(if necessary) within script **
  If --coupled is false only creates SWMM inp
  Data created with swmm_mf-prep.py

  Area   : West Neck Creek, Virginia Beach
  Created: 2017-1-15
  Updated: 2017-02-24
  Author : Brett Buzzanga

"""
from BB_PY import BB
from components import swmm, bcpl, wncNWT, wncSWMM
import os
import os.path as op
import shutil

import time
from datetime import datetime, timedelta
from subprocess import call, Popen
import numpy as np
import pandas as pd
import re

from docopt import docopt
from schema import Schema, Use, Or
from collections import OrderedDict

from multiprocessing import Pool

def _fmt(child_dir, slr_name='', uzfb=False):
    """ Prepare Directory Structure """
    if not op.isdir(child_dir):
        os.makedirs(child_dir)
    if not op.isdir(op.join(child_dir, 'MF')):
        os.makedirs(op.join(child_dir, 'MF'))
    if not op.isdir(op.join(child_dir, 'SWMM')):
        os.makedirs(op.join(child_dir, 'SWMM'))
    if not op.isdir(op.join(child_dir, 'Data')):
        data_dir = op.join(op.split(op.split(child_dir)[0])[0], 'Data')
        os.symlink(data_dir, op.join(child_dir, 'Data'))

    ### Remove files that control flow / uzf gage files """
    remove_start=['swmm', 'mf']
    regex = re.compile('({}.uzf[0-9]+)'.format(slr_name[-5:]))
    [os.remove(op.join(child_dir,f)) for f in os.listdir(child_dir) for j in remove_start if f.startswith(j)]
    if op.isdir(op.join(child_dir, 'MF', 'ext')):
        shutil.rmtree(op.join(child_dir, 'MF', 'ext'))
    if uzfb:
        [os.remove(op.join(child_dir, 'MF', f)) for f in os.listdir(op.join(child_dir, 'MF'))
                                                if regex.search(f)]

def _debug(steps, tstep, verb_level, errs=[], path_root=''):
    """
    Control Messages Printing.
    Verb_level 0 == silent
    Verb_level 1 == to console
    Verb_level 2 == to Coupled_today.log
    Verb_level 3 == to console and log
    Verb_level 4 == to console, make sure still working
    Errs = final SWMM Errors.
    """
    message=[]
    if not isinstance(steps, list):
        steps=[steps]
    if 'new' in steps:
        message.extend(['','  *************', '  NEW DAY: {}'.format(tstep), '  *************',''])
    if 'last' in steps:
        message.extend(['','  #############', '  LAST DAY: {}'.format(tstep), '  #############', ''])
    if 'from_swmm' in steps:
        message.extend(['Pulling Applied INF and GW ET', 'From SWMM: {} For MF-Trans: {}'.format(tstep,tstep), '             .....'])
    if 'for_mf' in steps:
        message.extend(['FINF & PET {} arrays written to: {}'.format(tstep+1, op.join(path_root, 'mf', 'ext'))])
    if 'mf_run' in steps:
        message.extend(['', 'Waiting for MODFLOW Day {} (Trans: {})'.format(tstep+1, tstep)])
    if 'mf_done' in steps:
        message.extend(['---MF has finished.---', '---Calculating final SWMM steps.---'])
    if 'for_swmm' in steps:
        message.extend(['Pulling head & soil from MF Trans: {} for SWMM day: {}'.format(tstep, tstep+1)])
    if 'set_swmm' in steps:
        message.extend(['', 'Setting SWMM values for new SWMM day: {}'.format(tstep + 1)])
    if 'swmm_done' in steps:
        message.extend(['', '  *** SIMULATION HAS FINISHED ***','  Runoff Error: {}'.format(errs[0]), '  Flow Routing Error: {}'.format(errs[1]), '  **************************************'])
    if 'swmm_run' in steps:
        message.extend(['', 'Waiting for SWMM Day: {}'.format(tstep+1)])

    if verb_level == 1 or verb_level == 3:
        print '\n'.join(message)
    if verb_level == 2 or verb_level == 3:
        with open(op.join(path_root, 'Coupled_{}.log'.format(time.strftime('%m-%d'))), 'a') as fh:
                [fh.write('{}\n'.format(line)) for line in message]

    if verb_level == 4 and 'swmm_run' in steps or 'mf_run' in steps:
        print '\n'.join(message)

    return message

def _log(path_root, slr_name,  elapsed='', params=''):
    path_log = op.join(path_root, 'Coupled_{}.log'.format(slr_name[-5:]))
    opts     = ['START_DATE', 'name', 'days', 'END_DATE']
    headers  = {'Width': 'Subcatchments:', 'N-Imperv': 'SubAreas:',
               'MaxRate' : 'Infiltration:', 'Por' : 'Aquifer:',
               'Node' : 'Groundwater:', 'InitDepth' : 'Junctions:',
               'Elevation': 'Outfalls:', 'ELEvation' : 'Storage:',
               'Roughness' : 'Links:', 'Depth' : 'Transects:'}

    with open (path_log, 'a') as fh:
        if params:
            fh.write('{} started at: {} {}\n'.format(slr_name, slr_name[-5:],
                                                     time.strftime('%H:%m:%S')))
            fh.write('MODFLOW Parameters: \n')
            [fh.write('\t{} : {}\n'.format(key, value)) for key, value in params['MF'].items()]
            fh.write('SWMM Parameters: \n')
            fh.write('\tOPTIONS\n')
            [fh.write('\t\t{} : {}\n'.format(key, value)) for key, value in
                                          params['SWMM'].items() if key in opts]

            for key, value in params['SWMM'].items():
                if key in opts: continue;
                if key in headers.keys():
                    fh.write('\t{}\n'.format(headers[key]))
                fh.write('\t\t{} : {}\n'.format(key, value))

        elif elapsed:
            fh.write('\n{} finished in ::: {} min\n'.format(slr_name,
                                                          round(elapsed/60, 2)))

def _store_results(path_root, path_store, slr_name):
    ### MF
    cur_run   = '{}'.format(slr_name)
    mf_dir    = op.join(path_root, 'MF')
    cur_files = [op.join(mf_dir, mf_file) for mf_file in os.listdir(mf_dir)
                    if cur_run in mf_file]
    ext_dir   = op.join(mf_dir, 'ext')
    dest_dir  = op.join(path_store, cur_run)
    try:
        os.makedirs(dest_dir)
    except:
        dest_dir = op.join(dest_dir, 'temp')
        os.makedirs(dest_dir)

    [shutil.copy(mf_file, op.join(dest_dir, op.basename(mf_file))) for mf_file in
              cur_files]
    try:
        shutil.copytree(ext_dir, op.join(dest_dir, 'ext'))
        [os.remove(mf_file) for mf_file in cur_files]
        shutil.rmtree(ext_dir)
    except:
        print 'ext dir not copied or removed'

    ### SWMM
    swmm_dir  = op.join(path_root, 'SWMM')
    cur_files = [op.join(swmm_dir, swmm_file) for swmm_file in os.listdir(swmm_dir) if cur_run in swmm_file]
    [shutil.copy(swmm_file, op.join(dest_dir, op.basename(swmm_file))) for swmm_file in cur_files]
    [os.remove(cur_file) for cur_file in cur_files]

    ### LOG
    log       = op.join(path_root, 'Coupled_{}.log'.format(slr_name[-5:]))
    if op.exists(log):
        shutil.copy (log, op.join(dest_dir, op.basename(log)))
        os.remove(log)
    print 'Results moved to {}\n'.format(dest_dir)

def _run_coupled(STEPS_mf, STEPS_swmm, path_root, verbose=1, **params):
    """ Run MF and SWMM together """
    time.sleep(1) # simply to let modflow finish printing to screen first

    # if using changing leak to storage, use this with wncSWMM-leakouts.py
    # df_subcatchs = pd.read_pickle(op.join(path_root, 'Data', '{}_subs.df'.format(slr_name)))
    # load the ss leaking and pits (pit probably would be in here if not edited)
    # stors        = (pd.read_pickle(op.join(path_root, 'Data', '{}_stors.df'.format(slr_name)))
    #                                     .index.tolist() + [11965, 11966])

    slr_name     = params.get('name')

    df_subcatchs = pd.read_csv(op.join(path_root, 'Data', 'SWMM_subs.csv'),
                                                           index_col='Zone')
    stors        = [11965, 11966, 11970, 12022]
    nrows        = int(max(df_subcatchs.ROW));
    ncols        = int(max(df_subcatchs.COLUMN));
    # storage and outfalls within study area  -- ONE INDEXED
    sub_ids      = df_subcatchs.index.tolist()
    NAME_swmm    = '{}.inp'.format(slr_name)
    swmm.initialize(op.join(path_root, 'SWMM', NAME_swmm))

    # idea is run day 0 mf steady state, then day 1 of swmm,
    # then day 1 of mf (stress 2), then day 2 of swmm
    for STEP_mf in range(1, STEPS_mf):
        last_step = True if STEP_mf == (STEPS_mf - 1) else False

        if not last_step:
            _debug('new', STEP_mf, verbose)
            _debug('from_swmm', STEP_mf, verbose)
        else:
            _debug('last', STEP_mf, verbose)

        ### run and pull from swmm
        _debug('swmm_run', STEP_mf, verbose)
        for_mf       = bcpl.swmm_get_all(STEPS_swmm, sub_ids, last_step, stors,
                                         nrows*ncols)

        if last_step == True:    break
        ### overwrite new MF arrays
        finf      = for_mf[:,0].reshape(nrows, ncols)
        pet       = for_mf[:,1].reshape(nrows, ncols)
        bcpl.write_array(op.join(path_root, 'MF', 'ext'), 'finf_', STEP_mf+1, finf)
        bcpl.write_array(op.join(path_root, 'MF', 'ext'), 'pet_' , STEP_mf+1, pet)
        _debug('for_mf', STEP_mf, verbose, path_root=path_root)
        bcpl.swmm_is_done(path_root, STEP_mf)

        ### MF step is runnning
        _debug('mf_run', STEP_mf, verbose)
        bcpl.mf_is_done(path_root, STEP_mf, v=verbose)
        _debug('for_swmm', STEP_mf, verbose)

        ### get SWMM values for new step from uzfb and fhd
        mf_step_all = bcpl.mf_get_all(path_root, STEP_mf, **params)
        _debug('set_swmm', STEP_mf, verbose)

        # set MF values to SWMM
        bcpl.swmm_set_all(mf_step_all)

    errors = swmm.finish()
    _debug('swmm_done', STEP_mf, True, errors)
    return

def _get_params(days, slr, path_root=False):

    slr_name   = 'SLR-{}_{}'.format(slr, time.strftime('%m-%d'))

    MF_params  = OrderedDict([
                ('slr', slr), ('name' , slr_name), ('days' , days),
                ('path_exe', op.join('/', 'opt', 'local', 'bin')),
                ('ss', False), ('ss_rate', 0.0002783601), ('strt', 1),
                ('extdp', 3.0), ('extwc', 0.101), ('eps', 3.75),
                ('thts', 0.3), ('sy', 0.25), ('vks', 0.18), ('surf', 0.3048),
                ('noleak', False),
                ])

    # to adjust rain/temp or perviousness change the actual csv files
    SWMM_params = OrderedDict([
               #OPTIONS
               ('START_DATE', '06/29/2011'), ('name', slr_name), ('days', days),
            #    ('START_DATE', '12/29/2011'), ('name', slr_name), ('days', days),
               # SUBCATCHMENTS
               ('Width', 200),
               # SUBAREAS
               ('N-Imperv', 0.011), ('N-Perv', 0.015),
               ('S-Imperv', 0.05), ('S-Perv', 2.54),
               # INFILTRATION
               ('MaxRate', 50), ('MaxInfil', 0),# ('MinRate', 0.635),
               ('Decay', 4), ('DryTime', 7),
               # AQUIFER
               ('Por', MF_params['thts']), ('WP', MF_params['extwc']- 0.001),
               ('FC', MF_params['sy']), ('Ksat' , MF_params['vks']),
               ('Kslope', 25), ('Tslope', 0.00),
               ('ETu', 0.50), ('Ets', MF_params['extdp']),
               ('Seep', 0), ('Ebot' ,  0), ('Egw', 0),
               #('Umc', MF_params['extwc']- 0.001), ('ETupat', ''),
               ### GROUNDWATER
               ('Node', 13326),
               ('a1' , 0.00001), ('b1', 0), ('a2', 0), ('b2', 0), ('a3', 0),
               ('Dsw', 0), ('Ebot', 0),
               ### JUNCTIONS
               # note elevation and maxdepth maybe updated and overwrittten
               ('Elevation', ''), ('MaxDepth', ''), ('InitDepth', 0),
               ('Aponded', 40000), ('surf', MF_params['surf']*0.5),
               ### OUTFALLS
               #('ELEvation', 0),
               ### STORAGE
               # ksat = steady state infiltration
               #('eleVATION', -3), ('MAXDepth', 4), ('INITDepth', 1),
               #('lakes', [11970, 12022]), ('KSAT', 0.0114),
               ('A1', 0), ('A2', 0), ('A0', 40000),
               ### LINKS
               ('Roughness', 0.02), ('InOffset', 0), ('OutOffset' , 0),
               ### TRANSECTS (shape)
               ('Depth', 2.0), ('WIDTH', 10),
               ('Side1', 0.5), ('Side2', 0.5), ('Barrels', 1),
               ### INFLOWS (slr)
               ('slr', slr)
               ])

    start_date  = datetime.strptime(SWMM_params['START_DATE'], '%m/%d/%Y')
    SWMM_params['END_DATE'] = datetime.strftime(start_date + timedelta(
                                         days=days+1), '%m/%d/%Y')

    Params = {'MF': MF_params, 'SWMM' : SWMM_params}

    return Params

def main(Days, SLR, Coupled, V):
    start      = time.time()
    Child_dir  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled',
                                    time.strftime('%b')+'_2', 'Child_{}'.format(SLR))
    PATH_store = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-10')
    Params     = _get_params(Days, SLR, Child_dir)
    name       = Params['MF']['name']

    ### remove old control files, uzf files, create directories
    _fmt(Child_dir, name, uzfb=True)
    os.chdir(Child_dir)

    ### Create SWMM Input File
    wncNWT.main_SS(Child_dir, quiet=V, **Params['MF'])
    wncSWMM.main(Child_dir, verbose=V, **Params['SWMM'])

    if Coupled:
        ### MF
        mf_inits   = wncNWT.mf_init(Child_dir, Coupled, **Params['MF'])
        flopy_objs = wncNWT.mf_inps(mf_inits, **Params['MF'])
        wncNWT.write(flopy_objs)
        wncNWT.mf_run(mf_inits, quiet=V)
        ### SWMM
        _run_coupled(Days+1, STEPS_swmm=24, slr_name=name, path_root=Child_dir, verbose=V, **Params['SWMM'])

    End = time.time()
    Elapsed = End-start
    # write elapsed to log
    _log(Child_dir, name, elapsed=Elapsed)

    # move current run to Results dir
    if Coupled:
        ### write parameters to log
        if SLR == 0.0:
            _log(Child_dir, Params['MF']['name'], params=Params)
        _store_results(Child_dir, PATH_store, name)

def main_help(args):
    KPERS, SLR, COUPLED, VERBOSE = args
    main(KPERS, SLR, COUPLED, VERBOSE)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    typecheck = Schema({'<kpers>'  : Use(int),  '--coupled' : Use(int),
                       '--verbose' : Use(int)}, ignore_extra_keys=True)
    args = typecheck.validate(arguments)
    SLR  = np.linspace(0, 2.4, num=13, endpoint=True) if args['--coupled'] else [0.0]

    #for all scenarios
    SLR = [0.0, 1.0, 2.0]
    # SLR = [0.0]
    # zip up args into lists for pool workers
    ARGS = zip([args['<kpers>']]*len(SLR), SLR, [args['--coupled']]*len(SLR),
               [args['--verbose']]*len(SLR))# child_dirs)


    # num of processes to do
    pool = Pool(processes=len(SLR))
    pool.map(main_help, ARGS)

    # for debugging
    # main(args['<kpers>'], SLR[0], args['--coupled'], args['--verbose'])

    # make pickles
    # path_result = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-10')
    # call(['PickleRaw.py', path_result])
