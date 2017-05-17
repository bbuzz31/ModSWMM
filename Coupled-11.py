#!/usr/bin/env python
"""
Run SWMM and MODFLOW2005 sequentially

Usage:
  Coupled-11.py KPERS [options]

Examples:
  Change **params dictionary within script.
  Run a year coupled simulation
  ./Coupled-11.py 550

Arguments:
  kpers       Total number of MODFLOW Steps / SWMM Days

Options:
  -c, --coupled=BOOL  Run a Coupled Simulation | SWMM input file    [default: 1]
  -d, --dev=BOOL      Run a Coupled Sim for 0.0 only, dont move dir [default: 0]
  -h, --help          Print this message

Purpose:
    Create SWMM Input or
    Run MODFLOW and SWMM, sequentially

Notes:
  ** Set SLR runs, parameters and path(if necessary) within script **
  If --coupled is false only creates SWMM inp
  Data created with data_prep.py

  Area   : West Neck Creek, Virginia Beach
  Created: 2017-1-15
  Updated: 2017-05-12
  Author : Brett Buzzanga

"""

from components import wncNWT, wncSWMM, bcpl, swmmSOL as swmm, 
from utils import PickleRaw, PickleFmt

import BB
import os
import os.path as op
import time
from datetime import datetime, timedelta
from collections import OrderedDict
import re
import shutil
import numpy as np
import pandas as pd

from subprocess import call
from multiprocessing import Pool

from docopt import docopt
from schema import Schema, Use, Or

class InitSim(object):
    """
    Initialize Simulation
    Remove Old Control Files
    Run MF SS
    Create SWMM Input File
    """
    def __init__(self, slr, days, ext='_1'):
        self.slr        = slr
        self.days       = days
        self.verbose    = 4
        self.coupled    = True
        self.slr_name   = 'SLR-{}_{}'.format(self.slr, time.strftime('%m-%d'))
        self.mf_parms   = self.mf_params()
        self.swmm_parms = self.swmm_params()
        self.path_child = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled',
                                time.strftime('%b')+ext, 'Child_{}'.format(slr))
        self.path_data  = op.join(self.path_child, 'Data')
        self.path_res   = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-17')
        self.start      = time.time()

    def mf_params(self):
        MF_params  = OrderedDict([
                    ('slr', self.slr), ('name' , self.slr_name),
                    ('days' , self.days), ('START_DATE', '06/29/2011'),
                    ('path_exe', op.join('/', 'opt', 'local', 'bin')),
                    ('ss', False), ('ss_rate', 0.0002783601), ('strt', 1),
                    ('extdp', 3.0), ('extwc', 0.101), ('eps', 3.75),
                    ('thts', 0.3), ('sy', 0.25), ('vks', 0.12), ('surf', 0.3048),
                    ('noleak', False),
                    ('coupled', self.coupled), ('Verbose', self.verbose)
                    ])
        return MF_params

    def swmm_params(self):
        # to adjust rain/temp or perviousness change the actual csv files
        SWMM_params = OrderedDict([
               #OPTIONS
               ('START_DATE', self.mf_parms['START_DATE']),
               ('name', self.slr_name), ('days', self.days), ('diff', 0.05),
               # SUBCATCHMENTS
               ('Width', 200),
               # SUBAREAS
               ('N-Imperv', 0.011), ('N-Perv', 0.015),
               ('S-Imperv', 0.05), ('S-Perv', 2.54),
               # INFILTRATION
               ('MaxRate', 50), ('MaxInfil', 0),# ('MinRate', 0.635),
               ('Decay', 4), ('DryTime', 7),
               # AQUIFER
               ('Por', self.mf_parms['thts']), ('WP', self.mf_parms['extwc']- 0.001),
               ('FC', self.mf_parms['sy']), ('Ksat' , self.mf_parms['vks']),
               ('Kslope', 25), ('Tslope', 0.00),
               ('ETu', 0.50), ('Ets', self.mf_parms['extdp']),
               ('Seep', 0), ('Ebot' ,  0), ('Egw', 0),
               ### GROUNDWATER
               ('Node', 13326),
               ('a1' , 0.0000), ('b1', 0), ('a2', 0), ('b2', 0), ('a3', 0),
               ('Dsw', 0), ('Ebot', 0),
               ### JUNCTIONS
               # note elevation and maxdepth maybe updated and overwrittten
               ('Elevation', ''), ('MaxDepth', ''), ('InitDepth', 0),
               ('Aponded', 40000), ('surf', self.mf_parms['surf']*0.5),
               ### STORAGE
               # ksat = steady state infiltration
               ('A1', 0), ('A2', 0), ('A0', 40000),
               ### LINKS
               ('Roughness', 0.02), ('InOffset', 0), ('OutOffset' , 0),
               ### TRANSECTS (shape)
               ('Depth', 2.0), ('WIDTH', 10),
               ('Side1', 0.5), ('Side2', 0.5), ('Barrels', 1),
               ### INFLOWS (slr)
               ('slr', self.slr),
               ('coupled', self.coupled), ('Verbose', self.verbose)
               ])

        start_date  = datetime.strptime(self.mf_parms['START_DATE'], '%m/%d/%Y')
        SWMM_params['END_DATE'] = datetime.strftime(start_date + timedelta(
                                             days=self.days+1), '%m/%d/%Y')

        return SWMM_params

    def init(self):
        """ Remove old control files / create SWMM input file """
        self._fmt_dir(uzfb=False)
        #
        wncNWT.main_SS(self.path_child, **self.mf_parms)
        wncSWMM.main(self.path_child, **self.swmm_parms)

    def _fmt_dir(self, uzfb):
        """ Prepare Directory Structure """
        if not op.isdir(self.path_child):
            os.makedirs(self.path_child)
        if not op.isdir(op.join(self.path_child, 'MF')):
            os.makedirs(op.join(self.path_child, 'MF'))
        if not op.isdir(op.join(self.path_child, 'SWMM')):
            os.makedirs(op.join(self.path_child, 'SWMM'))
        if not op.isdir(self.path_data):
            data_dir = op.join(op.split(op.split(self.path_child)[0])[0], 'Data')
            os.symlink(data_dir, self.path_data)

        ### Remove files that control flow / uzf gage files """
        remove_start=['swmm', 'mf']
        regex = re.compile('({}.uzf[0-9]+)'.format(self.slr_name[-5:]))
        [os.remove(op.join(self.path_child,f)) for f in os.listdir(self.path_child)
                                         for j in remove_start if f.startswith(j)]

        if op.isdir(op.join(self.path_child, 'MF', 'ext')):
            shutil.rmtree(op.join(self.path_child, 'MF', 'ext'))
        if uzfb:
            [os.remove(op.join(self.path_child, 'MF', f)) for f in
                os.listdir(op.join(self.path_child, 'MF')) if regex.search(f)]

class RunSim(object):
    """ Run Coupled MODSWMM Simulation """
    def __init__(self, initobj):
        self.init       = initobj
        self.swmm_steps = 24
        self.df_subs    = pd.read_csv(op.join(self.init.path_data, 'SWMM_subs.csv'),
                                                             index_col='Zone')
        self.nrows      = int(max(self.df_subs.ROW))
        self.ncols      = int(max(self.df_subs.COLUMN))

        self.stors      = [11965, 11966, 11970, 12022]
        os.chdir(self.init.path_child)

    def run_coupled(self):
        """ Run MF and SWMM together """
        ## start MODFLOW
        wncNWT.main_TRS(self.init.path_child, **self.init.mf_parms)

        time.sleep(1) # simply to let modflow finish printing to screen first
        STEPS_mf  = self.init.days+1
        path_root = self.init.path_child
        v         = self.init.swmm_parms.get('Verbose', 4)
        slr_name  = self.init.slr_name

        # storage and outfalls within study area  -- ONE INDEXED
        sub_ids      = self.df_subs.index.tolist()
        NAME_swmm    = '{}.inp'.format(slr_name)

        swmm.initialize(op.join(self.init.path_child, 'SWMM', NAME_swmm))

        # idea is run day 0 mf steady state, then day 1 of swmm; 1 of MF, 2 SWMM
        for STEP_mf in range(1, self.init.days+1):
            last_step = True if STEP_mf == (STEPS_mf - 1) else False

            if not last_step:
                self._debug('new', STEP_mf, v)
                self._debug('from_swmm', STEP_mf, v)
            else:
                self._debug('last', STEP_mf, v)
            ### run and pull from swmm
            self._debug('swmm_run', STEP_mf, v)
            for_mf       = bcpl.swmm_get_all(self.swmm_steps, sub_ids, last_step,
                                             self.stors, self.nrows*self.ncols)

            if last_step == True:    break
            ### overwrite new MF arrays
            finf      = for_mf[:,0].reshape(self.nrows, self.ncols)
            pet       = for_mf[:,1].reshape(self.nrows, self.ncols)
            ext_dir   = op.join(self.init.path_child, 'MF', 'ext')
            bcpl.write_array(ext_dir, 'finf_', STEP_mf+1, finf)
            bcpl.write_array(ext_dir, 'pet_' , STEP_mf+1, pet)
            self._debug('for_mf', STEP_mf, v, path_root=self.init.path_child)
            bcpl.swmm_is_done(path_root, STEP_mf)

            ### MF step is runnning
            self._debug('mf_run', STEP_mf, v)
            bcpl.mf_is_done(self.init.path_child, STEP_mf, v=v)
            self._debug('for_swmm', STEP_mf, v)

            ### get SWMM values for new step from uzfb and fhd
            mf_step_all = bcpl.mf_get_all(self.init.path_child, STEP_mf, **self.init.swmm_parms)
            self._debug('set_swmm', STEP_mf, v)

            # set MF values to SWMM
            bcpl.swmm_set_all(mf_step_all)
            # wait = True
            # while wait:
                # time.sleep(10)

        errors = swmm.finish()
        self._debug('swmm_done', STEP_mf, True, errors)
        return

    def _debug(self, steps, tstep, verb_level, errs=[], path_root=''):
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
            with open(op.join(self.init.path_child, 'Coupled_{}.log'.format(
                        time.strftime('%m-%d'))), 'a') as fh:
                        [fh.write('{}\n'.format(line)) for line in message]

        if verb_level == 4 and 'swmm_run' in steps or 'mf_run' in steps:
            print '\n'.join(message)

        return message

class FinishSim(object):
    """ Write Log, Move to External HD, Write Pickles, Backup Pickles """
    def __init__(self, init):
        self.init    = init
        self.date    = self.init.slr_name.split('_')[1]

    def log(self):
        """ Write Elapsed time and Parameters to Log File """
        # def _log(path_root, slr_name,  elapsed='', params=''):
        path_log = op.join(self.init.path_child, 'Coupled_{}.log'.format(self.date))
        opts     = ['START_DATE', 'name', 'days', 'END_DATE']
        headers  = {'Width': 'Subcatchments:', 'N-Imperv': 'SubAreas:',
                   'MaxRate' : 'Infiltration:', 'Por' : 'Aquifer:',
                   'Node' : 'Groundwater:', 'InitDepth' : 'Junctions:',
                   'Elevation': 'Outfalls:', 'ELEvation' : 'Storage:',
                   'Roughness' : 'Links:', 'Depth' : 'Transects:'}

        with open (path_log, 'a') as fh:
            fh.write('{} started at: {} {}\n'.format(self.init.slr_name, self.date,
                                                     time.strftime('%H:%m:%S')))
            fh.write('MODFLOW Parameters: \n')
            [fh.write('\t{} : {}\n'.format(key, value)) for key, value in
                                                    self.init.mf_parms.items()]
            fh.write('SWMM Parameters: \n')
            fh.write('\tOPTIONS\n')
            [fh.write('\t\t{} : {}\n'.format(key, value)) for key, value in
                                          self.init.swmm_parms.items() if key in opts]

            for key, value in self.init.swmm_parms.items():
                if key in opts: continue;
                if key in headers.keys():
                    fh.write('\t{}\n'.format(headers[key]))
                fh.write('\t\t{} : {}\n'.format(key, value))

            end     = time.time()
            elapsed = round((end - self.init.start)/60., 2)
            fh.write('\n{} finished in : {} min\n'.format(self.init.slr_name, elapsed))

    def store_results(self):
        ### MF
        mf_dir    = op.join(self.init.path_child, 'MF')
        cur_files = [op.join(mf_dir, mf_file) for mf_file in os.listdir(mf_dir)
                        if self.init.slr_name in mf_file]
        ext_dir   = op.join(mf_dir, 'ext')
        dest_dir  = op.join(self.init.path_res, self.init.slr_name)

        if op.isdir(dest_dir):
            dest_dir = op.join(dest_dir, 'temp')
            os.makedirs(dest_dir)
        else:
            os.makedirs(dest_dir)

        [shutil.copy(mf_file, op.join(dest_dir, op.basename(mf_file))) for
                                                mf_file in cur_files]
        try:
            shutil.copytree(ext_dir, op.join(dest_dir, 'ext'))
            [os.remove(mf_file) for mf_file in cur_files]
            shutil.rmtree(ext_dir)
        except:
            print 'ext directory not copied or removed'

        ### SWMM
        swmm_dir  = op.join(self.init.path_child, 'SWMM')
        cur_files = [op.join(swmm_dir, swmm_file) for swmm_file in
                           os.listdir(swmm_dir) if self.init.slr_name in swmm_file]
        [shutil.copy(swmm_file, op.join(dest_dir, op.basename(swmm_file)))
                                                  for swmm_file in cur_files]
        [os.remove(cur_file) for cur_file in cur_files]

        ### LOG
        log       = op.join(self.init.path_child, 'Coupled_{}.log'.format(self.date))
        if op.exists(log):
            shutil.copy (log, op.join(dest_dir, op.basename(log)))
            os.remove(log)
        print 'Results moved to {}\n'.format(dest_dir)

    def pickles(self):
        call(['PickleRaw.py', self.path_res]) # DONT use rel path (os.getcwd())

        print '\nFormatting Data ...\n'
        picklefmt.main(PATH_result)

        print '\nBacking up to Time Capsule ...\n'
        call(['TC_backup.py', self.path_res])

def main(args):
    """ Run MODSWMM """
    slr, days = args
    InitObj   = InitSim(slr, days)
    # run SS and create SWMM .inp
    InitObj.init()
    # start transient MF and SWMM sol
    RunSim(InitObj).run_coupled()
    # save log, move results, pickle?
    FinishSim(InitObj).log()
    FinishSim(InitObj).store_results()
    # FinishSim(slr, days).pickles() # needs to go outside of the loop?

if __name__ == '__main__':
    arguments = docopt(__doc__)
    typecheck = Schema({'KPERS'   : Use(int),  '--coupled' : Use(int),
                        '--dev' : Use(int)}, ignore_extra_keys=True)
    # this is the test
    args = typecheck.validate(arguments)
    SLR = [0.0, 1.0, 2.0]

    if args['--dev']:
        # short, 0.0 only simulation
        InitObj = InitSim(SLR[0], args['KPERS'], ext='_DEV')
        InitObj.init()
        RunSim(InitObj).run_coupled()

    elif args['--coupled']:
        # zip up args into lists for pool workers
        ARGS = zip(SLR, [args['KPERS']]*len(SLR))
        pool = Pool(processes=len(SLR))
        pool.map(main, ARGS)

    else:
        # run MF SS and create SWMM .INP
        InitSim(SLR[0], args['KPERS'], ext='_SS').init()
