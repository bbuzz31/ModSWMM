import BB
import os
import os.path as op
import time
from datetime import datetime, timedelta
from collections import OrderedDict
import re
import shutil

from components import wncNWT, wncSWMM, swmm, bcpl
import numpy as np
import pandas as pd

from subprocess import call


class InitSim(object):
    """ booking functions: _fmt, _debug, _log, _store_results """
    def __init__(self, days=5, slr=0.0, ext=''):
        self.days       = days
        self.slr        = slr
        self.slr_name   = 'SLR-{}_{}'.format(self.slr, time.strftime('%m-%d'))
        self.coupled    = True
        self.verbose    = 4
        self.mf_parms   = self.mf_params()
        self.swmm_parms = self.swmm_params()
        self.path_child = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled',
                                time.strftime('%b')+ext, 'Child_{}'.format(slr))
        self.path_data  = op.join(self.path_child, 'Data')
        self.path_res   = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-11')
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
               ('name', self.slr_name), ('days', self.days),
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
               ('a1' , 0.00001), ('b1', 0), ('a2', 0), ('b2', 0), ('a3', 0),
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

class RunSim(InitSim):
    """ Run Coupled MODSWMM Simulation """
    def __init__(self):
        InitSim.__init__(self)
        _ = self.init()
        self.swmm_steps = 24
        self.df_subs    = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'),
                                                             index_col='Zone')
        self.nrows      = int(max(self.df_subs.ROW))
        self.ncols      = int(max(self.df_subs.COLUMN))

        self.stors      = [11965, 11966, 11970, 12022]
        os.chdir(self.path_child)

        ### Run
        wncNWT.main_TRS(self.path_child, **self.mf_parms)
        self.run_coupled()

    def run_coupled(self):
        """ Run MF and SWMM together """
        time.sleep(1) # simply to let modflow finish printing to screen first
        STEPS_mf = self.days+1
        path_root = self.path_child
        v  = self.swmm_parms.get('Verbose', 4)
        slr_name  = self.swmm_parms.get('name')


        # storage and outfalls within study area  -- ONE INDEXED
        sub_ids      = self.df_subs.index.tolist()
        NAME_swmm    = '{}.inp'.format(slr_name)

        swmm.initialize(op.join(self.path_child, 'SWMM', NAME_swmm))

        # idea is run day 0 mf steady state, then day 1 of swmm; 1 of MF, 2 SWMM
        for STEP_mf in range(1, self.days+1):
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
            bcpl.write_array(op.join(self.path_child, 'MF', 'ext'), 'finf_', STEP_mf+1, finf)
            bcpl.write_array(op.join(self.path_child, 'MF', 'ext'), 'pet_' , STEP_mf+1, pet)
            self._debug('for_mf', STEP_mf, v, path_root=self.path_child)
            bcpl.swmm_is_done(path_root, STEP_mf)

            ### MF step is runnning
            self._debug('mf_run', STEP_mf, v)
            bcpl.mf_is_done(self.path_child, STEP_mf, v=v)
            self._debug('for_swmm', STEP_mf, v)

            ### get SWMM values for new step from uzfb and fhd
            mf_step_all = bcpl.mf_get_all(self.path_child, STEP_mf, **self.swmm_parms)
            self._debug('set_swmm', STEP_mf, v)

            # set MF values to SWMM
            bcpl.swmm_set_all(mf_step_all)

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
            with open(op.join(path_root, 'Coupled_{}.log'.format(time.strftime('%m-%d'))), 'a') as fh:
                    [fh.write('{}\n'.format(line)) for line in message]

        if verb_level == 4 and 'swmm_run' in steps or 'mf_run' in steps:
            print '\n'.join(message)

        return message

class FinishSim(InitSim):
    """ Write Log, Move to External HD, Write Pickles, Backup Pickles """
    def __init__(self):
        InitSim.__init__(self)
        self.cur_run = '{}'.format(self.slr_name)
        self.date    = self.slr_name.split('_')[1]
        self.end     = time.time()
        self.elapsed = round((self.end - self.start)/60., 2)
        self.log()
        self.store_results()
        self.pickles()

    def log(self):
        """ Write Elapsed time and Parameters to Log File """
        # def _log(path_root, slr_name,  elapsed='', params=''):
        path_log = op.join(self.path_child, 'Coupled_{}.log'.format(self.date))
        opts     = ['START_DATE', 'name', 'days', 'END_DATE']
        headers  = {'Width': 'Subcatchments:', 'N-Imperv': 'SubAreas:',
                   'MaxRate' : 'Infiltration:', 'Por' : 'Aquifer:',
                   'Node' : 'Groundwater:', 'InitDepth' : 'Junctions:',
                   'Elevation': 'Outfalls:', 'ELEvation' : 'Storage:',
                   'Roughness' : 'Links:', 'Depth' : 'Transects:'}

        with open (path_log, 'a') as fh:
            fh.write('{} started at: {} {}\n'.format(self.slr_name, self.start,
                                                     time.strftime('%H:%m:%S')))
            fh.write('MODFLOW Parameters: \n')
            [fh.write('\t{} : {}\n'.format(key, value)) for key, value in
                                                    self.mf_parms.items()]
            fh.write('SWMM Parameters: \n')
            fh.write('\tOPTIONS\n')
            [fh.write('\t\t{} : {}\n'.format(key, value)) for key, value in
                                          self.swmm_parms.items() if key in opts]

            for key, value in self.swmm_parms.items():
                if key in opts: continue;
                if key in headers.keys():
                    fh.write('\t{}\n'.format(headers[key]))
                fh.write('\t\t{} : {}\n'.format(key, value))

            fh.write('\n{} finished in : {} min\n'.format(self.slr_name, self.elapsed))

    def store_results(self):
        ### MF
        mf_dir    = op.join(self.path_child, 'MF')
        cur_files = [op.join(mf_dir, mf_file) for mf_file in os.listdir(mf_dir)
                        if self.cur_run in mf_file]
        ext_dir   = op.join(mf_dir, 'ext')
        dest_dir  = op.join(self.path_res, self.cur_run)

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
        swmm_dir  = op.join(self.path_child, 'SWMM')
        cur_files = [op.join(swmm_dir, swmm_file) for swmm_file in
                           os.listdir(swmm_dir) if self.cur_run in swmm_file]
        [shutil.copy(swmm_file, op.join(dest_dir, op.basename(swmm_file)))
                                                  for swmm_file in cur_files]
        [os.remove(cur_file) for cur_file in cur_files]

        ### LOG
        log       = op.join(self.path_child, 'Coupled_{}.log'.format(self.date))
        if op.exists(log):
            shutil.copy (log, op.join(dest_dir, op.basename(log)))
            os.remove(log)
        print 'Results moved to {}\n'.format(dest_dir)

    def pickles(self):
        call(['PickleRaw.py', self.path_res]) # DONT use rel path (os.getcwd())

# def main(Days, SLR, Coupled, V):
#     start      = time.time()
#     Child_dir  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled',
#                                     time.strftime('%b')+'_1', 'Child_{}'.format(SLR))
#     PATH_store = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-11')
#     Params     = _get_params(Days, SLR, Child_dir)
#     Params['MF']['coupled'] = Coupled
#     Params['MF']['Verbose'] = True
#     Params['SWMM']['Verbose'] = True
#     name       = Params['MF']['name']
#
#     ### remove old control files, uzf files, create directories
#     _fmt(Child_dir, name, uzfb=True)
#     os.chdir(Child_dir)
#
#     ### Create SWMM Input File
#     wncNWT.main_SS(Child_dir, **Params['MF'])
#     wncSWMM.main(Child_dir, **Params['SWMM'])
#
#     if Coupled:
#         ### MF
#         wncNWT.main_TRS(Child_dir, **Params['MF'])
#         ### SWMM
#         _run_coupled(Days+1, STEPS_swmm=24, slr_name=name, path_root=Child_dir, verbose=V, **Params['SWMM'])
#
#     End = time.time()
#     Elapsed = End-start
#     # write elapsed to log
#     _log(Child_dir, name, elapsed=Elapsed)
#
#     # move current run to Results dir
#     if Coupled:
#         ### write parameters to log
#         if SLR == 0.0:
#             _log(Child_dir, Params['MF']['name'], params=Params)
#         # _store_results(Child_dir, PATH_store, name)
#
# # def main_help(args):
# #     KPERS, SLR, COUPLED, VERBOSE = args
# #     main(KPERS, SLR, COUPLED, VERBOSE)
# #
# # if __name__ == '__main__':
# #     arguments = docopt(__doc__)
# #     typecheck = Schema({'<kpers>'  : Use(int),  '--coupled' : Use(int),
# #                        '--verbose' : Use(int)}, ignore_extra_keys=True)
# #     args = typecheck.validate(arguments)
# #     SLR  = np.linspace(0, 2.4, num=13, endpoint=True) if args['--coupled'] else [0.0]
# #
# #     #for all scenarios
# #     # SLR = [0.0, 1.0, 2.0]
# #     SLR = [0.0]
# #     # zip up args into lists for pool workers
# #     ARGS = zip([args['<kpers>']]*len(SLR), SLR, [args['--coupled']]*len(SLR),
# #                [args['--verbose']]*len(SLR))# child_dirs)
# #
# #
# #     # num of processes to do
# #     # pool = Pool(processes=len(SLR))
# #     # pool.map(main_help, ARGS)
# #
# #     # for debugging
# #     main(args['<kpers>'], SLR[0], args['--coupled'], args['--verbose'])
# #
# #     # make pickles
# #     # path_result = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-10')
# #     # call(['./components/PickleRaw.py', path_result])

RunSim()
FinishSim()
