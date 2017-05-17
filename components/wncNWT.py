#!/usr/bin/env python
"""
Run MODFLOW-NWT with FloPy Wrapper

Usage:
  wncNWT.py PATH KPERS [options]

Examples:
  Run a SS MF-ONLY model with 1 stress periods
  wncNWT.py /Users/bb/Google_Drive/WNC/Coupled/May_SS 0

  Run a Transient MF model with 5 stress periods and 1m SLR
  wncNWT.py . 5 --slr 1

Arguments:
  PATH        Path .../Coupled/[month]
  KPERS       Total number of MODFLOW Steps

Options:
  -c, --coupled==BOOL Not Coupled/Coupled                           [default: 0]
  -s, --steady=BOOL   Transient/Steady                              [default: 1]
      --slr==FLOAT    Amount (m) to add to CHD (already @ 1 ft)    [default: 0.0]
  -v, --verbose=INT   Print MODFLOW messages(0:4)                   [default: 4]
  -h, --help          Print this message

Notes:
  SS Only will run 2 steps always, which don't differ from one step
  SS Only will result in diff soil moisture than trs (trs used for coupled)
  Directory with csv files  must be at <path>/Data. Can be symlink.
  Data is prepared in dataprep.py
"""
import os
import os.path as op
import time
import numpy   as np
import pandas  as pd

from subprocess import Popen
import flopy.modflow as flomf

from docopt import docopt
from schema import Schema, Use, Or

class WNC_Base(object):
    def __init__(self, path_root, **params):
        """ Initialize with path_root where 'MF' directory will be made """
        self.path      = path_root
        self.path_data = op.join(path_root, 'Data')
        self.params    = params
        self.kpers     = self.params.get('days', 5)
        _              = self.load_data()
        __             = self.init_mf()

    def load_data(self):
        """ load data from csvs """
        # check here and one directory up
        if not op.isdir(self.path_data):
            self.path_data = op.join(op.dirname(self.path), 'Data')
            if not op.isdir(self.path_data):
                raise OSError('Data Directory not Found')

        self.df_mf  = pd.read_csv(op.join(self.path_data, 'MF_GRID.csv'), index_col='FID')
        self.df_chd = pd.read_csv(op.join(self.path_data, 'MF_CHD.csv'), index_col='Zone')
        # for uzf gages
        self.df_subs = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'), index_col='Zone')

    def init_mf(self):
        """ Initialize MODFLOW object """
        NAME_mf  = '{}'.format(self.params.get('name', 'MISSING'))
        MODEL    = op.join(self.path, 'MF', NAME_mf)
        path_exe = self.params.get('path_exe', op.join('/', 'opt', 'local', 'bin'))

        if self.params['coupled']:
            self.mf = flomf.Modflow(MODEL, exe_name=op.join(path_exe, 'NWT_BB'),
                                        version='mfnwt', silent=False, verbose=False,
                                  external_path=op.join(self.path, 'MF', 'ext'))
        else:
            self.mf = flomf.Modflow(MODEL, exe_name=op.join(path_exe, 'mfnwt'),
                                        version='mfnwt',
                                  external_path=op.join(self.path, 'MF', 'ext'))

        if self.params['ss']:
            self.steady      = [True]  * self.kpers
        else:
            self.steady      = [False] * self.kpers
            self.steady[0]   = True

    def run(self):
        self.mf.write_name_file()
        quiet = self.params.get('Verbose', 4)
        if self.params.get('coupled'):
            Popen([self.mf.exe_name, self.mf.namefile])
        else:
            if quiet == 1 or quiet == 3:
                quiet = False
            else:
                quiet = True
            ### silent doesn't work; i think flopy problem
            success, buff = self.mf.run_model(silent=True)
            if success and quiet:
                print '\n  ************************', \
                      '\n  MODFLOW (flopy) is done.'
                if self.kpers ==  2:
                    print '  Creating new SWMM .inp from MF SS gages'
                elif self.kpers >  2 and not self.params.get('ss'):
                    trans = [step for step in self.steady if step is False]
                    print '  Ran', str(len(trans)), 'transient steps.'
                print '  ************************'
            return success

    def _chd_format(self, chd, layers=[0]):
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

    def _uzf_gage_format(self, listoflists):
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

class WNC_Inps(WNC_Base):
    """ Class for making WNC input files """
    def __init__(self, path_data, **params):
        WNC_Base.__init__(self, path_data, **params)
        self.nrows    = int(max(self.df_mf.ROW));
        self.ncols    = int(max(self.df_mf.COLUMN_));
        self.last_col = len(self.df_mf.columns)
        self.nlays    = len(self.df_mf.filter(like='KX').columns)

    def nwt(self):
        nwt = flomf.ModflowNwt(self.mf, iprnwt=1)
        return nwt

    def dis(self):
        delrc    = 200
        ztop     = self.df_mf['MODEL_TOP'].values.reshape(self.nrows, self.ncols);
        botms    = self.df_mf.filter(like='BOT').T.values.reshape(self.nlays, self.nrows, self.ncols)

        # prj = NAD_1983_2011_UTM_Zone_18N
        dis      = flomf.ModflowDis(self.mf, self.nlays, self.nrows, self.ncols,
                                    delr=delrc, delc=delrc, botm=botms,
                                    top=ztop, nper=self.kpers,
                                    steady=self.steady, rotation=19,
                                    xul=400232.64777, yul=4072436.50165,
                                    proj4_str='EPSG:6347',
                                    start_datetime=self.params.get('START_DATE', '06/30/2011'))

        return dis

    def bas(self):
        """ self.mf must be initialized and discretized to right shape """
        active = self.df_mf['IBND'].values.reshape(self.nrows, self.ncols)
        ibound = np.zeros((self.nlays, self.nrows, self.ncols), dtype=np.int32) + active
        bas    = flomf.ModflowBas(self.mf, ibound=ibound,
                                  strt=self.params.get('strt', 1), ichflg=True)
        return bas

    def upw(self):
        hk  = self.df_mf.filter(like='KX').T.values.reshape(self.nlays, self.nrows, self.ncols)
        vk  = self.df_mf.filter(like='KZ').T.values.reshape(self.nlays, self.nrows, self.ncols)
        upw = flomf.ModflowUpw(self.mf, laytyp=np.ones(self.nlays),
                                   layavg=np.ones(self.nlays)*2, hk=hk, vka=vk,
                                   sy=self.params.get('sy', 0.25))
        return upw

    def uzf(self):
        uzfbnd        = self.df_mf['UZF_IBND'].values.reshape(self.nrows, self.ncols)
        # list of 2 dimensional array; see finf array in uzf example notebook github
        Finf          = [self.df_mf['FINF'].values.reshape(self.nrows, self.ncols)]
        Pet           = [self.df_mf['ET'].values.reshape(self.nrows, self.ncols)]
        ss_rate       = self.params.get('ss_rate', 0.0002783601)
        # extend list to num of timesteps; overwritten in coupled simulation
        [Finf.append(Finf[0]) for   i in range (self.kpers-1)]
        [Pet.append(Pet[0])   for   i in range (self.kpers-1)]

        ### Gage Settings
        sub_id_mf   = [sub_id - 10000 for sub_id in self.df_subs.index.tolist()]
        listofgages = self.df_mf.iloc[:,2:4][self.df_mf.OBJECTID.isin(sub_id_mf)].values
        gage_info   = self._uzf_gage_format(listofgages)
        uzf         = flomf.ModflowUzf1(self.mf, iuzfopt=1, ietflg=1, nuztop=1,
                                        iuzfbnd=uzfbnd, iuzfcb2=61, ntrail2=15,
                                        eps=self.params.get('eps', 3.75),
                                        vks=self.params.get('vks', 0.12),
                                        thts=self.params.get('thts', 0.3),
                                        surfdep=self.params.get('surf', 0.3048),
                                        nosurfleak=self.params.get('noleak', False),
                                        extdp=self.params.get('extdp', 3.0),
                                        extwc=self.params.get('extwc', 0.101),
                                        nuzgag=len(gage_info), uzgag=gage_info,
                                        finf=Finf, pet=Pet)
        return uzf

    def chd(self):
        chead    = self.df_chd[['ROW', 'COLUMN_', 'START', 'END']]
        # SLR ; df has 0.3048 stored as 0m rise
        chead['START'] += self.params.get('slr', 0)
        chead['END']   += self.params.get('slr', 0)
        chd_lst  = [list(row) for row in chead.values]
        chd_info = self._chd_format(chd_lst, layers=range(self.nlays))
        chd      = flomf.ModflowChd(self.mf, stress_period_data=chd_info,
                                        options=['AUXILIARY IFACE NOPRINT MXACTC'])
        return chd

    def oc(self):
        # May have to alter spd[0] (stress period) to print other results.
        spd    = {(0, 0): ['save head', 'save drawdown', 'save budget']}
        ext    = ['oc', 'fhd', 'ddn', 'cbc']
        #units  = [14, 51, 57, 63, 65]
        # for formatted head file
        ihedfm = 0 #10
        fmt    = '(1X1PE13.5)'
        oc     = flomf.ModflowOc(self.mf, stress_period_data=spd, compact=True, #unitnumber=units,
                                 ihedfm=ihedfm, chedfm=fmt, cddnfm=fmt, extension=ext)
        return oc

    def pcg(self):
        pcg = flomf.ModflowPcg(self.mf, mxiter=60, iter1=20, hclose=0.25, rclose=0.25);
        return pcg

    def hob(self):
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
        tmax = self.mf.dis.get_final_totim()
        hob_data = []
        for i, ob in enumerate(hobs):
            data = np.array([tmax, ob])
            hob_data.append(flomf.HeadObservation(self.mf, row=rows[i], column=columns[i],
                                    roff=roffs[i], coff=coffs[i],
                                              obsname=obs[i], time_series_data=data))

        hob = flomf.ModflowHob(self.mf, iuhobsv=59, hobdry=-9999, obs_data=hob_data)
        return hob

    def write(self):
        """ Write MODFLOW inp files. Remove checks. """
        # refactor with getattr
        inp_files = []
        inp_files.append(self.nwt())
        inp_files.append(self.dis())
        inp_files.append(self.bas())
        inp_files.append(self.upw())
        inp_files.append(self.uzf())
        inp_files.append(self.chd())
        inp_files.append(self.oc())
        inp_files.append(self.hob())

        [mf_file.write_file() for mf_file in inp_files]
        [os.remove(f) for f in os.listdir(os.getcwd()) if f.endswith('chk') ]

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

def main_SS(path_root, **params):
    """ Run a SS simulation for generating SWMM inp (initial heads/wc) """
    params['days']    = 2
    params['coupled'] = False

    WNC_model = WNC_Inps(path_root, **params)
    WNC_model.write()
    WNC_model.run()

def main_TRS(path_root, **params):
    """ Run a Transient simulation, MODFLOW only """
    if not params.get('days'):
        raise ValueError('Need to specify days in Params')

    params['ss'] = False
    if not params.get('coupled'):
        params['coupled'] = False

    WNC_model = WNC_Inps(path_root, **params)
    WNC_model.write()
    WNC_model.run()

if __name__ == '__main__':
    arguments = docopt(__doc__)
    typecheck = Schema({'KPERS'     : Use(int),  'PATH'      : os.path.exists,
                        '--coupled' : Use(int),  '--slr'     : Use(float),
                        '--steady'  : Use(int),  '--verbose' : Use(int)},
                        ignore_extra_keys=True)

    args      = typecheck.validate(arguments)
    PATH      = op.abspath(args['PATH'])
    slr_name  = 'SLR-{}_{}'.format(args['--slr'], time.strftime('%m-%d'))

    params    = {'name'    : slr_name,            'days' : args['KPERS'],
                 'coupled' : args['--coupled'],   'ss'   : args['--steady'],
                 'slr'     : args['--slr'],     'noleak' : 0,
                 'Verbose': args['--verbose']}

    if args['--steady']:
        main_SS(PATH, **params)
    else:
        main_TRS(PATH, **params)
