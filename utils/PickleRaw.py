#!/usr/bin/env python
"""
Save/load pickles from Coupled Runs

Usage:
  Pickles.py PATH [options]

Examples:
  Create all:
  Pickles.py /Volumes/BB_4TB/Thesis/Results_03-19

  Create matrices of swmm heads (edit heads in script)
  Pickles.py . --sub

Arguments:
  PATH  Directory with list of Model Runs (starts with Results_)

Options:
  --fmt    Run Formatting Script                                    [default: 0]
  --sub    Make Matrixes of a SWMM Variable
  --help   Print this message

Notes:
  Created: 2017-03-06
  Update: 2017-05-03
"""

import BB
import os
import os.path as op
import shutil

import time
import linecache
from collections import OrderedDict
import numpy as np
import pandas as pd

import components.bcpl
import PickleFmt, swmmtoolbox as swmtbx
import flopy.utils.formattedfile as ff
import flopy.utils.binaryfile as bf
#
from docopt import docopt
from schema import Schema, Use

from multiprocessing import Pool

class pickle_base(object):
    def __init__(self, path_result):
        self.path       = path_result
        self.path_picks = self._pick_dir()
        self.scenarios  = [op.join(self.path, slr) for slr in
                        os.listdir(op.join(self.path)) if slr.startswith('SLR')]
        self.slr        = [op.basename(scenario).split('_')[0][-3:] for
                                         scenario in self.scenarios]
        # self.slr_names?
        self.st_end     = self._st_end()
        self.ts_hr      = pd.date_range(self.st_end[0], self.st_end[1], freq='H')
        self.ts_day     = pd.date_range(self.st_end[0], self.st_end[1], freq='D')

    def _pick_dir(self, verbose=0):
        """ Make Pickles Directory if it doesn't exist """
        path_pickle = op.join(self.path, 'Pickles')
        try:
            os.makedirs(path_pickle)
        except:
            if verbose:
                print 'WARNING - Overwriting exiting pickles in: \n\t{}'.format(path_pickle)

        return path_pickle

    def _st_end(self):
        """ Pull start/end time from any .out file """
        slr_name   = 'SLR-{}_{}'.format(self.slr[0], self.scenarios[0][-5:])
        out_file   = op.join(self.path, slr_name, slr_name + '.out')
        st_end     = swmtbx.SwmmExtract(out_file).GetDates() # returns a tuple
        return st_end

class pickle_swmm(pickle_base):
    def __init__(self, path_result):
        pickle_base.__init__(self, path_result)

    def sys_out(self):
        """
        Create a dataframe with SWMM out variables
        Careful with times: SWMM stops @ 1 hour of new day(12-30)-00:00); remove
            doesn't update last day because no precip i guess
        There should be a date given in path_result
        """
        # precip, pet, flood, inf, runoff, evap
        varnames  = ['Precip', 'Pet', 'Flood', 'Vol_Stored', 'Infil', 'Runoff', 'Surf_Evap']
        variables = [1, 14, 10, 12, 3, 4, 13]
        sys_mat   = np.zeros([len(self.ts_hr), len(varnames) * len(self.scenarios)])
        colnames  = []

        for i, scenario in enumerate(self.scenarios):
            slr_name = op.basename(scenario)
            slr      = slr_name[4:7]
            out_file = op.join(scenario, '{}.out'.format(slr_name))

            colnames.extend(['{}_{}'.format(var_name, slr) for var_name in varnames])
            for j,v in enumerate(variables):
                sys_mat[:, j+i*len(variables)] = swmtbx.extract_arr(out_file,
                                                 'system,{},{}'.format(v,v))
        swmm_sys = pd.DataFrame(sys_mat, index=self.ts_hr, columns=colnames)
        path_res = op.join(self.path_picks, 'swmm_sys.df')
        swmm_sys.to_pickle(path_res)
        print 'DataFrame pickled to: {}'.format(path_res)

    def _sub_heads(self, scenario):
        """
        All Subcatchments, All Times. Just Head.
        Pickle a npy for each scenario separately.
        Based on Pickles_orig().subs_rungw
        """
        varnames  = ['Head']
        variables = [6]

        slr_name  = op.basename(scenario)
        slr       = slr_name[4:7]
        out_file  = op.join(scenario, '{}.out'.format(slr_name))
        sub_names = [int(name) for name in swmtbx.listdetail(out_file,'subcatchment')]
        sys_mat   = np.zeros([len(self.ts_hr), len(sub_names)*len(varnames)])

        for i, sub in enumerate(sub_names):
            for j, var in enumerate(variables):
                sys_mat[:, j+i*len(variables)] = swmtbx.extract_arr(out_file,
                                                'subcatchment,{},{}'.format(sub,var))

        path_arr  = op.join(self.path_picks, 'swmm_heads_{}.npy'.format(slr))
        np.save(pickle_path, sys_mat)

    def _sub_run(self, scenario):
        """
        All Subcatchments, All Times. Just Runoff.
        Pickle a npy for each scenario separately.
        Based on subs_rungw
        """
        varnames  = ['Run']
        variables = [4]

        slr_name  = op.basename(scenario)
        slr       = slr_name[4:7]
        out_file  = op.join(scenario, '{}.out'.format(slr_name))
        sub_names = [int(name) for name in swmtbx.listdetail(out_file,'subcatchment')]
        sys_mat   = np.zeros([len(self.ts_hr), len(sub_names)*len(varnames)])

        for i, sub in enumerate(sub_names):
            for j, var in enumerate(variables):
                sys_mat[:, j+i*len(variables)] = swmtbx.extract_arr(out_file,
                                                'subcatchment,{},{}'.format(sub,var))

        path_arr = op.join(self.path_picks, 'swmm_run_{}.npy'.format(slr))
        np.save(pickle_path, sys_mat)

class pickle_uzf(pickle_base):
    def __init__(self, path_result):
        pickle_base.__init__(self, path_result)

    def uzf_arrays(self):
        """
        Make 3d numpy arrays of shape (74*51*549)
        """
        varnames      = ['surf_leak', 'uzf_rch', 'uzf_et', 'uzf_run']
        variables     = ['SURFACE LEAKAGE', 'UZF RECHARGE', 'GW ET', 'HORT+DUNN']
        for scenario in self.scenarios:
            slr_name  = op.basename(scenario)
            slr       = slr_name[4:7]
            uzf_file  = op.join(scenario, '{}.uzfcb2.bin'.format(slr_name))
            try:
                uzfobj = bf.CellBudgetFile(uzf_file, precision='single')
            except:
                uzfobj = bf.CellBudgetFile(uzf_file, precision='double')
            for i, variable in enumerate(variables):
                uzf_data      = uzfobj.get_data(text=variable)
                sys_mat       = np.zeros([len(self.ts_day), 74, 51])
                for j in range(len(self.ts_day)):
                    sys_mat[j,:,:] = uzf_data[j]
                # save separately so can load separately and faster
                path_res = op.join(self.path_picks, '{}_{}.npy'.format(varnames[i], slr))
                np.save(path_res, sys_mat)
        print 'UZF arrays pickled to to: {}'.format(self.path_picks)

class pickle_ext(pickle_base):
    def __init__(self, path_result):
        pickle_base.__init__(self, path_result)

    def ts_sums(self):
        varnames      = ['FINF', 'GW_ET']
        variables     = ['finf', 'pet']
        sys_mat       = np.zeros([len(self.ts_day), len(varnames) * len(self.scenarios)])
        colnames      = []
        for i, scenario in enumerate(self.scenarios):
            slr_name  = op.basename(scenario)
            slr       = slr_name[4:7]
            ext_dir   = op.join(self.path, scenario, 'ext')
            colnames.extend(['{}_{}'.format(var_name, slr) for var_name in varnames])
            for j in range(1, len(self.ts_day)+1):
                for k, v in enumerate(variables):
                    v_file = op.join(ext_dir, '{}_{}.ref'.format(v, j))
                    var = np.fromfile(v_file, sep= ' ')
                    sys_mat[j-1, k+i*len(varnames)] = var.reshape(74, 51).sum()

        ext_sys  = pd.DataFrame(sys_mat, index=self.ts_day, columns=colnames)
        path_res = op.join(self.path_picks, 'ext_sums.df')
        ext_sys.to_pickle(path_res)
        print 'DataFrame pickled to: {}'.format(path_res)

### multiprocessing cannot use class methods
def _ts_heads(args):
    """ Pull heads from fhd file in parallel """
    scenario, path_pickle = args
    slr_name  = op.basename(scenario)
    slr       = slr_name[4:7]
    head_file = op.join(scenario, op.basename(scenario) + '.fhd')
    try:
        hds  = ff.FormattedHeadFile(head_file, precision='single')
    except:
        hds  = ff.FormattedHeadFile(head_file, precision='double')
    heads    = hds.get_alldata(mflay=0)
    res_path = op.join(path_pickle, 'heads_{}.npy'.format(slr))
    np.save(res_path, heads)
    # print 'Np array pickled to to: {}'.format(res_path)

def _sub_var(args):
    """
    All Subcatchments, All Times. One Variable.
    Pickle a npy for each scenario separately.
    Based on subs_rungw
    """
    param_map = {'inf' : 3, 'run' : 4, 'gwout' : 5, 'heads' : 6, 'soil' : 7}
    scenario, varname, ts, path_pickle = args

    # varnames  = [varname]
    variables = [param_map[varname]]

    slr_name  = op.basename(scenario)
    slr       = slr_name[4:7]
    out_file  = op.join(scenario, '{}.out'.format(slr_name))
    sub_names = [int(name) for name in swmtbx.listdetail(out_file,'subcatchment')]
    sys_mat   = np.zeros([len(ts), len(sub_names)*len(variables)])

    for i, sub in enumerate(sub_names):
        for j, var in enumerate(variables):
            sys_mat[:, j+i*len(variables)] = swmtbx.extract_arr(out_file,
                                            'subcatchment,{},{}'.format(sub,var))

    path_arr = op.join(path_pickle, 'swmm_{}_{}.npy'.format(varname, slr))
    np.save(path_arr, sys_mat)

def main(path_result):
    swmm_obj = pickle_swmm(path_result)
    swmm_obj.sys_out()
    pickle_uzf(path_result).uzf_arrays()
    pickle_ext(path_result).ts_sums()
    return swmm_obj

if __name__ == '__main__':
    start       = time.time()
    arguments   = docopt(__doc__)
    typecheck   = Schema({'PATH' : os.path.exists, '--fmt' : Use(int)},
                          ignore_extra_keys=True)
    PATH_result = op.abspath(typecheck.validate(arguments)['PATH'])
    # remove undefined items from dictionary
    args = {k:v for k, v in arguments.iteritems() if v}
    # this may introduce a bug
    args = typecheck.validate(args)

    ### 1 CPU
    swmm_obj              = main(PATH_result)
    scenarios, path_picks = swmm_obj.scenarios, swmm_obj.path_picks
    ts_hr                 = swmm_obj.ts_hr

    ### Multiprocessing
    if len(args) == 1:
        print 'Pickling FHD heads to: {}'.format(path_picks)
        pool = Pool(processes=len(scenarios))
        res = pool.map(_ts_heads, zip(scenarios, [path_picks] * len(scenarios)))

        print 'Pickling SWMM Heads to {} ... '.format(path_picks)
        pool = Pool(processes=len(scenarios))
        res = pool.map(_sub_var, zip(scenarios, ['heads']*len(scenarios),
                               [ts_hr]*len(scenarios), [path_picks]*len(scenarios)))

        print 'Pickling SWMM Runoff to {} ... '.format(path_picks)
        pool = Pool(processes=len(scenarios))
        res = pool.map(_sub_var, zip(scenarios, ['run']*len(scenarios),
                               [ts_hr]*len(scenarios), [path_picks]*len(scenarios)))

        end = time.time()
        print 'Pickles made in ~ {} min'.format(round((end-start)/60., 2))

    else:
        print 'Pickling SWMM Soil to {} ... '.format(path_picks)
        pool = Pool(processes=len(scenarios))
        res = pool.map(_sub_var, zip(scenarios, ['soil']*len(scenarios),
                               [ts_hr]*len(scenarios), [path_picks]*len(scenarios)))
    if args['--fmt']:
        picklefmt.main()
