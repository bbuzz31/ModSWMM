"""
Class/Functions necessary for running coupled MODFLOW/SWMM
Brett Buzzanga, 2017
"""
from __future__ import print_function
import os
import os.path as op
import time
import re
import linecache
import numpy as np
import pandas as pd

import flopy.utils.formattedfile as ff
import flopy.utils.binaryfile as bf
import swmmSOL as swmm

class StepDone(object):
    def __init__(self, path_child='', kper=-1, v=4):
        self.path_child = path_child
        self.kper       = kper
        self.v          = v

    def swmm_is_done(self):
        """
        Creates an empty file that fortran needs before beginnnig uzf calcs.
        Should be called AFTER new FINF and PET arrays are written to disc.
        Format needs to match fortran.
        """
        swmm_done  = 'swmm_done{:05d}.txt'.format(self.kper)
        open (op.join(self.path_child, swmm_done), 'w').close()

        message = ['SWMM step done for next MODFLOW Trans: {}'.format(self.kper)]
        if self.v == 1 or self.v == 3:
            print ('\n'.join(message))
        if self.v == 2 or self.v == 3:
            with open(op.join(self.path_child, 'Coupled_{}.log'.format(
                                    op.basename(self.path_child))), 'a') as fh:
                fh.write('\n'.join(message))

    def mf_is_done(self):
        """ Check for mf_doneXYZ.txt, wait till exists / exit if nonconverge """
        mf_pass = op.join(self.path_child, 'mf_done{:05d}.txt'.format(self.kper))
        mf_fail = op.join(self.path_child, 'mf_fail{:05d}.txt'.format(self.kper))

        delay = 0.5
        found = False
        while found == False:
            if op.isfile(mf_fail):
                swmm.finish()
                print ('\n  ##################################################',
                      '\n  MODFLOW {} FAILED to converge at Trans Day: {}'.format(
                                  os.path.basename(self.path_child), self.kper),
                      '\n  ################################################## \n')
                raise SystemExit('---Stopping SWMM---')
            elif op.isfile(mf_pass):
                found = True
            else:
                time.sleep(delay)
        if self.v > 0 and self.v < 4:
            if self.kper > 0:
                print ('  MODFLOW Trans step', str(self.mf_pass),'done.\n')
            else:
                print ('  MODFLOW Steady State Done.\n')

    def swmm_set(self, data4swmm):
        """
        Purpose:
            set head/theta/leakage calculated in MF to SWMM
        Args:
            data4swmm: arr size len(subcatchs) x 6
                       cols: row, col, head, theta, leak, index (index never used)
        Return:
            Nothing
        Notes:
            Wrapper around swmm.setGW
        """
        for i, cell in enumerate(data4swmm[:, 5]):
            loc = str(int(cell))
            swmm.setGW(loc, swmm.HEAD,  swmm.SI, data4swmm[i,2])
            swmm.setGW(loc, swmm.THETA, swmm.SI, data4swmm[i,3])
            swmm.setGW(loc, swmm.LEAK,  swmm.SI, data4swmm[i,4])
            # if loc == '10004':
                # print data4swmm[i, 3]
        return

    def mf_set(self, data, var):
        """
        Purpose: Write array for MODFLOW to disk.
        Arguments:
            data = infiltration or evaporation rates (numpy 2d grid)
            var  = file name root (finf, pet)
        Note:
            kper is increased by TWO in calling function
        """
        ext_dir  = op.join(self.path_child, 'MF', 'ext')
        ext_file = op.join(ext_dir, '{}_{}.ref'.format(var, self.kper))
        fmt      = '%15.6E'
        with open(ext_file, 'w') as fh:
            np.savetxt(fh, data, fmt=fmt, delimiter='')

def mf_get_all(path_root, mf_step, **params):
    """
    Purpose:
        Get head from .fhd (need head even when water is discharging to land)
        Get soil water from uzf gage files.
        Soil water is average over whole unsaturated zone (or just top).
    Args:
        path_root: filepath where MF dir is with .uzfbs and .fhd
        mf_step: timestep to use
    Returns:
        pd.DataFrame
    Notes:
        Set soil water when water at land surface to porosity
                    Need to set to something.
    """

    mf_model   = os.path.join(path_root, 'MF', params.get('name'))

    try:
        hds    = ff.FormattedHeadFile(mf_model + '.fhd', precision='double')
    except:
        raise SystemError(mf_model+'.fhd does not exist.\nCheck paths')
    try:
        uzf    = bf.CellBudgetFile(mf_model + '.uzfcb2.bin', precision='single')
    except:
        uzf    = bf.CellBudgetFile(mf_model + '.uzfcb2.bin', precision='double')

    head_fhd   = hds.get_data(totim=mf_step+1, mflay=0)
    uzf_data   = abs(uzf.get_data(text='SURFACE LEAKAGE', totim=mf_step+1)[0])

    arr_surf   = np.load(op.join(op.dirname(op.dirname(path_root)),
                                'Data', 'Land_Z.npy')).reshape(head_fhd.shape)

    # intialize numpy arrays that will get updated based on row/col location
    index         = np.linspace(10001, 10000 + head_fhd.size, head_fhd.size, dtype=int)
    theta_uzfb    = np.empty(head_fhd.shape)
    theta_uzfb[:] = np.nan
    row_np        = np.empty(head_fhd.shape)
    row_np[:]     = np.nan
    col_np        = np.empty(head_fhd.shape)
    col_np[:]     = np.nan
    regex = re.compile('({}.uzf[0-9]+)'.format(params.get('name')[-5:]))
    gage_files = [os.path.join(os.path.dirname(mf_model), uzfile) for uzfile
                  in os.listdir(os.path.dirname(mf_model))
                  if regex.search(uzfile)]
    # convert time step to guess at line where it starts in uzfb file
    # gage_files = [test for test in gage_files if test.endswith('2008')]# or test.endswith('2004')]
    for i, each in enumerate(gage_files):
        linecache.clearcache()
        nums = re.findall(r'\d+', linecache.getline(each, 1))
        # convert row/col to 0 index
        row = int(nums[1]) - 1
        col = int(nums[2]) - 1

        # store row/col in np array for table
        row_np[row][col] = row
        col_np[row][col] = col
        line_start = mf_step * 40 + 4
        looking = True
        # begin looking for correct time step
        while looking:
            # line with time; blank when surface leakage - cause time to be wrong
            header = linecache.getline(each, line_start).split()
            # mf_step + 1, skip steady state (0)
            try:
                if int(float(header[1])) == mf_step + 1:
                    theta = 0
                    # first depth, for checking coupling -- or maybe use
                    for x in range(line_start, line_start+1):
                        # average:
                        # for x in range(line_start, line_start+40):
                            # theta += float(linecache.getline(each, x).split()[-1])/40
                        theta_uzfb[row][col] = theta
                    looking = False

                elif int(float(header[1])) < mf_step + 1:
                    theta_uzfb[row][col] = params.get('Por')
                    head_fhd[row][col]   = arr_surf[row][col] - params.get('diff')
                    linecache.clearcache()
                    looking = False

                else:
                    raise TypeError('How is mf_step + 1 less than header?')

            except:
                line_start -= 40
                linecache.clearcache()
                if line_start < 4:
                    # print (row,col)
                    theta_uzfb[row][col] = params.get('Por', 0.3)
                    head_fhd[row][col]   = arr_surf[row][col] - params.get('diff')
                    looking = False

    row_np    = row_np + 1
    col_np    = col_np + 1
    # stack 1ds into multidimensional nd array for pandas columns to be correct
    to_stack  = [row_np, col_np, head_fhd, theta_uzfb, uzf_data]
    unraveled = [each.ravel() for each in to_stack]
    stacked   = np.column_stack(unraveled)
    unraveled.append(index)
    stacked   = np.column_stack(unraveled)

    # drop non subcatchments (where row is nan)
    mf_subs    = stacked[~np.isnan(stacked[:,0])]

    return mf_subs

def swmm_get_all(cells, gridsize):
    """
    Purpose:
        Run SWMM get functions to obtain values for next MF step
            enables smaller time steps than modflow
    Args:
        cells: list of cells to pull (get from SWMM_subs_new.csv)
        gridsize: length of modflow grid; for reshaping purposes

    Returns:
        Dataframe of index cells with the SUM of steps for each category
    """
    steps    = 24 # hours
    stors    = [11965, 11966, 11970, 12022]
    elapsed  = []
    mat_swmm = np.zeros([gridsize, 2], dtype=np.float32)
    for timesteps in range(0,steps):
        if swmm.is_over():
            print ('\n   ### ERROR: SWMM has less steps than MF  ### \n')
            break

        elapsed.append (swmm.get_time())
        swmm.run_step()
        # skip if MF is done
        for cell in cells:
            # convert to 0 index
            pos = cell - 10000 - 1
            ''' THIS IS CHANGED '''
            mat_swmm[pos, 0]   += swmm.get(str(cell), swmm.INFIL, swmm.SI)
            mat_swmm[pos, 1]   += swmm.get(str(cell), swmm.GW_ET, swmm.SI)
        # set storage units to steady state rate
        for unit in stors:
            mat_swmm[unit-10001, 0]  += 0.0115983375
            mat_swmm[unit-10001, 1]  += 0.0115983375

    # convert mm to m; hours to day accounted for by the summing CORRECT: 4/5/17
    mat_swmm *= 0.001
    return mat_swmm

def cell_num(row, col, n_cols=51, show=False):
    """
    Give 0 indexed row / col.
    Use row and column to find flattened MODFLOW number
    Searches row wise, n_row is how many rows in grid.
    """
    row += 1
    col += 1
    if row == 1:
        cellnum = col
    if row >= 2:
        cellnum = n_cols * (row-1) + col
    if show:
        print ('1D Loc = {}'.format(cellnum + 10000))
    return cellnum
