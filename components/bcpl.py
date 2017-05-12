"""
Functions necessary for running coupled MODFLOW/SWMM
Brett Buzzanga, 2017
"""

import BB

import os
import time
import re
import linecache
import numpy as np
import pandas as pd

import flopy.utils.formattedfile as ff
import swmmSOL as swmm

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
        print '1D Loc = {}'.format(cellnum + 10000)
    return cellnum

def mf_is_done(f_path, kper, f_n_root='mf_done', delay=0.5, ext='.txt', v=0):
    """ Check for file (mf_done), wait till there / exit if nonconvergence"""

    pass_fname = os.path.join(f_path, f_n_root + '{:05d}'.format(kper) + ext)
    fail_fname = os.path.join(f_path, 'mf_fail' + '{:05d}'.format(kper) + ext)

    found = False
    while found == False:
        if os.path.isfile(fail_fname):
            swmm.finish()
            print '\n  #############################################', \
                  '\n  MODFLOW {} FAILED to converge at Trans Day: {}'.format(os.path.basename(f_path), kper), \
                  '\n  #############################################\n'
            raise SystemExit('---Stopping SWMM---')
        elif os.path.isfile(pass_fname):
            found = True
        else:
            time.sleep(delay)
    if v > 0 and v < 4:
        if kper > 0:
            print '  MODFLOW Trans step', str(kper),'done.\n'
        else:
            print '  MODFLOW Steady State Done.\n'
    #return found

def mf_get_all(path_root, mf_step, **params):
    """
    Purpose:
        Get head from .fhd (need head even when water is discharging to land)
        Get soil water from uzf gage files.
        Soil water is average over whole unsaturated zone.
    Args:
        mf_model: filepath where .uzfbs and .fhd is located
        mf_step: timestep to use
    Returns:
        pd.DataFrame

    Notes:
        Set soil water when water at land surface to porosity?????????
            Need to set to something.
    """
    # get heads @ time step from .fhd : totim = 1 index; len includes CHD & stor

    mf_model   = os.path.join(path_root, 'MF', params.get('name'))
    # mf_model   = os.path.join(path_root, slr_name)
    # path_cup = os.path.join('/', 'Users', 'bb', 'Google_Drive','WNC', 'Coupled', time.strftime('%b'))
    try:
        hds    = ff.FormattedHeadFile(mf_model + '.fhd', precision='double')
    except:
        raise SystemError(mf_model+'.fhd does not exist.\nCheck paths')
    head_fhd      = hds.get_data(totim=mf_step+1, mflay=0)
    arr_surf      = np.load(os.path.join(path_root, 'Data', 'Land_Z.npy')).reshape(head_fhd.shape)
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
                        #for x in range(line_start, line_start+40):
                        theta += float(linecache.getline(each, x).split()[-1])#/40
                        theta_uzfb[row][col] = theta
                    looking = False

                elif int(float(header[1])) < mf_step + 1:
                    theta_uzfb[row][col] = params.get('Por')
                    head_fhd[row][col]   = arr_surf[row][col] - 0.05
                    linecache.clearcache()
                    looking = False

                else:
                    raise TypeError('How is mf_step + 1 less than header?')

            except:
                line_start -= 40
                linecache.clearcache()
                if line_start < 4:
                    # print (row,col)
                    theta_uzfb[row][col] = params.get('Por')
                    head_fhd[row][col]   = arr_surf[row][col] - 0.05
                    looking = False

    row_np    = row_np + 1
    col_np    = col_np + 1
    # stack 1ds into multidimensional nd array for pandas columns to be correct
    to_stack  = [row_np, col_np, head_fhd, theta_uzfb]
    unraveled = [each.ravel() for each in to_stack]
    stacked   = np.column_stack(unraveled)
    unraveled.append(index)
    stacked   = np.column_stack(unraveled)

    # drop non subcatchments (where row is nan)
    mf_subs    = stacked[~np.isnan(stacked[:,0])]

    # DEPRECATED; make dataframe
    # mf_all    = pd.DataFrame(stacked, columns=['ROW', 'COL', 'HEAD', 'THETA'])
    # mf_subs   = mf_all.dropna(subset=['ROW', 'COL'])
    # mf_subs.index += 10001
    return mf_subs

def swmm_get_all(steps, cells, mf_done, constants, mf_len=3774):
    """
    Purpose:
        Run SWMM get functions to obtain values for next MF step
            enables smaller time steps than modflow
    Args:
        steps: number of swmm steps to run; depends on MF (25 if running MF daily)
        cells: list of cells to pull (get from SWMM_subs_new.csv)
        mf_done: boolean, skips 'gets' if MODFLOW has finished
        stor_unit: storage units get constant infiltration
        mf_len: length of modflow grid; for reshaping purposes

    Returns:
        Dataframe of index cells with the SUM of steps for each category
    """

    elapsed = []
    mat_swmm = np.zeros([mf_len, 2], dtype=np.float32)
    for timesteps in range(0,steps):
        if swmm.is_over():
            print '\n   ### ERROR: SWMM has less steps than MF  ### \n'
            break

        elapsed.append (swmm.get_time())
        swmm.run_step()
        # skip if MF is done
        if not mf_done:
            for cell in cells:
                # convert to 0 index
                pos = cell - 10000 - 1
                ''' THIS IS CHANGED '''
                mat_swmm[pos, 0]   += swmm.get(str(cell), swmm.INFIL, swmm.SI)
                mat_swmm[pos, 1]   += swmm.get(str(cell), swmm.GW_ET, swmm.SI)
            # set storage units to steady state rate
            for unit in constants:
                mat_swmm[unit-10001, 0]  += 0.0115983375
                mat_swmm[unit-10001, 1]  += 0.0115983375
        else:
            print '  Writing SWMM .rpt '
            return None

    # convert mm to m; hours to day accounted for by the summing CORRECT: 4/5/17
    mat_swmm *= 0.001
    #print '    ### WARNING: storage units ET may be wrong ### \n'
    return mat_swmm

def swmm_is_done(f_path, kper, f_n_root='swmm_done', ext='.txt', verb_lvl=0):
    """
    Creates an empty file that fortran needs before beginnnig uzf calcs.
    Should be called AFTER new FINF and PET arrays are written to disc.
    Format needs to match fortran.
    """
    f_name = '{}{:05d}{}'.format(f_n_root, kper, ext)
    open (os.path.join(f_path, f_name), 'w').close()

    message = ['SWMM steps done for next MODFLOW Trans:'.format(kper)]
    if verb_lvl == 1 or verb_lvl == 3:
        print '\n'.join(message)
    if verb_lvl >= 2:
        with open(op.join(path_root, 'Coupled_{}.log'.format(time.strftime('%m-%d'))), 'a') as fh:
            fh.write('\n'.join(message))
    return True

def swmm_set_all(data4swmm):
    """
    Purpose:
        set head/theta calculated in MF to SWMM
    Args:
        data4swmm: pd.Dfwith cols HEAD & THETA; correct idx (from mf_get_all)
    Return:
        Nothing
    Notes:
        Just a wrapper around swmm.setGW

    """
    ### convert dataframe to matrix for faster accessing

    for i, cell in enumerate(data4swmm[:, 4]):
        loc = str(int(cell))
        swmm.setGW(loc, swmm.HEAD,  swmm.SI, data4swmm[i,2])
        swmm.setGW(loc, swmm.THETA, swmm.SI, data4swmm[i,3])
        # if loc == '10004':
            # print data4swmm[i, 3]
    return

def write_array(f_path, f_n_root, kper, data, ext='.ref', fmt='%15.6E'):
    """
    Purpose: Write array for MODFLOW to disk.
    Arguments:
        f_path = path
        f_n_root = file name root
        kper = time step (2 will be added to align with swmm)
        data = infiltration or evaporation rates (numpy 2d grid)
        ext = file extension, default = '.ref'
        fmt = number format
    """
    f_name = os.path.join(f_path, '{}{}{}'.format(f_n_root, kper, ext))
    with open(f_name, 'w') as fh:
        np.savetxt(fh, data, fmt=fmt, delimiter='')
