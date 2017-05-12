"""
AnalysisObjs2.py
Refactored 05-05
"""
from BB_PY import BB

import os
import os.path as op
from datetime import datetime, timedelta
import re
import linecache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import flopy.utils.formattedfile as ff
import flopy.utils.binaryfile as bf

from components import bcpl, swmmtoolbox_BB as swmtbx

class res_base(object):
    def __init__(self, path_result):
        self.path       = path_result
        self.path_picks = op.join(path_result, 'Pickles')
        self.path_data  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC',
                                                           'Coupled', 'Data')

        self.df_sys     = pd.read_pickle(op.join(self.path_picks, 'swmm_sys.df'))#.loc['2012-01-01-00':, :]
        self.slr        = BB.uniq([float(slr.rsplit('_', 1)[1]) for slr
                                                        in self.df_sys.columns])
        self.sys_vars   = BB.uniq([slr.rsplit('_', 1)[0] for slr in self.df_sys.columns])

        self.ts_day     = self.df_sys.resample('D').first().index
        self.ts_hr      = self.df_sys.index
        self.df_swmm    = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'),
                                                           index_col='Zone')
        self.subs       = self.df_swmm.index.values
        self.fig_path   = op.join(self.path, 'Figures')
        self.slr_sh     = ['0.0', '1.0', '2.0']
        self.seasons    = ['Winter', 'Spring', 'Summer', 'Fall']


    def ts_all(self, param, loc=False, slr=0.0, dates=[0, -0], plot=False):
        """
        Pull a time series from a swmm binary '.out' file
        Arguments:
            Path to .out
            Param (eg 'precip')
            Loc ID number (eg 10735)
            Dates can be integers or datetimes
            Plot True if you want to see plots
        Returns:
            Pandas dataframe with datetime index
        """
        name = 'SLR-{}_{}'.format(slr, op.basename(self.path).split('_')[1])
        # print name

        out_file = op.join(self.path, name, '{}.out'.format(name))
        # map parameters to type, individual code, units (for plot), system code
        param_map = {'precip' : ['subcatchment,', ',0', 'mm', ',1'],
                     'flow'   : ['link,', ',2', 'm/s'],
                     'flood'  : ['node,', ',5', 'cms', ',10'],
                     'inf'    : ['subcatchment,', ',3', 'mm/hr?', ',3'],
                     'evap'   : ['subcatchment,', ',2', 'mm/hr?', ',13'],
                     'run'    : ['subcatchment,', ',4', 'cms', ',4'],
                     'head'   : ['subcatchment,', ',6', 'm'],
                     'soil'   : ['subcatchment,', ',7', 'theta'],
                     'gwout'  : ['subcatchment,', ',5', '??', '6?'],

                     'pet'    : ['subcatchment,', '', 'mm/hr?', ',14'],

                    }
        #swmtbx.listdetail(out_file, 'link')
        if loc:
            t_series_raw = swmtbx.extract(out_file, param_map[param][0] + str(loc)
                                                         + param_map[param][1])
        else:
            t_series_raw = swmtbx.extract(out_file, 'system' + param_map[param][3]
                                                         + param_map[param][3])
        t_series = t_series_raw[dates[0]:]

        if plot:
            t_series.resample('MS').mean().plot()
            plt.ylabel (param_map[param][2])

        return t_series

    def ts_gage(self, row, col):
        """ Pull ts from a uzf gage file. Give 0 idx, it uses 1 idx row/col """
        ### find correct uzf file
        gage_ext = op.splitext(self._find_gage(row, col))[1]
        df_all   = pd.DataFrame(index=self.ts_day)
        for slr in self.slr:
            slr_name  = 'SLR-{}_{}'.format(slr, op.basename(self.path).split('_')[1])
            path_gage = op.join(self.path, slr_name, slr_name+gage_ext)
            ### pull data from file
            data = []
            with open(path_gage) as input:
                for i in range(3):
                    next(input)
                for line in input:
                    line = line.strip().split()
                    if len(line) == 6:
                        layer = int(line.pop(0))
                        time = float(line.pop(0))
                        head = float(line.pop(0))
                        uzthick = float(line.pop(0))
                    depth = float(line.pop(0))
                    watercontent = float(line.pop(0))
                    # data.append([layer, time, head, uzthick, depth, watercontent])
                    data.append([time, head, watercontent])
            # df_wc = pd.DataFrame(data, columns=['layer', 'time', 'head', 'uzthick',
                                                        #    'depth', 'watercontent'])

            cols  = ['{}_{}'.format(var, slr) for var in ['Time', 'Head', 'Water']]
            df_wc = pd.DataFrame(data, columns=cols)
            # start_date = datetime(2011, 06, 30)

            df_wc = df_wc.groupby('Time_{}'.format(slr)).first().reset_index()

            df_wc.index = df_wc['Time_{}'.format(slr)].apply(lambda x:
                                           self.ts_day[0] + timedelta(days=x-1))
            df_wc.drop('Time_{}'.format(slr), 1, inplace=True)
            # print self.ts[0] + timedelta(days=1)
            df_all = pd.merge(df_all, df_wc, left_index=True, right_index=True, how='left')
        return df_all

    def _find_gage(self, row, col):
        """ Find the gage file for a specific location. Give 0 idx. """
        slr_name = 'SLR-{}_{}'.format(self.slr[0], op.basename(self.path).split('_')[1])
        regex = re.compile('({}.uzf[0-9]+)'.format(slr_name))
        gage_files = [op.join(self.path, slr_name, uzfile) for uzfile
                                in os.listdir(op.join(self.path, slr_name))
                                if regex.search(uzfile)]
        for i, each in enumerate(gage_files):
            linecache.clearcache()
            nums = re.findall(r'\d+', linecache.getline(each, 1))
            ROW = int(nums[1])
            COL = int(nums[2])
            if ROW == row+1 and COL == col+1:
                return each
            else:
                continue
        return 'Gagefile for row {}, col {}, not found in: {}'.format(row, col, self.path)

    def _load_swmm(self):
        """ Dictionary of SLR : 3d Matrix of hrs x row x col """
        dict_run = {}

        for slr in self.slr_sh:
            dict_run[slr] = np.load(op.join(self.path_picks,
                                        'swmm_run_grid_{}.npy').format(slr))
        return dict_run

    def _load_fhd(self):
        """ Load Pickled FHD Arrays """
        dict_fhd         = {}
        for slr in self.slr:
            fhd_pickle    = 'heads_{}.npy'.format(slr)
            pickle_file   = op.join(self.path_picks, fhd_pickle)
            dict_fhd[slr] = (np.load(pickle_file))
        return dict_fhd

    def _load_uzf(self):
        """ Load Pickled UZF Arrays """
        # position 0 is leakage, 1 is rch, 2 is et
        list_uzf = []
        for var in ['surf_leak', 'uzf_rch', 'uzf_et']:
            pickle_files = [op.join(self.path_picks, pick_file) for pick_file in
                            os.listdir(self.path_picks) if pick_file.startswith(var)]
            tmp_dict = {}
            for leak_mat in pickle_files:
                SLR = float(op.basename(leak_mat).split('_')[2][:-4])
                tmp_dict[SLR] = np.load(leak_mat)

            # list of dictionary of scenario (0.0) : 74x51x549 matrix
            list_uzf.append(tmp_dict)

        return list_uzf

    @staticmethod
    def fill_grid(ser):
        """ Create WNC grid from a series with 1 indexed zones and values """
        mat = np.zeros([3774])
        mat[:] = np.nan
        # parse numpy array where column 1 is an index
        if isinstance(ser, np.ndarray):
            for i, idx in enumerate(ser[1, :]):
                idx -= 10001
                mat[int(idx)] = ser[0, i]
            return mat.reshape(74,51)

        # parse series
        else:
            idx = ser.index
            if not np.nansum(ser) == 0:
                for i in idx:
                    mat[int(i)-10001] = ser[i]
            else:
                for i in idx:
                    mat[int(i)-10001] = 1

            return mat.reshape(74,51)

class summary(res_base):
    """ Overall information PLOT HEAD CONTOURS """
    def __init__(self, path_results, row=0, col=2):
        res_base.__init__(self, path_results)
        self.var_map        = {0: 'leak', 1 : 'rch', 2: 'et'}
        self.row, self.col  = row, col
        self.loc_1d         = bcpl.cell_num(self.row, self.col) + 10000

    ### SWMM
    # make this a fill plot, except for precip
    def plot_ts_sys_var(self):
        """ Plot SWMM System Variables by Var """
        print 'this should be a fill plot, except for precip'
        sys_vars  = ['Infil', 'Runoff', 'Surf_Evap', 'Precip']
        fig, axes = plt.subplots(2, 2, figsize=(16,9))
        title     = fig.suptitle('SWMM System Variables')
        axe       = axes.ravel()
        df_mon    =(self.df_sys.loc['2011-12-01-00':'2012-11-30-00',:].resample('Q-NOV')
                                                                 .mean())

        df_mon    = self.df_sys.resample('M').mean().iloc[1:]
        for i, var in enumerate(sys_vars):
            df_slr = df_mon.filter(like=var)
            # print df_slr
            # BB.fill_plot(df_slr, axe[i], title=var)
            df_slr.plot.bar(ax=axe[i], title=var, grid=True)
            axe[i].set_xticklabels(df_slr.index.map(lambda t: t.strftime('%b')))

        fig.autofmt_xdate()
        fig.set_label(title.get_text())
        return fig

    def plot_slr_sys_sums(self):
        """ System Sums vs SLR """
        ser_sums = self.df_sys.sum()
        colnames = [var[:-4] for var in ser_sums.index][:7]
        dict_slr   = {}
        for slr in self.slr:
            dict_slr[slr] = ser_sums.filter(like=str(slr)).values
        df_slr    = (pd.DataFrame(dict_slr, index=colnames).T
                                           .drop(['Vol_Stored', 'Flood', 'Precip', 'Pet'], 1))
        fig, axes = plt.subplots(ncols=3,figsize=(16,9), sharex=True)
        title     = plt.suptitle('SWMM System Sums')
        df_slr.plot(ax=axes, subplots=True, grid=True)
        axes[1].set_xlabel('SLR (m)')

        plt.subplots_adjust(left=None, right=None, wspace=0.4)
        fig.set_label(title.get_text())
        return fig

    ### MF
    # make this a fill plot
    def plot_ts_uzf_sums(self):
        """ Plot Sum of Recharge, ET, and Leakage at all locs, each Step """
        print 'this should be a fill plot'

        list_uzf = self._load_uzf()
        df_sums = pd.DataFrame(index=self.ts_day)
        for i, var in enumerate(list_uzf):
            for slr in self.slr:
                mat_sum = var[slr].reshape(550,-1).sum(1)
                df_sums['{}_{}'.format(self.var_map[i], slr)] = mat_sum

        df_sums   = abs(df_sums)
        fig, axes = plt.subplots(ncols=3, figsize=(16,9))
        title     = fig.suptitle('Sums of UZF Variables by Type')
        axe       = axes.ravel()

        for i, var in enumerate(self.var_map.values()):
            df_slr   = df_sums.filter(like=str(var))
            df_mon   = df_slr.resample('M').mean().iloc[1:, :]
            df_mon.plot.bar(ax=axe[i], title=var, grid=True, stacked=False)
            # BB.fill_plot(df_mon, axe[i], title=var)

            axe[i].set_xticklabels(df_mon.index.map(lambda t: t.strftime('%b')))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.35)

        fig.set_label(title.get_text())
        return fig

class runoff(res_base):
    def __init__(self, path_result):
        res_base.__init__(self, path_result)
        self.df      = pd.read_pickle(op.join(self.path_picks, 'run_seasons.df'))
        self.df_area = pd.read_pickle(op.join(self.path_picks, 'percent_vols.df'))
        self.vols    = BB.uniq([float(vol.split('_')[1]) for vol in self.df_area.columns])
        self.seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        self.dict    = self._load_swmm()

    def plot_area_vol(self):
        """ Plot area vs hours, curves of SLR, subplots of rates """

        area_bins   = np.linspace(1, 100, 11)
        fig, axes   = plt.subplots(2, 2, True, True, figsize=(16,9))
        title       = fig.suptitle('percent area experiencing given runoff depth')
        axe         = axes.ravel()
        for i, vol in enumerate(self.vols):
            cols        = [col for col in self.df_area.columns if col.split('_')[-1] == str(vol)]
            df_area_one = self.df_area[cols]
            df_area_one.columns = ['SLR: {} m'.format(slr.split('_')[0]) for slr in df_area_one.columns]
            df_days     = pd.DataFrame(index=area_bins, columns=df_area_one.columns)
            for area in area_bins:
                df_days.loc[area, :] = (df_area_one>=area).sum()

            # df_days.plot(ax=axe[i], title='DTW: {}'.format(dtw), grid=True)
            BB.fill_plot(df_days, axe[i], title= 'Depth >= {} (mm)'.format(vol))
            axe[i].set_xlabel('% Area')
            axe[i].set_ylabel('Hours')
        fig.set_label(title.get_text())
        return fig

    def plot_ts_sums(self):
        """ Time Series of Runoff Sums """
        df_run    = self.df_sys.filter(like='Runoff')
        fig, axes = plt.subplots(ncols=2, figsize=(16,9))
        title     = fig.suptitle('Runoff vs Time - xlabels are off')
        axe       = axes.ravel()
        # get rid of first day
        df_run = df_run.loc['2012-07-01':, :]

        df_run.plot(ax=axe[0], title='Hourly', grid=True)
        df_run.resample('MS').mean().plot(ax=axe[1], title='Monthly Mean', grid=True)

        axe[0].set_ylabel('Volume (cms)')
        axe[1].set_ylabel('Volume (cms)')
        fig.set_label(title.get_text())
        return fig
    # greater change 1 -0 than 2-1; looks like distribution of conductivities, add plot
    def plot_2d_chg_slr(self):
        """ Plot Grid Change in total Runoff due to SLR """

        low       = self.dict['0.0'].sum(0)
        med       = self.dict['1.0'].sum(0)
        high      = self.dict['2.0'].sum(0)

        chg_1     =  (med  - low) / med  * 100
        chg_2     =  (high - med) / high * 100

        fig, axes = plt.subplots(ncols=2, figsize=(16, 9))
        title     = fig.suptitle('Change in Total Runoff due to SLR', fontsize=14)
        axe       = axes.ravel()

        for i, chg in enumerate([chg_1, chg_2]):
            titles = ['SLR: 1.0 (m) - SLR: 0.0 (m)', 'SLR: 2.0 (m) - SLR: 1.0 (m)']
            im = axe[i].imshow(chg, cmap=plt.cm.jet, vmin=0, vmax=30)
            axe[i].set(title=titles[i])
            axe[i].title.set_size(11)
            axe[i].axis('off')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.175, 0.025, 0.6], xlabel='% Change')
        fig.colorbar(im, cbar_ax, spacing='proportional')
        fig.set_label(title.get_text())
        return fig

    ### MAKE PLOT, SEE IF ANY NEW RUNOFF OPENS UP DUE TO SLR
        # it would be nice if it did, but not necessary;
        # especially since most runoff generated by hortonian

class dtw(res_base):
    def __init__(self, path_result):
        res_base.__init__(self, path_result)
        self.df      = pd.read_pickle(op.join(self.path_picks, 'dtw_seasons.df'))
        self.df_area = pd.read_pickle(op.join(self.path_picks, 'percent_at_surface.df'))#.loc['2011-12-01-00':, :]
        self.dtws    = BB.uniq([float(dtw.split('_')[1]) for dtw in self.df_area.columns])

    def plot_area_days(self):
        """ Plot area vs days, curves of SLR, subplots of DTW """
        area_bins   = np.linspace(0, 100, 11)
        fig, axes   = plt.subplots(2, 2, True, figsize=(16,9))
        title       = fig.suptitle('percent area within given depth to water')
        axe         = axes.ravel()

        for i, dtw in enumerate(self.dtws):
            df_area_one = self.df_area.filter(like=str(dtw))
            df_area_one.columns = ['SLR: {} m'.format(slr.split('_')[0]) for slr in df_area_one.columns]
            df_hrs     = pd.DataFrame(index=area_bins, columns=df_area_one.columns)
            for area in area_bins:
                df_hrs.loc[area, :] = (df_area_one>=area).sum()

            # df_days.plot(ax=axe[i], title='DTW: {}'.format(dtw), grid=True)
            BB.fill_plot(df_hrs, axe[i], title= 'DTW <= {}'.format(dtw))
            axe[i].set_xlabel('% Area')
            axe[i].set_ylabel('Hours')
        fig.set_label(title.get_text())
        return fig

    ### OVERLAY THESE IN GIS; maybe with conductivities too
    def plot_interesting(self, maxchg=-10, maxdtw=30.48):
        """ Plot locs where significant changes due to SLR AND low dtw """
        figures   = []
        self.df     *= 100
        for i in range(len(self.slr_sh) - 1):
            ser_lower    = self.df.filter(like=self.slr_sh[i]).mean(axis=1)
            ser_higher   = self.df.filter(like=self.slr_sh[i+1]).mean(axis=1)
            # max change will be a meter since that's diff in slr_sh
            ser_chg      = ser_higher - ser_lower

            mat_high_dtw = ser_higher.values.reshape(74, 51)
            mat_change   = ser_chg.values.reshape(74, 51)

            df_mean      = pd.DataFrame({self.slr[i]: ser_lower,
                                    self.slr[i+1] : ser_higher,
                                    'change' : ser_chg})

            ### PLOT
            fig, axes = plt.subplots(ncols=3, figsize=(16,9))
            title     = fig.suptitle('Average DTW', fontsize=14)
            axe       = axes.ravel()

            # plot change
            im = axe[0].imshow(mat_change, cmap=plt.cm.jet_r,
                                 vmin=-80, vmax=maxchg)
            axe[0].set(title='Change in DTW (cm): {} m to {} (m) SLR'.format(
                                          self.slr_sh[i], self.slr_sh[i+1]))
            axe[0].title.set_size(10)

            # plot dtw
            im2 = axe[1].imshow(mat_high_dtw, cmap=plt.cm.jet_r,
                                     vmin=0, vmax=maxdtw)
            axe[1].set(title='DTW (cm) for SLR: {} (m) '.format(self.slr_sh[1]))
            axe[1].title.set_size(10)

            # plot intersection
            df_mean.where(df_mean.change.notnull(), -500, inplace=True)

            df_inter = df_mean.where(df_mean.change < maxchg, 100)

            df_inter.where(df_inter[self.slr[i+1]] < maxdtw, 100, inplace=True)
            df_inter.where(df_inter.change != -500, np.nan, inplace=True)

            im3  = axe[2].imshow(df_inter[self.slr[i+1]].values.reshape(74,51),
                                                        cmap=plt.cm.jet_r)
            axe[2].set(title='Locations with Change > {} (cm) and DTW < {}(cm)\nSLR: {} m'
                                        .format(maxchg, maxdtw, self.slr_sh[1]))
            axe[2].title.set_size(10)

        return

class methods(res_base):
    """ Plots for Methods Section. These fns could be cleaner. """
    def __init__(self, path_result, row=0, col=2):
        res_base.__init__(self, path_result)
        self.row    = row
        self.col    = col
        self.loc_1d = bcpl.cell_num(self.row, self.col) + 10000

    def plot_param_mf(self, params='KZL1'):
        df_mf = pd.read_csv(op.join(self.path_data, 'MF_GRID.csv'), index_col='UZF_IBND')
        df_mf[df_mf.index<1] = np.nan

        if not isinstance(params, list):
            params = [params]
        list_of_param_mats = []
        for param in params:
            list_of_param_mats.append(df_mf[param].values.reshape(74,51))


        fig, axes = plt.subplots(ncols=len(params), figsize=(16,9))
        title     = fig.suptitle('SWMM Parameters')
        axe       = axes.ravel() if len(params) > 1 else [axes]

        for i, mat_param in enumerate(list_of_param_mats):
            mat_param1 = np.where(mat_param > 0.00, mat_param, 0)
            mat_param1[np.isnan(mat_param)] = np.nan

            im = axe[i].imshow(mat_param1, cmap=plt.cm.jet)

            axe[i].set(title=params[i])
            axe[i].title.set_size(11)
            # print ax[i].properties()
            # axe[i].axis('off')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.175, 0.025, 0.6], xlabel='Percent')
        fig.colorbar(im, cbar_ax, spacing='proportional')
        fig.set_label(title.get_text())

    def plot_param_swmm(self, params=['perImperv', 'perSlope']):#, 'Min_Inf']):
        """ 2D Plot of a SWMM Parameter FIX BOUNDS """
        if not isinstance(params, list):
            params = [params]
        list_of_param_mats = []
        for param in params:
            list_of_param_mats.append(self.fill_grid(self.df_swmm[param]))

        fig, axes = plt.subplots(ncols=len(params), figsize=(16,9))
        title     = fig.suptitle('SWMM Parameters')
        axe       = axes.ravel() if len(params) > 1 else [axes]

        for i, mat_param in enumerate(list_of_param_mats):
            mat_param1 = np.where(mat_param > 35, mat_param, 0)
            mat_param1[np.isnan(mat_param)] = np.nan

            im = axe[i].imshow(mat_param1, cmap=plt.cm.jet)

            axe[i].set(title=params[i])
            axe[i].title.set_size(11)
            # print ax[i].properties()
            # axe[i].axis('off')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.175, 0.025, 0.6], xlabel='Percent')
        fig.colorbar(im, cbar_ax, spacing='proportional')
        fig.set_label(title.get_text())

    ### Compare 1 Location
    def plot_heads_1loc(self):
        """ Compare heads at SWMM & MF for 2 SLR Scenarios """
        arr_mf   = self._load_fhd()[self.slr[0]][:, self.row, self.col]
        df_heads = pd.DataFrame({'MF_{}'.format(self.slr[0]) :arr_mf}, index=self.ts_day)
        ### SWMM
        df_heads['SWMM_{}'.format(self.slr[0])]  = self.ts_all('head', self.loc_1d,
                                      slr=self.slr[0]).resample('D').first().values

        # add land surface elevation
        df_heads['Land_Surface'] = [np.load(op.join(self.path_data, 'Land_Z.npy'))
                             .reshape(74, 51)[self.row, self.col]] * len(df_heads)

        # start at Nov
        # df_heads = df_heads.loc['2011-12-01':, :]

        # fig, axes = plt.subplots(figsize=(16,9), nrows=2)
        fig  = plt.figure(figsize=(16,9))
        gs   = gridspec.GridSpec(2, 1,
                       width_ratios=[1],
                       height_ratios=[2,1]
                       )
        axe   = []

        axe.append(fig.add_subplot(gs[0]))
        axe.append(fig.add_subplot(gs[1]))

        title = fig.suptitle('Comparisons of Heads: Row {}, Col {}'.format(self.row+1, self.col+1))
        df_heads.plot(ax=axe[0], grid=True, sharex=True)
        axe[0].set_ylabel('GW Head (m)')
        axe[0].set_ylim(3.85, 4.075)
        axe[0].legend(loc='lower left')#, bbox_to_anchor=(0.0, 1.075))

        # secondary plot of surface leakage
        arr_leak = self._load_uzf()[0][self.slr[0]][:, self.row, self.col]
        df_leak  = pd.DataFrame({'Surface Leakage':arr_leak}, index=self.ts_day)
        # df_leak  = df_leak.loc['2011-12-01':, :]
        df_leak.plot(ax=axe[1], legend=False, title='Surface Leakage')
        axe[1].set_ylabel('Vol (cubic meters)')

    # this shouldn't be monthly
    def plot_theta_wc(self):
        """ Compare theta and water content from UZF gage; should be same  """
        print 'SHOULD NOT BE USING MONTHLY MEANS FOR THIS'
        # this loc only works for first row - still? (05-10??)
        fig, axes   = plt.subplots(ncols=len(self.slr), figsize=(16,9), sharey=True)
        title       = fig.suptitle('UZF Water Content vs SWMM Theta \n Loc: {}'
                                                         .format(self.loc_1d))
        axe         = axes.ravel()

        fig2, axes2 = plt.subplots(ncols=len(self.slr), figsize=(16,9), sharey=True)
        title2      = fig2.suptitle('Change in UZF vs SMWM Soil Water \n Loc: {}'.format(self.loc_1d))
        axe2        = axes2.ravel()

        df_wc       = self.ts_gage(self.row, self.col)
        df_wc.fillna(0.375, inplace=True) # fill missing times to match

        for i, slr in enumerate(self.slr):
            df                         = pd.DataFrame(index=self.ts_day)
            df['Water_{}'.format(slr)] = (df_wc.filter(like=str(slr))
                                                ['Water_{}'.format(slr)].values)
            df['SWMM_{}'.format(slr)]  = (self.ts_all('soil', self.loc_1d, slr)
                                                         .resample('D').first()
                                                                       .values)
            df_mon = df.resample('M').mean().iloc[1:, :]
            df_mon.plot.bar(ax=axe[i], title='SLR: {}'.format(slr), grid=True)
            df['mf_{}'.format(slr)] = df_wc['Water_{}'.format(slr)].values
            # df['swmm_{}'.format(slr)] = ser_swmm.values
            df['chg_{}'.format(slr)]  = df['Water_{}'.format(slr)] - df['SWMM_{}'.format(slr)]
            df['chg_{}'.format(slr)].plot(ax=axe2[i], grid=True, title='SLR: {}'.format(slr))

            # df['chg_{}'.format(slr)].plot(ax=ax[i], secondary_y=True)
            ax_lines = axe[i].get_lines() #+ ax[i].right_ax.get_lines()
            axe[i].legend(ax_lines, [l.get_label() for l in ax_lines], ncol=len(ax_lines),
                        loc='upper center', bbox_to_anchor=(0.5, 1.075))
            axe[i].set_xticklabels(df_mon.index.map(lambda t: t.strftime('%b')))

        fig.autofmt_xdate()
        fig.set_label(title.get_text())
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.35)
        return fig

class sensitivity(res_base):
    """ Sums? for sensitivity analysis """
    pass

def make_plots():
    PATH_stor   = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results')
    PATH_result = ('{}_05-08').format(PATH_stor)

    ## summary
    summary_obj = summary(PATH_result)
    # summary_obj.plot_ts_sys_var()
    # summary_obj.plot_slr_sys_sums()
    # summary_obj.plot_ts_uzf_sums()
    #
    # ## runoff
    runoff_obj = runoff(PATH_result)
    # runoff_obj.plot_area_vol()
    # runoff_obj.plot_ts_sums()
    # runoff_obj.plot_2d_chg_slr()
    #
    # ## dtw
    dtw_obj    = dtw(PATH_result)
    # dtw_obj.plot_area_days()
    # dtw_obj.plot_interesting()
    #
    # ## methods
    methods_obj = methods(PATH_result)
    # methods_obj.plot_param_mf()
    # methods_obj.plot_param_swmm()
    methods_obj.plot_heads_1loc()
    # methods_obj.plot_theta_wc()

    plt.show()

make_plots()
