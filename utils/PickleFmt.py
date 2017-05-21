"""
Format Pickled Output into Data for Plots
Call main from PickleRaw.py
Created: 2017-05-03
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import swmmtoolbox as swmtbx

class fmt_base(object):
    """ Base class for formatting Pickled Model Output """
    def __init__(self, path_result):
        self.path       = path_result
        self.path_picks = op.join(self.path, 'Pickles')
        self.path_data  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC',
                                                             'Coupled', 'Data')
        self.df_swmm    = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'),
                                                          index_col='Zone')
        _               = self.make_scenarios_slr()
        __              = self.make_ts()
        ___             = self.make_swmm_grid('heads'), self.make_swmm_grid('run')
        # self.seasons    = ['Winter', 'Spring', 'Summer', 'Fall']

    def make_scenarios_slr(self):
        """ Get Scenarios and SLR from Dirs """
        self.scenarios = [op.join(self.path, slr) for slr in os.listdir(self.path)
                                                if slr.startswith('SLR')]
        self.slr       = [op.basename(scenario).split('_')[0][-3:] for
                                         scenario in self.scenarios]

    def make_ts(self):
        """ Make Time Series, Hourly and Daily """
        slr_name    = 'SLR-{}_{}'.format(self.slr[0], self.scenarios[0][-5:])
        out_file    = op.join(self.path, slr_name, slr_name + '.out')
        st_end      = swmtbx.SwmmExtract(out_file).GetDates() # returns a tuple
        self.ts_hr  = pd.date_range(st_end[0], st_end[1], freq='H')
        self.ts_day = pd.date_range(st_end[0], st_end[1], freq='D')

    def make_swmm_grid(self, kind):
        """
        Reshape SWMM subcatchments into full grid (Hrs x 74 x 51
        'heads' or 'run'; Save to Pickle Dir
        """
        f_name = 'swmm_{}_grid_{}.npy'.format(kind, self.slr[0])
        if op.exists(op.join(self.path_picks, f_name)):
            print '{} exists, not making swmm grids...'.format(f_name)
            return
        for i, slr in enumerate(self.slr):
            f_name_old = 'swmm_{}_{}.npy'.format(kind, slr)
            mat_heads  = np.load(op.join(self.path_picks, f_name_old))
            mat_res    = np.zeros([mat_heads.shape[0], 74, 51])

            # iterate over times
            for t in range(mat_heads.shape[0]):
                mat_tmp    = (np.concatenate([mat_heads[t], self.df_swmm.index])
                                                            .reshape(2, -1))
                mat_res[t] = self.fill_grid(mat_tmp)
            f_name_new = 'swmm_{}_grid_{}.npy'.format(kind, slr)
            np.save(op.join(self.path_picks, f_name_new), mat_res)

    @staticmethod
    def fill_grid(ser):
        """ Create WNC grid from a series or array with 1 idx zones / values """
        mat = np.zeros([3774])
        # parse numpy array where column 1 is an index
        if isinstance(ser, np.ndarray):
            mat[:] = np.nan
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

class fmt_dtw(fmt_base):
    def __init__(self, path_result):
        fmt_base.__init__(self, path_result)

    def make_df_dtw_area(self, dtw=0.1524):
        """
        Get area where head within a list of DTWs (in m) of surface using SWMM
        Returns a Df, rows are day, columns are slr_dtw theshold.
            Values are percentage of entire area
        """
        dtws     = np.linspace(dtw, dtw*4, 4)
        colnames = ['{}_{}'.format(slr, dtw) for slr in self.slr for dtw in dtws]
        mat_res  = np.zeros([len(self.ts_hr), len(colnames)], dtype=np.float64)

        for i, slr in enumerate(self.slr):
            mat_dtw = self._make_mat_dtw(slr)

            for j, dtw in enumerate(dtws):
                # assign 1 to cells within threshold
                mat_dtw_below_thresh = np.where(mat_dtw[:] <= dtw, 1, 0)
                area_below_thresh    = np.nansum(mat_dtw_below_thresh.reshape(
                                                  len(self.ts_hr), -1), axis=1)
                for k in range(mat_dtw.shape[0]):
                    mat_res[k, j + i * len(dtws)] = area_below_thresh[k]

        area     = float(np.count_nonzero(~np.isnan(mat_dtw[0])))
        mat_res *= (100/area) # to percent area

        df_res   = pd.DataFrame(mat_res, index=self.ts_hr, columns=colnames)
        fname    = 'percent_at_surface.df'
        df_res.to_pickle(op.join(self.path_picks, fname))
        print 'DataFrame pickled to: {}'.format(op.join(self.path_picks, fname))

    def make_df_dtw_year(self):
        """
        Make annual average DTW
        Convert heads to DTW for all times, all SLR, all locations
        Return Df with 1 indexed row/col as index
        """
        df_yrs = []
        for i, slr in enumerate(self.slr):
            mat_dtw = self._make_mat_dtw(slr).reshape(-1, 3774)
            df_dtw  = pd.DataFrame(mat_dtw, index=self.ts_hr,
                                             columns=range(10001, 10001 + 3774))
            # take one full year of dates, resample by season
            df_yr   = df_dtw.loc['2011-12-01-00':'2012-11-30-00',:].mean(0).T
            print df_yr.head()
            df_yr.name = float(slr)
            df_yrs.append(df_yr)
        df_res = pd.concat([df_yr for df_yr in df_yrs], axis=1)
        fname = 'dtw_yr.df'
        df_res.to_pickle(op.join(self.path_picks, fname))
        print 'DataFrame pickled to: {}'.format(op.join(self.path_picks, fname))


    def _make_mat_dtw(self, slr):
        """ Make Matrix of Elevations - Heads (DTW) (ts x 74 x 51) """
        mat_surf     = self.fill_grid(self.df_swmm.Esurf)

        f_name    = 'swmm_heads_grid_{}.npy'.format(slr)
        mat_heads = np.load(op.join(self.path_picks, f_name))

        return mat_surf - mat_heads

class fmt_run(fmt_base):
    def __init__(self, path_result):
        fmt_base.__init__(self, path_result)
        # sec/step to vol at each hr  / area / m to mm
        self.rate2dep = (60 * 60) / (200. * 200.) * 1000

    def make_df_vol_area(self):
        """
        Get area where runoff greater than a list of volumes (in CM)
        Returns a Df, rows are hours, columns are slr_rate theshold.
            Values are percentage of entire area experiencing this vol or greater
        Pickled to: pickles/percent_rates.df
        Based on fhd_base().make_df_dtw_area()
        """
        vols     = np.array([0.1, 1, 10, 100])
        colnames = ['{}_{}'.format(slr, vol) for slr in self.slr for vol in vols]
        mat_res  = np.zeros([len(self.ts_hr), len(colnames)], dtype=np.float64)
        area     = float(len(self.df_swmm))
        for i, slr in enumerate(self.slr):
            f_name   = 'swmm_run_grid_{}.npy'.format(slr)
            mat_run  = np.load(op.join(self.path_picks, f_name)) * self.rate2dep

            for j, vol in enumerate(vols):
                # assign 1 to cells within threshold
                mat_area_above_thresh = np.where(mat_run[:] >= vol, 1, 0)
                area_above_thresh     = np.nansum(mat_area_above_thresh.reshape(
                                                len(self.ts_hr), -1), axis=1)
                for k in range(mat_run.shape[0]):
                    mat_res[k, j+ i* len(vols)] = area_above_thresh[k]

        mat_res *= (100/area)
        df_res   = pd.DataFrame(mat_res, index=self.ts_hr, columns=colnames)
        # print df_res.describe()

        fname    = 'percent_vols.df'
        df_res.to_pickle(op.join(self.path_picks, fname))
        print 'DataFrame pickled to: {}'.format(op.join(self.path_picks, fname))

        return df_res

def fmt_slr():
    """
    Format the tables produced in and exported from ArcGIS.
    Each table is a NOAA SLR-Viewer scenario intersected with grid.
    Table contains only the cells affected for each SLR scenario, w/ dupes.
    """
    PATH_home = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC')
    PATH_data = op.join(PATH_home, 'Coupled', time.strftime('%b'), 'Data')
    PATH_noaa = op.join(PATH_data, 'slr_tables')
    grid      = pd.read_csv(op.join(PATH_data, 'MF_GRID.csv')).rename(
                            columns={'Unnamed: 0':'Zone'}).set_index('Zone')

    keep = ['ROW', 'COLUMN', 'IBND', 'UZF_IBND', 'MODEL_TOP']
    grid = grid[keep]

    for i, slr in enumerate(os.listdir(PATH_noaa)):
        # ternary
        if not slr.endswith('.csv') : continue
        colname = op.splitext(slr)[0]
        df_noaa = pd.read_csv(op.join(PATH_noaa, slr))#.set_index('Zone')
        # remove duplicate rows
        df_noaa_1 = df_noaa.groupby('Zone').sum()
        # create series which will add to main grid
        df_noaa2 = pd.DataFrame({colname : colname[-1],
                                 colname+'_Area' : df_noaa_1['Shape_Area']},
                                 index=df_noaa_1.index)
        grid = grid.join(df_noaa2)
    grid.ROW = grid.ROW.astype(int)
    # add an indicator for where CHD cells are
    grid['CHD'] = grid.UZF_IBND - grid.IBND
    grid.to_pickle(op.join(PATH_noaa, 'NOAA_grid.df'))
    return grid

def main(path_result):
    dtw = fmt_dtw(path_result)
    dtw.make_df_dtw_season()
    dtw.make_df_dtw_area(dtw=0.1524)
    print ('Pickled DTW Results')

    run = fmt_run(path_result)
    run.make_df_vol_area(vol=0.01)
    run.make_df_run_season()
    print ('Pickled Runoff Results')

if __name__ == '__main__':
    # PATH_result = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-18')
    main(PATH_result)
