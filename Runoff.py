import BB
import os
import os.path as op

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas
import shapely
from shapely.geometry import Point

from Analysis import res_base, summary, runoff, rc_params
from Surf_Leak import surf_leak

class runoff(res_base):
    def __init__(self, path_result):
        res_base.__init__(self, path_result)
        self.df_area = pd.read_pickle(op.join(self.path_picks, 'percent_vols.df'))
        self.vols    = BB.uniq([float(vol.split('_')[1]) for vol in self.df_area.columns])
        self.dict    = self._load_swmm('run')

    def plot_ts_total(self):
        """ Monthly Times series of Total System Runoff """
        df_run    = self.df_sys.filter(like='Runoff').resample('MS').mean().loc[self.st:self.end, :]
        fig, axes = plt.subplots()
        colnames  = ['SLR: {} m'.format(c.split('_')[-1]) for c in df_run.columns]
        df_run.columns = colnames
        # could probably change global alpha and color cycles
        df_run.plot.bar(ax=axes, color=['darkblue', 'darkgreen', 'darkred'], alpha=0.75)
        axes.set_xticklabels(df_run.index.map(lambda t: t.strftime('%b')))
        axes.legend(loc='upper left')
        axes.set_xlabel('Time', labelpad=15, size=14)
        axes.set_ylabel('Runoff Rate (CMS)', labelpad=15, size=14)
        axes.yaxis.grid(True)
        axes.xaxis.grid(False)

        fig.autofmt_xdate(bottom=0.2, rotation=20, ha='center')
        fig.set_label('total_monthly_runoff')

    def shp_2d_chg(self):
        """ Write shapefile of 2d change due to SLR """
        low         = self.dict['0.0'][3696:-698, :, :].sum(0).reshape(-1)
        med         = self.dict['1.0'][3696:-698, :, :].sum(0).reshape(-1)
        high        = self.dict['2.0'][3696:-698, :, :].sum(0).reshape(-1)

        chg_1       =  (med  - low) / med  * 100
        chg_2       =  (high - med) / high * 100
        df          = self.df_xy.loc[:, ['POINT_X', 'POINT_Y']]
        df['1_0']   = chg_1
        df['2_0']   = chg_2
        df['geom']  = df.apply(lambda x: Point((float(x.POINT_X), float(x.POINT_Y))), axis=1)
        geo_df      = geopandas.GeoDataFrame(df, geometry='geom')
        geo_df.to_file(op.join(self.path_gis, 'Runoff_Chg.shp'), driver='ESRI Shapefile')
        print 'Runoff ShapeFile Written: {}'.format(op.join(self.path_gis, 'Runoff_Chg.shp'))

    def plot_area_vol(self):
        """ Plot area vs hours, curves of SLR, subplots of rates """
        area_bins   = np.linspace(1, 100, 80)
        fig, axes   = plt.subplots(2, 2, True, True)
        # title       = fig.suptitle('percent area experiencing given runoff depth')
        axe         = axes.ravel()
        for i, vol in enumerate(self.vols):
            cols        = [col for col in self.df_area.columns if col.split('_')[-1] == str(vol)]
            df_area_one = self.df_area[cols]
            df_area_one.columns = ['SLR: {} m'.format(slr.split('_')[0]) for slr in df_area_one.columns]
            df_days     = pd.DataFrame(index=area_bins, columns=df_area_one.columns)
            for area in area_bins:
                df_days.loc[area, :] = (df_area_one>=area).sum()

            df_days.plot(ax=axe[i], title='Depth >= {} mm'.format(vol),
                         color=['darkblue', 'darkgreen', 'darkred'], alpha=0.5)

            axe[i].set_xlabel('% Area', labelpad=15, size=14)
            axe[i].set_ylabel('Hours',  labelpad=15, size=14)
            axe[i].xaxis.grid(False)
            axe[i].yaxis.grid(True)
        fig.subplots_adjust(left=0.075, right=0.92, wspace=0.15, hspace=0.15)
        fig.set_label('runoff_area')
        return fig

    def __leak_Vs_runoff(self):
        """
        Compare surface leakage and runoff for whole simulation
        Runoff units may be off / and uzf. both time and area (200 * 200 * 1455?)
        Internal Check for Me
        """
        area    = 200. * 200.
        arr_uzf = abs(self._load_uzf('surf_leak', 2.0).reshape(len(self.ts_day), -1).sum(1)/area)
        ser_uzf = pd.Series(arr_uzf, index=self.ts_day).loc[self.st:self.end]
        arr_run = np.nansum(self._load_swmm('run', 2.0).reshape(len(self.ts_hr), -1), 1)/area#*3600
        ser_run = pd.Series(arr_run, index=self.ts_hr).resample('D').sum().loc[self.st:self.end]
        df      = pd.DataFrame({'Leak_2.0' : ser_uzf, 'Run_2.0' : ser_run}, index=ser_run.index)
        df.plot(subplots=True)

plt.style.use('seaborn')
PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-18')
rc_params()
runobj = runoff(PATH_res)
# runobj.shp_2d_chg()
runobj.plot_area_vol()
runobj.save_cur_fig()

plt.show()
