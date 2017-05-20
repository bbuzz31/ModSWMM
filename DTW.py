import BB
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas
import shapely
from shapely.geometry import Point

from Analysis import res_base, summary, runoff, rc_params
from Surf_Leak import surf_leak

class dtw(res_base):
    def __init__(self, path_result):
        res_base.__init__(self, path_result)
        # self.df      = pd.read_pickle(op.join(self.path_picks, 'dtw_seasons.df'))
        self.df_area = pd.read_pickle(op.join(self.path_picks, 'percent_at_surface.df'))#.loc['2011-12-01-00':, :]
        self.dtws    = BB.uniq([float(dtw.split('_')[1]) for dtw in self.df_area.columns])

    def plot_area_hours(self):
        """ Plot area vs hours, curves of SLR, subplots of DTW """
        area_bins   = np.linspace(0, 100, 40)
        fig, axes   = plt.subplots(2, 2, True, True)
        # title       = fig.suptitle('percent area within given depth to water')
        axe         = axes.ravel()
        for i, dtw in enumerate(self.dtws):
            df_area_one = self.df_area.filter(like=str(dtw)).loc[self.st:self.end, :]
            df_area_one.columns = ['SLR: {} m'.format(slr.split('_')[0])
                                        for slr in df_area_one.columns]
            df_hrs     = pd.DataFrame(index=area_bins, columns=df_area_one.columns)
            for area in area_bins:
                df_hrs.loc[area, :] = (df_area_one>=area).sum()
            BB.fill_plot(df_hrs, axe[i], title='DTW <= {} m'.format(dtw))
            axe[i].set_xlabel('% Area', size=15, labelpad=15)
            axe[i].set_ylabel('Hours', size=15, labelpad=15)
            axe[i].xaxis.grid(False)
            axe[i].legend(loc='lower left', frameon=True, shadow=True, facecolor='white')
            axe[i].set_ylim((0, len(df_area_one)*1.10))
            # axe[i].legend.get_frame(facecolor='white')
        fig.set_label('dtw_area')
        return fig

    def interesting(self, maxchg=-10, maxdtw=30.48):
        """ Make locs where significant changes due to SLR AND low dtw """
        figures   = []
        df  *= 100

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
            fig, axes = plt.subplots(ncols=3)
            title     = fig.suptitle('Average DTW')
            axe       = axes.ravel()

            # plot change
            im = axe[0].imshow(mat_change, cmap=plt.cm.jet_r,
                                 vmin=-80, vmax=maxchg)
            axe[0].set(title='Change in DTW (cm): {} m to {} (m) SLR'.format(
                                          self.slr_sh[i], self.slr_sh[i+1]))

            # plot dtw
            im2 = axe[1].imshow(mat_high_dtw, cmap=plt.cm.jet_r,
                                     vmin=0, vmax=maxdtw)
            axe[1].set(title='DTW (cm) for SLR: {} (m) '.format(self.slr_sh[1]))

            # plot intersection
            df_mean.where(df_mean.change.notnull(), -500, inplace=True)

            df_inter = df_mean.where(df_mean.change < maxchg, 100)

            df_inter.where(df_inter[self.slr[i+1]] < maxdtw, 100, inplace=True)
            df_inter.where(df_inter.change != -500, np.nan, inplace=True)

            im3  = axe[2].imshow(df_inter[self.slr[i+1]].values.reshape(74,51),
                                                        cmap=plt.cm.jet_r)
            axe[2].set(title='Locations with Change > {} (cm) and DTW < {}(cm)\nSLR: {} m'
                                        .format(maxchg, maxdtw, self.slr_sh[1]))

        return

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
            fig, axes = plt.subplots(ncols=3)
            title     = fig.suptitle('Average DTW')
            axe       = axes.ravel()

            # plot change
            im = axe[0].imshow(mat_change, cmap=plt.cm.jet_r,
                                 vmin=-80, vmax=maxchg)
            axe[0].set(title='Change in DTW (cm): {} m to {} (m) SLR'.format(
                                          self.slr_sh[i], self.slr_sh[i+1]))

            # plot dtw
            im2 = axe[1].imshow(mat_high_dtw, cmap=plt.cm.jet_r,
                                     vmin=0, vmax=maxdtw)
            axe[1].set(title='DTW (cm) for SLR: {} (m) '.format(self.slr_sh[1]))

            # plot intersection
            df_mean.where(df_mean.change.notnull(), -500, inplace=True)

            df_inter = df_mean.where(df_mean.change < maxchg, 100)

            df_inter.where(df_inter[self.slr[i+1]] < maxdtw, 100, inplace=True)
            df_inter.where(df_inter.change != -500, np.nan, inplace=True)

            im3  = axe[2].imshow(df_inter[self.slr[i+1]].values.reshape(74,51),
                                                        cmap=plt.cm.jet_r)
            axe[2].set(title='Locations with Change > {} (cm) and DTW < {}(cm)\nSLR: {} m'
                                        .format(maxchg, maxdtw, self.slr_sh[1]))

        return

    def shp_interesting(self):
        pass

plt.style.use('seaborn')
PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-18')
rc_params()
dtw = dtw(PATH_res)
# dtw.plot_area_hours()
dtw.plot_interesting()
plt.show()
