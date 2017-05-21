import BB
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import geopandas
import shapely
from shapely.geometry import Point

from Analysis import res_base, summary, runoff, rc_params
from Surf_Leak import surf_leak

class dtw(res_base):
    def __init__(self, path_result):
        res_base.__init__(self, path_result)
        # pickle converted this to a year
        self.df_year  = pd.read_pickle(op.join(self.path_picks, 'dtw_yr.df'))
        self.df_area  = pd.read_pickle(op.join(self.path_picks, 'percent_at_surface.df'))#.loc['2011-12-01-00':, :]
        self.dtws     = BB.uniq([float(dtw.split('_')[1]) for dtw in self.df_area.columns])

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

    def plot_hist_dtw(self, bins=10):
        """ Create a Histogram of DTW for each SLR Scenario """
        fig         = plt.figure()
        axe         = []
        gs          = gridspec.GridSpec(3, 3)
        colors      = ['darkblue', 'darkgreen', 'darkred']
        # colors      = ['#4C72B0', '#55A868', '#C44E52'] # seaborn defaults
        axe.append(fig.add_subplot(gs[:2, 0]))
        axe.append(fig.add_subplot(gs[:2, 1], sharey=axe[0]))
        axe.append(fig.add_subplot(gs[:2, 2], sharey=axe[0]))
        axe.append(fig.add_subplot(gs[2, :]))

        for i, slr in enumerate(self.df_year):
            mask_yr = np.ma.masked_invalid(self.df_year[slr].values).compressed()
            n, bin_edges, _ = axe[i].hist(mask_yr, bins=bins,
                            align='left', rwidth=0.55, histtype='bar',
                            facecolor=colors[i], alpha=0.825)

            colname = 'SLR-{} (m)'.format(slr)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2      # for unsmoothed
            # fit a line to the middles of the bins
            fit     = spline(bin_edges[:-1], n)
            x2      = np.linspace(bin_edges[:-1].min(), bin_edges[:-1].max(), 1000)
            y       = fit(x2)

            # plot smoothed curve on the bottom
            axe[3].plot(x2, y, color=colors[i], label=colname)
            # axe[3].plot(centers, n, color=colors[i], label=colname) # unsmoothed

            axe[i].set_title('SLR: {} m'.format(slr))
            # axe[i].xaxis.set_ticks(np.linspace(0.0, 9.0, 10.0))
            axe[i].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            # axe[i].set_xlim(-0.25, 7.7)
            axe[0].set_ylabel('Frequency', labelpad=15)
            # axe[1].set_xlabel('GW Head (m)', labelpad=15)

        # turn off shared ticks
        plt.setp(axe[1].get_yticklabels(), visible=False)
        plt.setp(axe[2].get_yticklabels(), visible=False)

        # bottom subplot properties
        axe[3].legend()
        axe[3].set_xlabel('DTW (m)', labelpad=10)
        axe[3].set_ylabel('Frequency', labelpad=15)
        # axe[3].xaxis.set_ticks(np.linspace(0.0, 9.0, 10))
        axe[3].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        # axe[3].set_xlim(-0.05, 7.7)
        gs.update(bottom=0.075, hspace=0.3, wspace=0.15)
        fig.set_label('hist_dtw')

    def interesting(self, maxchg=-30, maxdtw=30.48):
        """ Make locs where signif changes due to SLR AND low dtw  units=cm """
        figures        = []
        self.df_year  *= 100
        df_inter       = pd.DataFrame(index=self.df_year.index)
        for i in range(len(self.slr_sh) - 1):
            slr_low            = float(self.slr_sh[i])
            slr_high           = float(self.slr_sh[i+1])
            chg_col            = 'chg_{}'.format(slr_high)
            df_inter[slr_low]  = self.df_year[slr_low]
            df_inter[slr_high] = self.df_year[slr_high]
            # max change will be a meter since that's diff in slr_sh
            df_inter[chg_col]  = df_inter[slr_high] - df_inter[slr_low]

            ### make the intersections
            df_inter['inter_{}'.format(slr_high)] = (
                                       df_inter.loc[:, slr_high].where(
                                      (df_inter.loc[:, slr_high]<maxdtw)
                                    & (df_inter.loc[:, chg_col].values<maxchg)))

        return df_inter

    def plot_interesting(self, maxchg=-10, maxdtw=30.48):
        df_inter         = self.interesting()
        for i in range(len(self.slr_sh) - 1):
            slr_low      = float(self.slr_sh[i])
            slr_high     = float(self.slr_sh[i+1])
            chg_col      = 'chg_{}'.format(slr_high)
            inter_col    = 'inter_{}'.format(slr_high)
            arr_change   = df_inter[chg_col].values.reshape(74, 51)
            arr_high_dtw = df_inter[slr_high].values.reshape(74, 51)
            arr_inter    = df_inter[inter_col].values.reshape(74, 51)

            ### PLOT
            fig, axes = plt.subplots(ncols=3)
            title     = fig.suptitle('Average DTW')
            axe       = axes.ravel()

            # plot change
            im = axe[0].imshow(arr_change, cmap=plt.cm.jet_r,
                                 vmin=-80, vmax=maxchg)
            axe[0].set(title='Change in DTW cm: {} m to {} m SLR'.format(
                                                    slr_low, slr_high))

            # plot dtw
            im2 = axe[1].imshow(arr_high_dtw, cmap=plt.cm.jet_r,
                                     vmin=0, vmax=maxdtw)
            axe[1].set(title='DTW cm for SLR: {} m '.format(slr_high))

            im3  = axe[2].imshow(arr_inter, cmap=plt.cm.jet_r)
            axe[2].set(title='Locations with Abs Change > {} cm and DTW < {} cm\
                              \nSLR: {} m to {} m'.format(-1*maxchg, maxdtw,
                                                         slr_low, slr_high))
        return

    def shp_interesting(self):
        """ Write shapefile of interesting change locations due to SLR """
        df_interest = self.interesting()
        df_xy       = self.df_xy.loc[:, ['POINT_X', 'POINT_Y']]
        df_xy.index = df_interest.index
        df          = pd.concat([df_interest, df_xy], axis=1)
        df.columns  = [str(col).replace('.', '_') for col in df.columns]

        df['geom']  = df.apply(lambda x: Point((float(x.POINT_X), float(x.POINT_Y))), axis=1)
        geo_df      = geopandas.GeoDataFrame(df, geometry='geom')
        geo_df.to_file(op.join(self.path_gis, 'DTW_Chg.shp'), driver='ESRI Shapefile')
        print 'DTW ShapeFile Written: {}'.format(op.join(self.path_gis, 'DTW_Chg.shp'))

plt.style.use('seaborn')
PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-18')
rc_params()
dtw = dtw(PATH_res)
# dtw.plot_area_hours()
# dtw.plot_hist_dtw()
# dtw.plot_interesting()
# dtw.shp_interesting()
plt.show()
