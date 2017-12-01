from AnalysisObjs import *
import time
import seaborn as sns
import pickle

class Wetlands(res_base):
    def __init__(self, path_results, dtw_inc=0.01, hrs_per=50, z_thresh=3.75):
        super(Wetlands, self).__init__(path_results)
        self.path_res                         = path_results
        self.dtw_inc                          = dtw_inc
        self.hrs_per                          = 50
        self.z_thresh                         = 3.75
        self.mat_dtw, self.mat_dtw_masked     = self._make_dtw()
        self.df_wets, self.mat_wets           = self._ccap_wetlands()
        self.mask_wet    = np.isnan(self.mat_wets.reshape(-1))
        # highest values (closest to 0) = most hours with lowest DTW
        self.mat_dtw_sum = (self.mat_dtw * -1).sum(axis=1)
        self.summer      = pd.date_range('2012-06-01-00', '2012-09-01-00', freq='h')
        
    def _ccap_wetlands(self):
        """ Get CCAP wetlands df and full matrix grid """
        df_subs = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'))

        ## convert to full grid
        ser_landcover = df_subs.set_index('Zone').loc[:, 'Majority']
        mat_landcover = res_base.fill_grid(ser_landcover, fill_value=-1)
        mat_wetlands  = np.where(((mat_landcover < 13) | (mat_landcover > 18)),
                                            np.nan, mat_landcover)

        df_wet  = df_subs[((df_subs.Majority >= 13) & (df_subs.Majority < 19))].iloc[:, :3]
        return (df_wet, mat_wetlands)

    def _make_dtw(self, slr=0.0):
        """ dtw, all locations all times; z threshold cuts elevations, grabs 84% of wetlands """
        mat_heads = np.load(op.join(self.path_picks, 'swmm_heads_grid_{}.npy'
                                                                .format(slr)))
        mat_z     = np.load(op.join(self.path_data, 'Land_Z.npy'))
        mat_dtw   = (mat_z.reshape(self.nrows, self.ncols) - mat_heads).reshape(mat_heads.shape[0], -1)

        ### truncate init conditions & transpose
        mat_dtw_trunc = mat_dtw[self.ts_st:self.ts_end].T

        ### truncate by elevations
        ## need to keep track of index
        mat_dtw_indexed         = np.zeros([mat_dtw_trunc.shape[0], mat_dtw_trunc.shape[1]+1])
        mat_dtw_indexed[:, :-1] = mat_dtw_trunc
        mat_dtw_indexed[:, -1]  = np.arange(10001, 10001+mat_dtw_trunc.shape[0])

        mask_z = ((mat_z <= self.z_thresh) & (mat_z > 0.))

        return mat_dtw_trunc, mask_z

    def make_indicator(self, masked=True, seasonal=False):
        """
        Make an indicator by iterating over depth to water and hours at that dtw
        dtw_inc  = dtw increment, use 0.01 for increased precision (expensive)
        hrs_per  = percent of total hours to begin minimum threshold
        seasonal = search just summer?
        """
        start          = time.time()
        if masked:
            ### chop off cells whose elevation is too high
            mat_dtw  = self.mat_dtw[self.mat_dtw_masked]
            mask_wet = self.mask_wet[self.mat_dtw_masked]
            names    = ['dtw_hrs_wet_dry_masked.npy', 'dtw_hrs_wet_dry_masked.df']
        else:
            mat_dtw  = self.mat_dtw
            mask_wet = self.mask_wet
            names    = ['dtw_hrs_wet_dry.npy', 'dtw_hrs_wet_dry.df']

        ### select wetland/nonwetland from all dtw information
        mat_wet_dtw    = mat_dtw[~mask_wet]
        mat_nonwet_dtw = mat_dtw[mask_wet]
        mat_dry_dtw    = mat_nonwet_dtw[~np.isnan(mat_nonwet_dtw)].reshape(
                                                -1, mat_nonwet_dtw.shape[1])

        ## to view the amount of wetlands and drylands working with
        # print (mat_wet_dtw.shape)
        # print (mat_dry_dtw.shape)

        ## truncate for just summer
        if seasonal:
            df_wet_dtw1 = pd.DataFrame(mat_wet_dtw.T, index=self.ts_yr_hr)
            df_wet_dtw  = pd.DataFrame(mat_wet_dtw.T, index=self.ts_yr_hr).loc[self.summer, :]

            df_dry_dtw  = pd.DataFrame(mat_dry_dtw.T, index=self.ts_yr_hr).loc[self.summer, :]
            mat_wet_dtw = df_wet_dtw.values.T
            mat_dry_dtw = df_dry_dtw.values.T
            names       = ['dtw_hrs_wet_dry_summer1.npy', 'dtw_hrs_wet_dry_summer1.df']

        print ('Finding optimum criteria; will take a bit')
        dtw_tests = np.arange(0, 1, self.dtw_inc)
        hrs_tests = range(int(np.floor(1./self.hrs_per)*self.mat_dtw.shape[1]), self.mat_dtw.shape[1])
        mat_all   = np.zeros([len(dtw_tests) * len(hrs_tests), 7])

        for i, dtw_test in enumerate(dtw_tests):
            for j, hrs_test in enumerate(hrs_tests):
                res_wet = ((mat_wet_dtw <= dtw_test).sum(axis=1) > hrs_test).sum()
                res_dry = ((mat_dry_dtw <= dtw_test).sum(axis=1) > hrs_test).sum()
                mat_all[i*len(hrs_tests)+j, 0] = dtw_test
                mat_all[i*len(hrs_tests)+j, 1] = hrs_test
                mat_all[i*len(hrs_tests)+j, 2] = res_wet
                mat_all[i*len(hrs_tests)+j, 4] = res_dry

        mat_good       = mat_all[mat_all[:,2]>0]
        mat_good[:, 3] = mat_good[:,2]/float(mat_wet_dtw.shape[0])
        mat_best       = mat_good[mat_good[:,3] >= 0.50]
        mat_best[:, 5] = mat_best[:,4] / float(mat_dry_dtw.shape[0])
        mat_best[:, 6] = mat_best[:,3] / (1 - (mat_best[:,5]))
        colnames = ['dtw_thresh', 'hrs_thresh', 'n_wet', 'perWet', 'n_dry', 'perDry', 'perRatio']
        df_all  = pd.DataFrame(mat_best, columns=colnames).sort_values(by='perRatio', ascending=False)

        answered = False
        end      = time.time()
        while not answered:
            overwrite = raw_input('Overwrite pickles? (y/n) ')
            if overwrite == 'y':
                np.save(op.join(self.path_data, names[0]), mat_best)
                df_all.to_pickle(op.join(self.path_data, names[1]))
                answered = True
            elif overwrite == 'n':
                print ('Not overwriting pickles')
                answered = True
            else:
                print ('Choose y or n')

        print ('Elapsed time: ~{} min'.format(round((end-start)/60.), 4))

    def apply_indicator(self, seasonal=False):
        """ Analyze the indicator developed using make_indicator """
        if seasonal:
            names = ['dtw_hrs_wet_dry_summer.npy', 'dtw_hrs_wet_dry_summer.df']
            perWet_thresh = 0.61
            perDry_thresh = 0.35
        else:
            names = ['dtw_hrs_wet_dry.npy', 'dtw_hrs_wet_dry.df']
            perWet_thresh = 0.645
            perDry_thresh = 0.35
        mat_all = np.load(op.join(self.path_data, names[0]))
        df_all  = pd.read_pickle(op.join(self.path_data, names[1]))
        # print (df_all.head(25))
        ## do some cropping
        df_new = (df_all[((df_all.hrs_thresh > df_all.hrs_thresh.max()/2.) &
                                          (df_all.perWet > perWet_thresh) &
                                          (df_all.perDry < perDry_thresh))]
                                          .sort_values(by=['perDry', 'perWet'],
                                          ascending=[True, False]))

        ### can get about 1/2 the wetlands and 1/4 of the uplands
        ### best for all is ~ 65% wetlands, 35% of drylands
        ### best for summer is ~ 61% wetlands and 34.4% of drylands
        BB.print_all(df_new)

    def find_cells(self, dtw_thresh=0.05, hrs_thresh=4442):
        """ Find locations that meet the threshold conditions """
        df_dtw = pd.DataFrame(self.mat_dtw, columns=self.ts_yr_hr)

        ## number of hrs below dtw threshold
        df_dtw['dtw_counts'] = ((df_dtw<=dtw_thresh).sum(axis=1))

        ## put true if passed hrs threshold
        df_dtw['wet_test']   = df_dtw.dtw_counts.apply(lambda x: True if x >= hrs_thresh else False)

        ## attach elevation data ; zones col is one indexed in df_subs
        df_dtw.index += 10001
        df_merged = df_dtw.merge(self.df_swmm.loc[:, ['Esurf', 'Majority']],
                            left_index=True, right_index=True, how='outer')
        df_passed      = df_merged[df_merged['wet_test']]

        df_low         = df_passed[df_passed.Esurf <= self.z_thresh]
        # df_passed_wets = df_low[df_low.index.isin(self.df_wets.Zone)]
        # df_passed_drys = df_low[~df_low.index.isin(self.df_wets.Zone)]
        df_low['dry']  = df_low.index.map(lambda x: -1 if x in self.df_wets.loc[:, 'Zone'].values else 0)

        print ('Wets: {}'.format(df_low[df_low['dry']<0].shape[0]))
        print ('Drys: {}'.format(df_low[df_low['dry']==0].shape[0]))
        # return df_passed_wets, df_passed_drys
        return df_low


    def plot_wets_drys(self):
        df_lows = self.find_cells(dtw_thresh=0.05, hrs_thresh=4442)

        ## change value for cells not considered (visualization purposes)
        df_subs        = pd.DataFrame(index=self.df_swmm.index)
        df_subs['dry'] = df_subs.index.map(lambda x: df_lows.loc[x, 'dry']
                                            if x in df_lows.index.values else 1)

        ## converts dfs to matrixes
        mat_lows  = res_base.fill_grid(df_subs.dry, fill_value=-2)

        fig, axes = plt.subplots(ncols=3, sharey=True)
        axe       = axes.ravel()
        # cm        = plt.get_cmap('coolwarm')
        cm        = mpl.colors.ListedColormap(['black', 'blue', 'green', 'gray'])
        # axe       = axes.ravel()
        for i, ax in enumerate(axes.ravel()):
            im        = axe[i].imshow(mat_lows.reshape(self.nrows, self.ncols), cmap=cm)

            axe[i].set(title='SLR: {}'.format(0.0),
                       xlabel='# Wetlands: {}'.format(
                       df_lows[((df_lows['dry']==0) | (df_lows['dry']==1))].shape[0]),
                       )

        cbar_ax   = fig.add_axes([0.945, 0.18, 0.025, 0.62])

        cbar      = fig.colorbar(im, cbar_ax, spacing='proportional')
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.set_xlabel('Legend', fontsize=12, labelpad=10)
        cbar.ax.xaxis.set_label_position('top')
        for j, lab in enumerate(['inactive','wetland (CCAP)','wetland(nonCCAP)','upland']):
            cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.75), rotation=90)

        fig.subplots_adjust(left=0.05, right=0.915, wspace=0.25, hspace=0.25)
        plt.show()

    def optimize(self, increment=1):
        """ Maximize the percent correctly identiified """
        optimal     = []
        cutoff      = []
        n_correct   = []
        n_incorrect = []
        for test in np.arange(np.floor(np.nanmin(self.mat_dtw_sum)), 0, increment):
            # print('Cutoff: {}'.format(test))
            result, n_corr, n_incorr = self.indicator_all(test)
            optimal.append(result)
            n_correct.append(n_corr)
            n_incorrect.append(n_incorr)
            cutoff.append(test)

        results = list(zip(optimal, n_correct, n_incorrect, cutoff))
        optimal   = (max(results, key=lambda item:item[0]))
        # sorted_by_incorrect = sorted(results, reverse=True, key=lambda item: item[0])

        print ('Performance?  {}'.format(round(optimal[0], 4)))
        print ('Correct: {}'.format(optimal[1]))
        print ('Incorrect: {}'.format(optimal[2]))
        print ('Cutoff: {}'.format(optimal[3]))

    def plot_indicators(self, cut=-2500):
        """
        Make histogram of cells, show their locations
        Show cells above cutoff and wetland locations
        """
        mat_highest = self.indicator_all(cut, show=True)
        mask = np.isnan(self.mat_dtw_sum)
        bins = np.linspace(-5000, 0, 21)

        fig, axes = plt.subplots(ncols=2, nrows=2)
        axe       = axes.ravel()
        axe[0].hist(self.mat_dtw_sum[~mask], bins=bins)
        axe[1].imshow(self.mat_dtw_sum.reshape(74, -1), cmap=plt.cm.jet)
        axe[2].imshow(mat_highest.reshape(74, -1), cmap=plt.cm.jet)
        axe[3].imshow(self.mat_wets.reshape(74, -1), cmap=plt.cm.jet)

        titles = ['Hist of summed negative dtws', 'Total annual DTW',
                  'Locs of cells above dtw cutoff: {}'.format(cut), 'Locs of wetlands cells']
        for i, t in enumerate(titles):
            axe[i].set_title(t)

        fig.subplots_adjust(top=0.95, hspace=0.5)

        plt.show()

    ### probably useless
    def indicator_all(self, cutoff=-2500, show=False):
        """ Compare results form 'developing indicator' to actual CCAP wetlands """
        ## get just highest cells (longest time at lowest DTW)
        mat_indicator = np.where(self.mat_dtw_sum <= cutoff, np.nan, self.mat_dtw_sum)
        if show:
            print ('Indicated cells: {}'.format(np.count_nonzero(~np.isnan(mat_indicator))))
            print ('Wetland cells: {}'.format(np.count_nonzero(~np.isnan(self.mat_wets))))
            return mat_indicator

        mat_ind  = mat_indicator[~self.mask_wet & ~np.isnan(mat_indicator)]

        df_ind = pd.DataFrame(mat_indicator.reshape(-1)).dropna()
        df_wet = pd.DataFrame(self.mat_wets.reshape(-1)).dropna()

        ## get count of how many correct
        count_correct   = (len(mat_ind))
        ## get count of how many incorrect
        count_incorrect = (len(mat_indicator[~np.isnan(mat_indicator)]) - len(mat_ind))
        # print ('Correct: {}'.format(count_correct))
        # print ('Incorrect: {}'.format(count_incorrect))

        # performance     = (float(count_correct) / float(count_incorrect)  * 100.
        performance     = (count_correct - count_incorrect) / float(np.count_nonzero(~mask_wet)) * 100
        # print ('Percent corretly identified: {} %\n'.format(round(performance, 3)))
        return (performance, count_correct, count_incorrect)

    ### probably useless
    def dtw_wet_all(self, transpose=False):
        """ Separate ALL dtw / cells / times by wetlands/nonwetland """
        ## convert to dtw for subsetting
        df_dtw  = pd.DataFrame(self.mat_dtw)

        df_dtw.index = range(10000, 13774)
        df_wets = df_dtw[df_dtw.index.isin(self.df_wets.Zone)]
        df_drys = df_dtw[~df_dtw.index.isin(self.df_wets.Zone)].dropna()
        # print (df_wets.shape)
        # print (df_drys.shape)
        if transpose:
            return df_wets.T, df_drys.T
        return df_wets, df_drys

    ### probably useless
    def comp_histograms(self, kind='avg', plot=True):
        """ Plot dtw hists of ann avg (dtw_wet_ann_avg) or all (dtw_wet_all) """
        if kind == 'avg':
            df_wets_all, df_drys_all = self.dtw_wet_avg_ann()
            df_wets = df_wets_all['0.0']
            df_drys = df_drys_all['0.0']
            xlab    = 'Ann avg dtw'
        else:
            print ('this will take a min (& isnt very useful)')
            df_wets, df_drys = self.dtw_wet_all(transpose=True)
            df_wets.dropna(axis=1, inplace=True)
            df_drys.dropna(axis=1, inplace=True)
            xlab    = 'dtw (all)'

        fig, axes        = plt.subplots(ncols=2, figsize=(10,6))
        axe              = axes.ravel()
        bins             = np.arange(0, 5.5, 0.5)

        if plot:
            axe[0].hist(df_wets, bins=bins)
            axe[1].hist(df_drys, bins=bins)
            titles = ['Wetlands', 'Uplands']
            for i, t in enumerate(titles):
                axe[i].set_title(t)
                axe[i].set_xlabel(xlab)
            axe[0].set_ylabel('Frequency')
            # fig.subplots_adjust(right=0.92, wspace=0.175, hspace=0.35)
            fig.subplots_adjust(bottom=0.15)
            plt.show()
        else:
            print (df_wets.describe())
            print (df_drys.describe())

    ### probably useless
    def dtw_wet_avg_ann(self):
        """ Subset dtw df (all cells, annual average to just wetland cells """

        ## avg annual dtw
        df_dtw  = dtw(self.path_res).df_year
        df_dtw.columns = [str(col) for col in df_dtw.columns]

        df_wets = df_dtw[df_dtw.index.isin(self.df_wets.Zone)]
        df_drys = df_dtw[~df_dtw.index.isin(self.df_wets.Zone)].dropna()
        return df_wets, df_drys


PATH_res = op.join(op.expanduser('~'), 'Google_Drive',
                    'WNC', 'Wetlands_Paper', 'Results_Default')
res      = Wetlands(PATH_res, dtw_inc=0.01, hrs_per=50, z_thresh=3.75)
# res.optimize(increment=10)
# res.make_indicator(masked=True, seasonal=False)
# res.apply_indicator(seasonal=False)

## nonseasonal indicators: dtw < 0.05; hrs_thresh>4443 --------- best
## seasonal indicators   : dtw < 0.17; hrs_thresh > 1211

# res.find_cells()
res.plot_wets_drys()
