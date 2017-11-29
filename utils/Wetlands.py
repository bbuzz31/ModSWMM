from AnalysisObjs import *
import time
import seaborn as sns
import pickle

class Wetlands(res_base):
    def __init__(self, path_results):
        super(Wetlands, self).__init__(path_results)
        self.path_res    = path_results
        self.mat_dtw     = self._make_dtw()
        self.df_subs, self.mat_wetlands = self._ccap_wetlands()
        self.mask_wet    = np.isnan(self.mat_wetlands.reshape(-1))
        # highest values (closest to 0) = most hours with lowest DTW
        self.mat_dtw_sum = (self.mat_dtw * -1).sum(axis=1)

    def _ccap_wetlands(self):
        """ Get CCAP wetlands df and full matrix grid """
        df_subs = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'))

        ## convert to full grid
        ser_landcover = df_subs.set_index('Zone').loc[:, 'Majority']
        mat_landcover = res_base.fill_grid(ser_landcover, fill_value=-1)
        mat_wetlands  = np.where(((mat_landcover < 13) | (mat_landcover > 18)),
                                            np.nan, mat_landcover)
        return (df_subs, mat_wetlands)

    def _make_dtw(self, slr=0.0):
        """ dtw, all locations all times """
        mat_heads = np.load(op.join(self.path_picks, 'swmm_heads_grid_{}.npy'
                                                                .format(slr)))
        mat_z     = (np.load(op.join(self.path_data, 'Land_Z.npy'))
                                                .reshape(self.nrows, self.ncols))
        mat_dtw   = (mat_z - mat_heads).reshape(mat_heads.shape[0], -1)

        ### truncate init conditions & transpose
        mat_dtw_trunc = mat_dtw[self.ts_st:self.ts_end].T

        return mat_dtw_trunc

    def indicator_all(self, cutoff=-2500, show=False):
        """ Compare results form 'developing indicator' to actual CCAP wetlands """
        ## get just highest cells (longest time at lowest DTW)
        mat_indicator = np.where(self.mat_dtw_sum <= cutoff, np.nan, self.mat_dtw_sum)
        if show:
            print ('Indicated cells: {}'.format(np.count_nonzero(~np.isnan(mat_indicator))))
            print ('Wetland cells: {}'.format(np.count_nonzero(~np.isnan(self.mat_wetlands))))
            return mat_indicator

        mat_ind  = mat_indicator[~self.mask_wet & ~np.isnan(mat_indicator)]

        df_ind = pd.DataFrame(mat_indicator.reshape(-1)).dropna()
        df_wet = pd.DataFrame(self.mat_wetlands.reshape(-1)).dropna()

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

    def indicator_wets_only(self, dtw_inc=0.01, hrs_beg=4000, make_new=False):
        """ Make an indicator that captures over 90 % of CCAP wetlands """
        ### select wetland from all dtw information
        mat_wet_dtw    = self.mat_dtw[~self.mask_wet]
        mat_nonwet_dtw = self.mat_dtw[self.mask_wet]
        mat_dry_dtw    = mat_nonwet_dtw[~np.isnan(mat_nonwet_dtw)].reshape(
                                                -1, mat_nonwet_dtw.shape[1])

        if op.exists(op.join(self.path_data, 'dtw_hrs_wet_dry.npy')) and not make_new:
            mat_all = np.load(op.join(self.path_data, 'dtw_hrs_wet_dry.npy'))
            df_all  = pd.read_pickle(op.join(self.path_data, 'dtw_hrs_wet_dry.df'))
            print ('Loaded mat and df of dtw/hrs for wet and drylands')

        else:
            print ('Finding optimum criteria; will take a bit')
            dtw_tests = np.arange(0, 1, dtw_inc)
            hrs_tests = range(hrs_beg, self.mat_dtw.shape[1])
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
            mat_best       = mat_good[mat_good[:,4] >= 0.90]
            mat_best[:, 5] = mat_best[:,4] / float(mat_dry_dtw.shape[0])
            mat_best[:, 6] = mat_best[:,3] / (1 - (mat_best[:,5]))
            colnames = ['dtw_thresh', 'hrs_thresh', 'n_wet', 'perWet', 'n_dry', 'perDry', 'perRatio']
            df_all  = pd.DataFrame(mat_best, columns=colnames).sort_values(by='perRatio', ascending=False)

            np.save(op.join(self.path_data, 'dtw_hrs_wet_dry.npy'), mat_best)
            df_all.to_pickle(op.join(self.path_data, 'dtw_hrs_wet_dry.df'))

        ## do some cropping
        df_new = df_all[df_all.hrs_thresh > 7000]
        BB.print_all (df_new.head(250))

        return mat_all, df_all

    ### possibly deprecated since optimizing with indicator_wets
    def apply_indicator(self):
        """ Test the indicator developed using indicator_wets_only on drys """
        ### eventually return these from other func
        dtw_thresh = 0.301
        hrs_thresh = 7000

        ## get only dryland cells
        mat_nonwet_dtw = self.mat_dtw[self.mask_wet]
        mat_dry_dtw    = mat_nonwet_dtw[~np.isnan(mat_nonwet_dtw)].reshape(-1, mat_nonwet_dtw.shape[1])

        # apply conditions
        result = ((mat_dry_dtw <= dtw_thresh).sum(axis=1) > hrs_thresh).sum()
        print (result)

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
        axe[3].imshow(self.mat_wetlands.reshape(74, -1), cmap=plt.cm.jet)

        titles = ['Hist of summed negative dtws', 'Total annual DTW',
                  'Locs of cells above dtw cutoff: {}'.format(cut), 'Locs of wetlands cells']
        for i, t in enumerate(titles):
            axe[i].set_title(t)

        fig.subplots_adjust(top=0.95, hspace=0.5)

        plt.show()

    ### probably useless
    def dtw_wet_all(self, transpose=False):
        """ Separate ALL dtw / cells / times by wetlands/nonwetland """
        ## convert to dtw for subsetting
        df_dtw  = pd.DataFrame(self.mat_dtw)
        df_wet  = self.df_subs[((self.df_subs.Majority >= 13) & (self.df_subs.Majority < 19))].iloc[:, :3]

        df_dtw.index = range(10000, 13774)
        df_wets = df_dtw[df_dtw.index.isin(df_wet.Zone)]
        df_drys = df_dtw[~df_dtw.index.isin(df_wet.Zone)].dropna()
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
        df_wet  = self.df_subs[((self.df_subs.Majority >= 13) & (self.df_subs.Majority < 19))].iloc[:, :3]

        ## avg annual dtw
        df_dtw  = dtw(self.path_res).df_year
        df_dtw.columns = [str(col) for col in df_dtw.columns]

        df_wets = df_dtw[df_dtw.index.isin(df_wet.Zone)]
        df_drys = df_dtw[~df_dtw.index.isin(df_wet.Zone)].dropna()
        return df_wets, df_drys

start    = time.time()
PATH_res = op.join(op.expanduser('~'), 'Google_Drive',
                    'WNC', 'Wetlands_Paper', 'Results_Default')
res      = Wetlands(PATH_res)
# res.optimize(increment=10)
res.indicator_wets_only(dtw_inc=0.01, hrs_beg=4000, make_new=True)
# res.apply_indicator()
# res.minimum_drys(make_new=True)
end      = time.time()

print ('Elapsed time: {} min'.format(round((end-start)/60.), 4))
