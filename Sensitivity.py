from utils.AnalysisObjs import *

class Sensitivity(res_base):
    def __init__(self, path_res):
        super(Sensitivity, self).__init__(path_res)
        self.path_all_picks = self._get_all_res()[0:-1]

    def totals(self, var='run'):
        """ Total Runoff for Whole Year """
        ### get this going for all 3 variables, subplot for each? might be too cluttered
        # make ids for each dir by hand, following chart in swmm
        # 'Results_05_21' : 'D'
        # use IDS instead of dates; for now just dummies
        # will have to set up some conversions
        var_map   = {'run' : 'Runoff Rate (CMS)',
                     'inf' : 'Infiltration Rate (M/D)',
                      'et' : 'Volume (CM)'}

        ids       = ['Default', 'U1H', 'U1L']
        arr_all   = np.ones([len(self.slr_sh), len(ids)])

        fig, axes = plt.subplots()
        markers   = ['D', 's', '*']

        for i, resdir in enumerate(self.path_all_picks):
            dict_var = self._load_swmm(var, path=resdir)
            y = [] # store sums and plot i times only; make legends better
            # use slr to maintain order
            for j, slr in enumerate(self.slr):
                y.append(np.nansum(dict_var[str(slr)][3696:-698, :, :]))
                arr_all[j, i] = y[-1]
            axes.scatter(self.slr, y, marker=markers[i],
                              label=ids[i], color=self.colors[i])

        axes.legend(loc='best', frameon=True, shadow=True, facecolor='w',
                                                                numpoints=1)
        axes.set_ylabel(var_map[var])
        axes.set_xlabel('SLR (m)')
        axes.set_xticks([float(slr) for slr in self.slr])
        axes.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        axes.yaxis.grid(True)

        df_all = pd.DataFrame(arr_all, index=self.slr, columns=ids)
        print df_all.head()
        return df_all

    def _get_all_res(self):
        """ Get Paths to all Result Directories """
        path_parent   = op.dirname(self.path)
        path_pickdirs = [op.join(path_parent, resdir, 'Pickles') for resdir in
                       os.listdir(path_parent) if resdir.startswith('Results_')]
        return path_pickdirs




# could be a self contained object, but need slr and save stuff

PATH_res       = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-21')
sensitivityObj = Sensitivity(PATH_res)
sensitivityObj.totals()
plt.show()