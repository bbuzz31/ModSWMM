from utils.AnalysisObjs import *
from collections import OrderedDict

class Sensitivity(res_base):
    def __init__(self, path_res):
        super(Sensitivity, self).__init__(path_res)
        self.results = self._get_all_res()

    def totals(self, var='run'):
        """ Total Runoff for Whole Year """
        ### get this going for all 3 variables, subplot for each? might be too cluttered
        # make ids for each dir by hand, following chart in swmm
        # 'Results_05_21' : 'D'
        # use IDS instead of dates; for now just dummies
        # will have to set up some conversions
        var_map   = {'run' : 'Runoff Rate (CMS)',
                     'inf' : 'Infiltration Rate (M/D)',
                    'evap' : 'Evaporation Volume (CM)'}

        ids = []
        arr_all   = np.ones([len(self.slr_sh), len(self.results)])
        markers   = [".",",","o","v","^","<",">","1","2","3","4","8","s","p",
                    "h","H","+","D","d","|","_",".",",","o","v","^","<",
                    ">","1","2","3","4","8","s","p","h","H","+","D","d",
                    "|","_"]
        colors    = self._make_colors(len(self.results))
        fig, axes = plt.subplots()

        for i, (ID, resdir) in enumerate(self.results.items()):
            dict_var = self._load_swmm(var, path=resdir)
            y        = []
            colors   = []

            # store sums and plot i times only; make legends better
            # use slr to maintain order
            for j, slr in enumerate(self.slr):
                y.append(np.nansum(dict_var[str(slr)][3696:-698, :, :]))
                arr_all[j, i] = y[-1]
            ids.append(ID)
            if ID == "Default":
                marker = '*'
            else:
                marker = markers[i]
            axes.scatter(self.slr, y, marker=marker, label=ID, color='b')#color=colors[i])

        axes.legend(loc='best', frameon=True, shadow=True, facecolor='w',
                                                                numpoints=1)
        axes.set_ylabel(var_map[var])
        axes.set_xlabel('SLR (m)')
        axes.set_xticks([float(slr) for slr in self.slr])
        axes.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        axes.yaxis.grid(True)
        axes.set_ylim([500000, 1300000])
        axes.legend(bbox_to_anchor=(1.1, 1.00))

        fig.set_label('{}_sensitivity'.format(var))

        df_all = pd.DataFrame(arr_all, index=self.slr, columns=ids)

        # print df_all.head()
        return df_all

    def _get_all_res(self):
        """ Get Paths to all Result Directories """
        path_parent   = op.dirname(self.path)
        res_dict      = OrderedDict()
        for resdir in os.listdir(path_parent):
            if resdir.startswith('Results_'):
                if 'M1' in resdir or 'S1' in resdir:
                    continue
                res_id       = resdir.split('_')[1]
                path_pickdir = op.join(path_parent, resdir, 'Pickles')
                res_dict[res_id] = path_pickdir
        # path_pickdirs = [op.join(path_parent, resdir, 'Pickles') for resdir in
                    #    os.listdir(path_parent) if resdir.startswith('Results_')]
        return res_dict

    def _make_colors(self, n):
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]



# could be a self contained object, but need slr and save stuff

PATH_res       = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_Default')
sensitivityObj = Sensitivity(PATH_res)

sensitivityObj.totals('inf')
# sensitivityObj.totals('evap')
# sensitivityObj.totals('run')
# savefigs(PATH_res)
plt.show()
