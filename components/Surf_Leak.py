""" Surface Leakage Analysis """
import BB
import os
import os.path as op

import numpy as np
import pandas as pd

import flopy.utils.formattedfile as ff

class ss_base(object):
    def __init__(self, path_result):
        self.path       = path_result
        self.path_picks = op.join(path_result, 'Pickles')
        self.path_data  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC',
                                                           'Coupled', 'Data')
        self.path_fig   = op.join(self.path, 'Figures')
        self.df_swmm    = pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'),
                                                           index_col='Zone')
        self.subs       = self.df_swmm.index.values

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
        """ Load Pickled FHD Arrays into a Dictionary of SLR: arr"""
        dict_fhd         = {}
        for slr in self.slr:
            fhd_pickle    = 'heads_{}.npy'.format(slr)
            pickle_file   = op.join(self.path_picks, fhd_pickle)
            dict_fhd[slr] = (np.load(pickle_file))
        return dict_fhd

    def _load_uzf(self, kind=False):
        """ Load Pickled UZF Arrays """
        # ordered dict helps with plotting
        dict_uzf = OrderedDict()
        for var in ['surf_leak', 'uzf_rch', 'uzf_et', 'uzf_run']:
            pickle_files = [op.join(self.path_picks, pick_file) for pick_file in
                            os.listdir(self.path_picks) if pick_file.startswith(var)]
            tmp_dict = OrderedDict()
            for leak_mat in pickle_files:
                SLR = float(op.basename(leak_mat).split('_')[2][:-4])
                tmp_dict[SLR] = np.load(leak_mat)
            # list of dictionary of scenario (0.0) : 74x51x549 matrix
            dict_uzf[var] = tmp_dict
        try:
            return dict_uzf[kind]
        except:
            return dict_uzf

    def save_cur_fig(self):
        """ Saves the current fig """
        try:
            curfig = plt.gcf()
        except:
            raise ValueError('No Current Figure Available')
        if not op.isdir(self.path_fig):
            os.makedirs(self.path_fig)
        curfig.savefig(op.join(self.path_fig, curfig.get_label()))

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

class surf_leak(ss_base):
    def __init__(self, path_result):
        super(surf_leak, self).__init__(path_result)

    def leaking(self):
        """
        Get always dry, always wet, and sometimes leaking locations.
        Return a list with 3 dictionarys and one data frame.
            Dictionarys have keys of SLR
            DF has number of cells dry, wet, leaky, and a percent of always wet.
        """
        ### always dry
        dry_dict = {}
        wet_dict = {}
        sometimes_dict = {}

        for slr, mat in self._load_uzf['surf_leak'].items():
            print slr
            print mat.shape
            return
            ### always dry
            # sum through time
            dry     = mat.sum(axis=2)
            # if leak is < 0 it experienced leakage; gets a 1 if dry
            dry_bin = np.where(dry < 0, 0, 1)
            dry_ser = pd.Series(dry_bin.flatten())
            dry_ser.index = dry_ser.index+10001
            always = dry_ser[dry_ser > 0]
            dry_dict[slr] = always.index.tolist()
            ### always wet
            wet_locs = np.ones([mat.shape[0], mat.shape[1]])
            for i in range(mat.shape[2]):
                # iterate through time, if leak multiply 1, else multiply by 0
                wet_locs[:, :] *= np.where(mat[:, :, i] < 0, 1, 0)
            wet_bin = np.where(wet_locs > 0, 1, 0)
            wet_ser = pd.Series(wet_bin.flatten())
            # careful; use 10001 to properly adjust for modflow 1 index
            wet_ser.index = wet_ser.index+10001
            always = wet_ser[wet_ser > 0]
            wet_dict[slr] = always.index.tolist()
            ### sometimes

        vs   = np.arange(10001, 10001+(74*51), 1)
        total_cells = 1459
        num_mat  = np.ones([len(self.slr), 4])
        for i, slr in enumerate(self.slr):
            leakage = vs[(np.in1d(vs, np.hstack([dry_dict[slr], wet_dict[slr]]),
                                                                  invert=True))]
            sometimes_dict[slr] = leakage
            #print leaking
            num_mat[i][0] = slr
            num_mat[i][1] = len(dry_dict[slr])
            num_mat[i][2] = len(wet_dict[slr])
            num_mat[i][3] = len(leakage)

        wet_df = pd.DataFrame(num_mat, columns=['SLR', 'n_cells_dry',
                        'n_cells_wet', 'n_cells_leaking']).set_index('SLR')

        wet_df['%_always_wet'] = wet_df.n_cells_wet / total_cells
        #leaky_df['%_somtimes_wet'] = leaky_df.n_cells_leaking / total_cells

        return [dry_dict, wet_dict, sometimes_dict, wet_df]

    def plot_2d_leaks(self, day=0):
        """
        Plots all locs that leak at specific day, all SLR.
        Can use to to compare leak locations with heads/land surface in fhd
        UZF grid will include [11965, 11966, 11970, 12022] but not 13267
        """
        #uzf_base(self.path).plot_uzf_ts()
        fig, axes         = plt.subplots (ncols=len(self.slr_sh), figsize=(16,9))
        title             = fig.suptitle('Locations Leaking at Day {}'.format(day))
        axe               = axes.ravel()
        for i, slr in enumerate(self.slr_sh):
            mat_leak      = np.load(op.join(self.path_picks, 'surf_leak_{}.npy'.format(slr)))
            count_leaking = np.count_nonzero(mat_leak[day, :,:])
            mat_leak_bin  = np.where(mat_leak[day, :, :] < 0, 1, 0)
            df_leak_bin   = pd.DataFrame(mat_leak_bin.flatten(),
                                    index=range(10001, 10001+mat_leak_bin.size))
            im            = axe[i].imshow(mat_leak_bin, cmap=plt.cm.jet)
            axe[i].axis('off')
            axe[i].set(title='SLR {} (m)'.format(slr))
            axe[i].title.set_size(11)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.175, 0.025, 0.6], xlabel='Leaking (y/n)')
        fig.colorbar(im, cbar_ax, spacing='proportional')
        fig.set_label(title.get_text())

        return fig

    def ss_heads(self):
        """ Pull Steady State Heads from FHD. Return SS Heads (arr 74 * 51) """
        fhd_file = [f for f in os.listdir(self.path) if f.endswith('fhd')][0]
        head_file = op.join(self.path, fhd_file)
        try:
            hds  = ff.FormattedHeadFile(head_file, precision='single')
        except:
            hds  = ff.FormattedHeadFile(head_file, precision='double')
        return hds.get_alldata(mflay=0)[0]

    def ss_dtw(self):
        arr_heads = self.ss_heads()
        arr_z     = np.load(op.join(self.path_data, 'Land_Z.npy')).reshape(74, 51)
        arr_dtw   = arr_z - arr_heads
        df        = pd.DataFrame({'heads' : arr_heads.reshape(-1),
                                      'z' : arr_z.reshape(-1),
                                    'dtw' : arr_dtw.reshape(-1)},
                                    index=range(3774))
        df_active  = df[df.z > 0]
        df_dtw_neg = df_active[df_active.dtw < 0]
        print df_active.describe()
        print 'Count of heads > land: {}'.format(len(df_dtw_neg))

PATH_ss  = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled', 'May_SS', 'MF')
PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-08')
# surf_leak(PATH_ss).ss_dtw()
surf_leak(PATH_res).leaking()
