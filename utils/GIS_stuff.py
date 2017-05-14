""" Functions for Dealing with Retarded ArcMAP Shit """
import BB
import os.path as op
import numpy as np
import pandas as pd
from collections import OrderedDict

class XY(object):
    def __init__(self, path_coupled):
        self.path       = path_coupled
        self.path_data  = op.join(self.path, 'Data')
        path_stor       = op.join('/', 'Volumes', 'BB_4TB', 'Thesis')
        self.path_res   = op.join(path_stor, 'Results_05-08')
        self.path_picks = op.join(self.path_res, 'Pickles')

        self.df_xy      = pd.read_csv(op.join(self.path_data, 'Grid_XY.csv'),
                                                          index_col='Zone')
        self.df_sys     = pd.read_pickle(op.join(self.path_picks, 'swmm_sys.df'))
        self.slr        = BB.uniq([float(slr.rsplit('_', 1)[1]) for slr
                                                    in self.df_sys.columns])
        self.ts_day     = self.df_sys.resample('D').first().index
        self.st         = '2011-12-01-00'
        self.end        = '2012-11-30-00'
    def add_Ks(self):
        """ Joins df with XY (UTM) w/ table of K vals & bounds in Arc (Ks). """
        keep_cols   = ['OBJECTID', 'ROW', 'COLUMN', 'IBND', 'UZF_IBND',
                       'MODEL_TOP', 'KXL1', 'KZL1']
        df_ks       = pd.read_csv(op.join(self.path_data, 'MF_GRID.csv'),
                                usecols= keep_cols, index_col='OBJECTID')

        df_xy_clean = self.df_xy[['POINT_X', 'POINT_Y']]

        df_join     =  df_ks.join(df_xy_clean)
        df_join.to_csv(op.join(self.path_data, 'XY_ks.csv'), sep=',')
        return df_join

    def grid_heads(self):
        """ Format and attach Avg Yearly Heads to Grid for Arc """
        df_heads = pd.DataFrame(index=self.df_xy.index)
        for slr in self.slr:
            fhd_pickle       = 'heads_{}.npy'.format(slr)
            pickle_file      = op.join(self.path_picks, fhd_pickle)
            arr_head         = np.load(pickle_file).reshape(-1, len(df_heads))
            # inactive to np.nan ;;; should do this in pickles
            arr_cln          = np.where(arr_head < -100, np.nan, arr_head)
            # truncate dates
            arr_yr           = arr_cln[154:519, :]
            df_heads[slr]    = arr_cln.mean(0)
        # add change columns
        df_heads['Chg_1_0']  = df_heads[1.0] - df_heads[0.0]
        df_heads['Chg_2_1']  = df_heads[2.0] - df_heads[1.0]
        # join to grid
        df_xy_heads = self.df_xy.join(df_heads).drop(['OBJECTID', 'ORIG_FID'], 1)
        # save to csv
        df_xy_heads.to_csv(op.join(self.path_data, 'XY_heads.csv'), sep=',')
        return df_xy_heads 


PATH   = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled')
xy_obj = XY(PATH).grid_heads()
