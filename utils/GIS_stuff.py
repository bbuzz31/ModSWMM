""" Functions for Dealing with Retarded ArcMAP Shit """
import os.path as op
import numpy as np
import pandas as pd

class XY(object):
    def __init__(self, path_coupled):
        self.path_data = op.join(path_coupled, 'Data')
        self.df_xy     = pd.read_csv(op.join(self.path_data, 'Grid_XY.csv'),
                                                          index_col='Zone')
        # add MF ks & uzf bound

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



PATH   = op.join('/', 'Users', 'bb', 'Google_Drive', 'WNC', 'Coupled')
xy_obj = XY(PATH)
print xy_obj.add_Ks()
