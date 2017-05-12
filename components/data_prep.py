'''
Create Junctions and Stream Reach Tables for SWMM
Create IBND, UZF_IBND, and UZF Parameters

Created: 2016-12-28
Updated: 2017-01-16, added mf_heads to table
         2017-01-22, changed paths to coupled dir
INCLUDES INSTRUCTIONS ON HOW TO MAKE TABLES; OUTFALLS MADE ENTIRELY HERE
-- once this is done, modswmm proceeds to just use the tables
'''
from BB_PY import BB
import os
import os.path as op
import time

import flopy.utils.formattedfile as ff

import numpy as np
import pandas as pd

# ****************************   EDIT JUNCTIONS   **************************** #
def junctions(junctions_raw):
    keep_columns = ['ROW', 'COLUMN_', 'land_elev', 'invert_z', 'max_depth',
                    'CENTROID_X', 'CENTROID_Y', 'NODE_Z', 'MEAN', 'MEDIAN']

    df_junc_int = junctions_raw.rename(columns={'MODEL_TOP' : 'land_elev'})

    outfalls = [(7, 20, 1.578167, '', '', 404343.408848, 4.072477e6,
                        1.578167, 1.578167, 1.578167),
               (65, 3, 0.000936, '', '', 404905.236212, 4.060402e6,
                       0.000936, 0.000936, 0.000936)]
    df_outs  = pd.DataFrame.from_records(outfalls, index=[326, 3267], columns=keep_columns)

    df_junctions = pd.concat([df_junc_int, df_outs])


    df_junctions['invert_z']  = df_junctions['MEDIAN'].apply(lambda x: round(x, 6))
    df_junctions['max_depth'] = 0
    keep_columns = ['ROW', 'COLUMN_', 'land_elev', 'invert_z', 'max_depth',
                    'CENTROID_X', 'CENTROID_Y', 'NODE_Z', 'MEAN', 'MEDIAN']
    df_junctions.ROW = df_junctions.ROW.astype(int)
    df_junctions.COLUMN_ = df_junctions.COLUMN_.astype(int)
    df_junctions = df_junctions[keep_columns]
    df_junctions.index.name='Zone'

    return df_junctions.sort_index()
# ******************************   EDIT LINKS   ****************************** #
def links(links, juncts):
    """ needs df_junctions to determine conduit slope and set to 0 if negative"""
    # remove extra columns
    elev_col = juncts['MEDIAN'].name
    # set invert elevations according to land surface - depth for to/from nodes
    node_z   = pd.DataFrame(juncts[elev_col])
    # NOTE junctions now contain end points instead of outfalls (03-19-2017)
    temp     = pd.merge(links, node_z, left_on='From Node',
                                                   right_index=True, how='left')

    df_links = pd.merge(temp, node_z, left_on='To Node',
                                                    right_index=True, how='left')

    df_links = df_links.rename(columns={'From Node'   : 'From_Node',
                                        'To Node'     : 'To_Node',
                                        elev_col+'_x' : 'InElev',
                                        elev_col+'_y' : 'OutElev'})

    # set wrong directional flow to 0 by adding an offset
    df_links['InOffset'] = np.where(df_links['InElev'] < df_links['OutElev'],
                                    df_links['OutElev'] - df_links['InElev']
                                    + 0.000001, 0)

    df_links['conv2cms'] = df_links.Geom2_width * 200 / (24 * 3600)
    # DEPRECATED; set wrong directions to 0
    def reverse_flow_dir(df_links):
        """ reverse flow direction if elevation dictates it """
        reverse = df_links['To_Node'].where(df_links['InElev'] < df_links['OutElev']).dropna()
        reverse_loc = reverse.index.tolist()
        # reverse to and from nodes
        df_links.loc[reverse_loc, 'To_Node'] = df_links.loc[reverse_loc, 'From_Node']
        df_links.loc[reverse_loc, 'From_Node'] = reverse.astype(int)
        # reverse in and out elev
        reversed_outs = df_links.OutElev.loc[reverse_loc]
        df_links.loc[reverse_loc, 'OutElev'] = df_links.loc[reverse_loc, 'InElev']
        df_links.loc[reverse_loc, 'InElev']  = reversed_outs

    # # set to 0 if negative; swmm will do it regardless
    # df_links = df_links.applymap(lambda x: 0 if x < 0 else x)
    chg = ['From_Node', 'To_Node', 'Geom2_width']
    df_links[chg] = df_links[chg].apply(lambda x: pd.to_numeric(x,downcast='signed'))
    df_links.index = df_links.index.astype(int)
    return df_links
    # BB.print_Df(df_links)
# ****************************   EDIT OUTFALLS   ***************************** #
def outfalls(swmm, junctions_raw):
    # THIS WILL GET STREAM END POINTS AND MISC [3627, 326, 483]

    # active zones from SWMM grid and WNC (these are removed from SWMM grid)
    zones = swmm.index.tolist() + junctions_raw.index.tolist()

    # find where SWMM flow direction to inactive cells, append to list
    count = 0
    df_outfalls = df_SWMM[~df_SWMM.Flow_ID.isin(zones)]

    ''' update to grid 2017-03-30 '''
    # add the two mining pit locations to outfalls - they mess up grid
    pit = [1965, 1966]
    df_outfalls = pd.concat([df_outfalls, df_SWMM.loc[pit, :]]).dropna(axis=1)

    ## sorting
    colnames    = ['ROW', 'COLUMN_', 'Accum', 'MODEL_TOP', 'CENTROID_X', 'CENTROID_Y']
    df_outfalls = df_outfalls[colnames]
    df_outfalls.sort_values(['ROW', 'COLUMN_'], inplace=True)
    # drop the two stream endpoints - they will be set as junctions
    # careful with modflow grid
    #BB.print_Df(df_outfalls)
    return df_outfalls.drop([326, 3267])
# ***************************   CREATE STORAGE   ***************************** #
def storage(swmm, junctions_raw):
    # lake sherwood cells (1970, 2022, col,row = 32|39, 33|40) to storage units
    # seepage loss = steady state infiltration rate, specified in MF?

    storage = [1970, 2022]

    df_lakes = swmm[swmm.index.isin(storage)].rename(columns={'COLUMN_': 'COLUMN'})
    keep = ['COLUMN', 'ROW', 'Accum', 'MODEL_TOP', 'CENTROID_X', 'CENTROID_Y']


    return df_lakes[keep]
# ***************************   CREATE INFLOWS   ***************************** #
def inflows(df_junctions, links_raw):
    # need to give a constant (tidal) inflow to all the junctions
    df_from = links_raw[links_raw['From Node'].isin(df_junctions.index)].set_index('From Node')
    df_to   = links_raw[links_raw['To Node'].isin(df_junctions.index) &
                       ~links_raw['To Node'].isin(links_raw['From Node'])].set_index('To Node')
    s_geom  = pd.concat([df_from, df_to])['Geom2_width'].sort_index()
    s_conv  = (s_geom * 200 / (3600*24)).rename('conv2cms')
    df_inflows = pd.concat([s_geom, s_conv], axis=1)
    df_inflows.index.name='Node'
    return df_inflows
# ******************************   EDIT SUBS   ******************************* #
def subs(swmm, soils, outfalls, storage):
    """
    NEED OUTFALLS & STORAGE TO RUN'
    subset of SWMM_grid_Active - this makes them active in MF and not WNC;
    also remove outfalls and storage units
    Added soil groups - 2017-04-03
    """
    remove = outfalls.index.tolist() + storage.index.tolist()
    # remove chd endpoints
    endpoints = [326, 3267]
    remove.extend(endpoints)
    # only subcatchments - used for subcatchs, areas, gw,
    df_subs = swmm[~swmm.index.isin(remove)]

    # make column names match SWMM names
    rename = {'COLUMN_' : 'COLUMN', 'Flow_ID' : 'Outlet', 'MEAN_IS' : 'perImperv',
              'Slope' : 'perSlope', 'Manning' : 'N_Perv', 'dep_stor' : 'S_Perv',
              'MODEL_TOP' : 'Esurf'}

    df_subs = df_subs.rename(columns=rename)
    ##### SUBCATCHMENTS
    keep = ['ROW', 'COLUMN', 'Outlet', 'perImperv', 'perSlope',
            'N_Perv', 'S_Perv', 'Esurf', 'Min_Inf', 'CENTROID_X', 'CENTROID_Y']


    ###### SOILS
    ### df created in arcgis by joining to modflow grid
    # join soils to df_subs
    df_soils = soils['Soil_Group_Maj']
    df_subs = df_subs.join(df_soils, how='left')
    df_null = df_subs.Soil_Group_Maj[df_subs.Soil_Group_Maj.isnull().values]
    # set null values to 7, same as surrounding area (1549 could be 6)
    df_subs.loc[df_null.index, 'Soil_Group_Maj'] = 7
    # 1, 2, 3, 4 = A, B, C, D; 5, 6, 7 = A/D, B/D, C/D
    def soil_map(group_int):
        """ Map hydrological soil groups to min infiltration values """
        # middle values; values = A, B, or C mm/hr; from soil group pg 97 swmm16
        # created from arc gis layer, which gives groups as ints
        A = 9.525; B = 5.715; C = 2.54; D = 0.635
        grp = int(group_int)
        if grp in (1, 5):
            return A
        elif grp in (2, 6):
            return B
        elif grp in (3, 7):
            return C
        else:
            return D
    df_subs['Min_Inf'] = df_subs.Soil_Group_Maj.apply(lambda x: soil_map(x))
    # only these are counted in the summary runoff calculations in swmm .rpt
    subs_to_node = df_subs[~df_subs.Outlet.isin(df_subs.index)]
    #print subs_to_node.head()
    #print len(subs_to_node[keep].index.tolist())

    return df_subs[keep]
# *******************************   MF_CHD   ******************************** #
def mf_chd(junctions, mf, chd_val=0.3048):
    # combine junctions and upstream and downstream location
    end_points = [326, 3267]
    chd_locs   = junctions.index.tolist() + end_points
    chd_val    = 0.3048

    # take chds from all raw
    df_chd                 = mf[mf.index.isin(chd_locs)]
    # format for MF
    keep                   = ['COLUMN_', 'ROW']
    df_chd                 = df_chd[keep]
    df_chd.loc[:, 'START'] = chd_val
    df_chd.loc[:, 'END']   = chd_val
    return df_chd
# *******************************   MF_ACTIVE  ******************************* #
def mf_active(mf, swmm, outfalls, CHD, ss_rate=0.0002783601):
    # add columns for IBND (bas package) and UZF_IBND
    df_grid                    = mf
    df_grid.loc[:, 'IBND']     = 0

    # remove outfalls from the active made from modflow and arc
    outfall_list  = outfalls.index.tolist()
    # remove mining cells from remove; they need to be active in MF
    [outfall_list.remove(pit) for pit in [1965, 1966]]

    df_ibnd = swmm[~swmm.index.isin(outfall_list)]
    # change 'IBND' to 1 where subs & storage units are (not outfalls)

    # set active to 1
    df_grid.set_value(df_ibnd.index.tolist(), 'IBND', 1)
    # set uzfbnd to 1 only where IBND is 1
    df_grid.loc[:, 'UZF_IBND'] = df_grid.IBND
    # change 'IBND' to 1 and 'UZF_IBND' to 0 where CHD is
    df_grid.set_value(CHD.index, 'IBND', 1)
    df_grid.set_value(CHD.index, 'UZF_IBND', 0)
    # set modeltop
    df_grid.set_value(df_grid.index.tolist(), 'MODEL_TOP', df_ibnd.MODEL_TOP)

    # SS UZF Parameters
    # get list of locations where ibnd is ON --- DOES NOT INCLUE STORAGE UNITS
    uzf_on = df_grid.where(df_grid.UZF_IBND>0).dropna()
    uzf_loc = uzf_on.index.tolist()
    # set UZF params for ibnd on
    df_grid.loc[uzf_loc, 'FINF']     = ss_rate
    df_grid.loc[uzf_loc, 'ET']       = ss_rate

    # set constant infiltration (seepage) at lake (storage units)
    # getting storage from difference between active - outfalls and subs # not
    storage = df_ibnd.loc[[1970, 2022], :]
    df_grid.set_value(storage.index.tolist(), ['FINF', 'ET'], ss_rate)
    df_grid.set_value(storage.index.tolist(), ['MODEL_TOP'], 0)

    # clean up
    # INACTIVE HEAD VALUE
    df_grid.MODEL_TOP.fillna(0, inplace=True)
    df_grid.fillna(0, inplace=True)

    #BB.print_Df(df_grid.drop(['Shape_Length', 'Shape_Area'], axis=1))
    return df_grid.drop(['Shape_Length', 'Shape_Area'], axis=1)
# *******************************   MF_PARAMS   ****************************** #
def mf_params(df_grid, df_KZ):
    # join mf output from modelmuse output
    # point here is just to add both grid and params into 1 dataframe

    return pd.concat([df_grid, df_KZ], axis=1)
# *********************************   WRITE  ********************************* #
def write_all():

    # SWMM
    # df_junctions.to_csv(op.join(PATH_data,'SWMM_junctions.csv'))
    # df_links.to_csv(op.join(PATH_data,'SWMM_links.csv'))
    # df_outfalls.to_csv(op.join(PATH_data,'SWMM_outfalls.csv'))
    df_storage.to_csv(op.join(PATH_data,'SWMM_storage.csv'))
    #df_subs.to_csv(op.join(PATH_data, 'SWMM_subs.csv'))
    # df_inflows.to_csv(op.join(PATH_data,'SWMM_inflows.csv'))
    #
    # #
    # # # MF
    # df_chd.to_csv(op.join(PATH_data, 'MF_CHD.csv'))
    # df_MF_final.to_csv(op.join(PATH_data, 'MF_GRID.csv'))
    print 'Overwrote .csvs in: ' + PATH_data

# ============================================================================ #

# *************************   INITIALIZE FILEPATHS  ************************** #
PATH_home         = op.expanduser('~')
PATH_root         = op.join(PATH_home, 'Google_Drive','WNC', 'Coupled', time.strftime('%b'))
PATH_data         = op.join(PATH_root, 'Data')
#mf_model          = op.join(PATH_root, 'Flopy', 'WNC_SS-Mac', 'WNC-SS')

### data
# export raw grid muse/flopy to arcgis
df_MF             = pd.read_csv(op.join(PATH_data, 'RAW_grid.csv'),
                                        index_col='Zone')
# create from geoswmm and arc model (i think)
df_SWMM           = pd.read_csv(op.join(PATH_data, 'RAW_swmm.csv'),
                                        index_col='Zone')

# from modelmuse - arc - csv (arc modelbuilder)
# Ks and bottoms
df_KZ             = pd.read_csv(op.join(PATH_data, 'MF_kz.csv'),
                                        index_col='ID')
# create from CHD file, minus the two ends
# get zs at stream by creating a new points shapefile at stream loc in each cell
# do this by hand, then spatial join, then manually put loc at 3 that aren't contained
df_junctions_raw  = pd.read_csv(op.join(PATH_data,'RAW_junctions.csv'),
                                        index_col='Zone')
# create by hand in excel following the junctions
df_links_raw      = pd.read_csv(op.join(PATH_data, 'RAW_links.csv'),
                                        index_col='Zone')

df_soils_raw      = pd.read_csv(op.join(PATH_data, 'RAW_soils.csv'),
                                        index_col='Zone')
# originally this df was df_links_raw; RAW_inflows.csv == RAW_links.csv
#df_inflows_raw    = pd.read_csv(op.join(PATH_data, 'RAW_inflows.csv'),
#                                        index_col='Zone')
# ============================================================================ #
reset_idx = True
write     = True

df_junctions = junctions(df_junctions_raw)
df_links = links(df_links_raw, df_junctions)
df_outfalls = outfalls(df_SWMM, df_junctions_raw)
df_storage  = storage(df_SWMM, df_junctions_raw)
df_inflows  = inflows(df_junctions, df_links_raw)
df_subs = subs(df_SWMM, df_soils_raw, df_outfalls, df_storage)
df_chd = mf_chd(df_junctions, df_MF)
df_grid = mf_active(df_MF, df_SWMM, df_outfalls, df_chd)
df_MF_final = mf_params(df_grid, df_KZ)


if 'reset_idx':
    # swmm.so 'get' needs a length of at least 3, so increase index of all
    all_Dfs = [df_junctions, df_links, df_outfalls, df_storage, df_subs, df_chd, df_MF_final]
    df_junctions.index   = df_junctions.index   + 10000
    df_links.From_Node   = df_links.From_Node   + 10000
    df_links.To_Node     = df_links.To_Node     + 10000
    df_outfalls.index    = df_outfalls.index    + 10000
    df_storage.index     = df_storage.index     + 10000
    df_subs.index        = df_subs.index        + 10000
    df_subs.Outlet       = df_subs.Outlet       + 10000
    df_inflows.index     = df_inflows.index     + 10000
    # print df_subs.head()
    # I dont think I have to change the MF ones
    # print df_chd.head()
    #print df_MF_final.head()
if write:
    write_all()
