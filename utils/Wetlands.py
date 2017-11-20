from AnalysisObjs import *

def dtw_wetlands(path_data, path_result):
    """ Subset dtw df (all cells, annual average to just wetland cells """
    df      = pd.read_csv(op.join(path_data, 'SWMM_subs.csv'))
    df_wet  = df[((df.Majority >= 13) & (df.Majority < 19))].iloc[:, :3]
    df_dtw  = dtw(path_result).df_year
    ## convert col names to string
    df_dtw.columns = [str(col) for col in df_dtw.columns]
    df_wets = df_dtw[df_dtw.index.isin(df_wet.Zone)]
    df_drys = df_dtw[~df_dtw.index.isin(df_wet.Zone)].dropna()
    return df_wets, df_drys

def comp_histograms(plot=True):
    df_wets, df_drys = dtw_wetlands(Result.path_data, PATH_res)
    fig, axes        = plt.subplots(ncols=2, figsize=(10,6))
    axe              = axes.ravel()
    bins = np.arange(0, 5.5, 0.5)
    if plot:
        axe[0].hist(df_wets['0.0'], bins=bins)
        axe[1].hist(df_drys['0.0'], bins=bins)
        plt.show()
    else:
        print (df_wets['0.0'].describe())
        print (df_drys['0.0'].describe())

def make_dtw(path_res, path_data, slr):
    """ dtw, all locations all times? """
    mat_heads = np.load(op.join(path_res, 'Pickles',
                                    'swmm_heads_grid_{}.npy'.format(slr)))
    mat_z     = np.load(op.join(path_data, 'Land_Z.npy')).reshape(74, 51)
    mat_dtw   = (mat_z - mat_heads).reshape(mat_heads.shape[0], -1)
    return mat_dtw.T

def dtw_wetlands2(mat_dtw, path_data):
    """ Separate ALL dtw / cells / times by wetlands/nonwetland """
    ## convert to dtw for subsetting
    df_dtw  = pd.DataFrame(mat_dtw)

    df_subs = pd.read_csv(op.join(path_data, 'SWMM_subs.csv'))
    df_wet  = df_subs[((df_subs.Majority >= 13) & (df_subs.Majority < 19))].iloc[:, :3]

    df_dtw.index = range(10000, 13774)
    df_wets = df_dtw[df_dtw.index.isin(df_wet.Zone)]
    df_drys = df_dtw[~df_dtw.index.isin(df_wet.Zone)].dropna()
    print (df_wets.shape)
    print (df_drys.shape)


PATH_res = op.join(op.expanduser('~'), 'Google_Drive',
                    'WNC', 'Wetlands_Paper', 'Results_Default')
Result   = res_base(PATH_res)


# print (dtw(PATH_res).plot_area_hours())
# print (dtw(path_res).df_year.head())
# comp_histograms(plot=False)
mat_dtw = make_dtw(PATH_res, Result.path_data, 0.0)
dtw_wetlands2(mat_dtw, Result.path_data)
