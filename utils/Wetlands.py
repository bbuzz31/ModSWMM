from AnalysisObjs import *
import seaborn as sns

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
    return df_wets, df_drys

    print (df_wets.shape)
    print (df_drys.shape)

def developing_indicator(mat_dtw, path_data, cutoff=-2500):
    """ Indicator built from longest time at lowest DTW """
    mat_dtw_neg = mat_dtw * -1
    mat_dtw_sum = mat_dtw_neg.sum(axis=1)
    mask = np.isnan(mat_dtw_sum)
    bins = np.linspace(-5000, 0, 21)
    # plt.hist(mat_dtw_sum[~mask], bins=bins)

    ## show all
    # plt.imshow(mat_dtw_sum.reshape(74, -1), cmap=plt.cm.jet)
    # fig, axes = plt.subplots(ncols=2)
    # axe       = axes.ravel()

    ## show just highest cells
    mat_highest = np.where(mat_dtw_sum <= cutoff, np.nan, mat_dtw_sum)
    # axe[0].imshow(mat_highest.reshape(74, -1), cmap=plt.cm.jet)

    ## show just wetland cells
    df_subs = pd.read_csv(op.join(path_data, 'SWMM_subs.csv'))
    ## convert to full grid
    ser_landcover = df_subs.set_index('Zone').loc[:, 'Majority']
    mat_landcover = res_base.fill_grid(ser_landcover, fill_value=-1)
    mat_wetlands  = np.where(((mat_landcover < 13) | (mat_landcover > 18)), np.nan, mat_landcover)
    # axe[1].imshow(mat_wetlands.reshape(74, -1), cmap=plt.cm.jet)
    # sns.heatmap(mat_dtw)

    return (mat_highest, mat_wetlands)
    # print ('Indicated cells: {}'.format(np.count_nonzero(~np.isnan(mat_highest))))
    # print ('Wetland cells: {}'.format(np.count_nonzero(~np.isnan(mat_wetlands))))
    # plt.show()

def testing_indicator(mat_dtw, path_data, cutoff):
    """ Compare results form 'developing indicator' to actual CCAP wetlands """
    mat_indicator, mat_wetlands = developing_indicator(mat_dtw, path_data, cutoff)
    df_ind = pd.DataFrame(mat_indicator.reshape(-1)).dropna()
    df_wet = pd.DataFrame(mat_wetlands.reshape(-1)).dropna()

    ## get count of how many correct
    count_correct   = (sum(df_wet.index.isin(df_ind.index)))
    ## get count of how many incorrect
    count_incorrect = (len(df_wet) - count_correct)

    ## performance
    performance     = (count_correct - count_incorrect) / float(len(df_wet)) * 100
    # print ('Percent correctly identified: {} %.'.format(round(performance, 3)))

def optimize(mat_dtw, path_data):
    """ Maximize the percent correctly identiified """
    optimal = 0
    cutoff  = 0
    for test in np.arange(-5500, 0, 0.2):
        result = (testing_indicator(mat_dtw, path_data, cutoff))
        if result > optimal:
            optimal = result
            cutoff  = test
    print ('hi')
    print (optimal, cutoff)

PATH_res = op.join(op.expanduser('~'), 'Google_Drive',
                    'WNC', 'Wetlands_Paper', 'Results_Default')
Result   = res_base(PATH_res)


# print (dtw(PATH_res).plot_area_hours())
# print (dtw(path_res).df_year.head())
# comp_histograms(plot=False)
mat_dtw = make_dtw(PATH_res, Result.path_data, 0.0)
# dtw_wetlands2(mat_dtw, Result.path_data)
# developing_indicator(mat_dtw, Result.path_data)
# testing_indicator(mat_dtw, Result.path_data)
optimize(mat_dtw, Result.path_data)
