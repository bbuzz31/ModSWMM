"""
Run Analysis
05.21.2017
"""
import os
import os.path as op
import matplotlib as mpl

from utils.AnalysisObjs import *

def run_summary():
    summary_obj = summary(PATH_res)
    # summary_obj.plot_ts_uzf_sums()
    # summary_obj.plot_hypsometry()
    summary_obj.plot_hist_head()
    # summary_obj.plot_land_z()     # not used
    # summary_obj.shp_heads()

def run_dtw():
    dtw_obj = dtw(PATH_res)
    dtw_obj.plot_area_hours()
    # dtw_obj.plot_hist_dtw()       # not used
    # dtw_obj.interesting()    # arcmap
    # dtw_obj.plot_interesting()    # arcmap
    # dtw_obj.shp_interesting()

def run_runoff():
    runobj_obj = runoff(PATH_res)
    runobj_obj.plot_ts_total()
    runobj_obj.plot_area_vol()
    # runobj_obj.shp_chg()

def run_sensitivity():
    sensitivity_obj = sensitivity(PATH_res)
    sensitivity_obj.total_et()

PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-21')
# run_summary()
# run_dtw()
# run_runoff()
# savefigs(PATH_res)
plt.show()
