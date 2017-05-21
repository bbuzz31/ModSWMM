"""
Run Analysis
05.21.2017
"""
import os
import os.path as op
import matplotlib as mpl

from utils.AnalysisObjs import *

def savefigs(path):
    path_fig = op.join(path, 'Figures')
    if not op.isdir(path_fig):
        os.makedirs(path_fig)
    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        # mpl.rcParams['figure.figsize']   = (18, 12) # figure out a way to make this work
        fig.savefig(op.join(path_fig, fig.get_label()), dpi=300)


def run_summary():
    summary_obj = summary(PATH_res)
    summary_obj.plot_ts_uzf_sums()
    summary_obj.plot_hypsometry()
    summary_obj.plot_hist_head()
    # summary_obj.plot_land_z()     # not used
    # summary_obj.save_cur_fig()

def run_dtw():
    dtw_obj = dtw(PATH_res)
    # dtw_obj.plot_area_hours()
    # dtw_obj.plot_hist_dtw()       # not used
    # dtw_obj.plot_interesting()    # arcmap
    # dtw_obj.shp_interesting()
    # dtw.save_cur_fig()

def run_runoff():
    runobj_obj = runoff(PATH_res)
    runobj_obj.plot_ts_total()
    runobj_obj.plot_area_vol()
    # runobj_obj.shp_chg()
    # runobj.save_cur_fig()

PATH_res = op.join('/', 'Volumes', 'BB_4TB', 'Thesis', 'Results_05-18')
run_summary()
run_dtw()
run_runoff()
savefigs(PATH_res)
# plt.show()
