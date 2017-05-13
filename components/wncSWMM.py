""" wncSWMM Classes """

import BB
import os
import os.path as op
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from swmmPUT import *
import bcpl

class WNC_Base(object):
    def __init__(self, path_root, **params):
        self.path      = path_root
        self.path_data = op.join(self.path, 'Data')
        self.params    = params
        _              = self.swmm_load()
        __             = self.mf_load()

    def swmm_load(self):
        self.df_subs =  pd.read_csv(op.join(self.path_data, 'SWMM_subs.csv'),
                                           index_col='Zone')
        self.df_jcts   = pd.read_csv(op.join(self.path_data, 'SWMM_junctions.csv'),
                                           index_col='Zone')
        self.df_links   = pd.read_csv(op.join(self.path_data, 'SWMM_links.csv'),
                                           index_col='Zone')
        self.df_outs    = pd.read_csv(op.join(self.path_data, 'SWMM_outfalls.csv'),
                                           index_col='Zone')
        self.df_stors   = pd.read_csv(op.join(self.path_data, 'SWMM_storage.csv'),
                                           index_col='Zone')
        self.df_inflows = pd.read_csv(op.join(self.path_data, 'SWMM_inflows.csv'),
                                           index_col='Node')
        # combine for mapping purposes
        self.df_nodes   = pd.concat([self.df_jcts, self.df_outs, self.df_stors])

    def mf_load(self):
        """ Load steady state head and soil moisture data from uzfb files """
        # retrieve head and soil moisture from steady state files (these only have subcatchs);
        mat_mf       = bcpl.mf_get_all(self.path, 0, **self.params) #
        self.df_mf   = pd.DataFrame(mat_mf, index=mat_mf[:,4],
              columns=['ROW', 'COL','HEAD', 'THETA', 'IDX']).drop('IDX', axis=1)
        try:
            self.df_mf.index = self.df_subs.index
        except:
            raise SystemError('Incorrect amount of uzfb gage files. Run MF 0 (SS).')

class WNC_SWMM_Inps(WNC_Base):
    def __init__(self, path_root, **params):
        WNC_Base.__init__(self, path_root, **params)
        self.dict_idx = self.indices()

    def indices(self):
        dict_idx = {}
        # sub data
        dict_idx['subs']    = self.df_subs.index.tolist()
        # select cells by hand in arc, convert to csv, send to excel
        dict_idx['outs']    = self.df_outs.index.tolist()
        # create junction at each outfall to stream
        dict_idx['jcts']    = self.df_jcts.index.tolist()
        # combine all nodes for mapping purposes
        dict_idx['nodes']   = self.df_nodes.index.tolist()
        # create stream reaches in excel; one per constant head cell
        dict_idx['links']   = self.df_links.index.tolist()
        # select storage units using python list in swmm_surface
        dict_idx['stors']   = self.df_stors.index.tolist()
        # junction and storage units - lakes (aka chd nodes)
        dict_idx['inflows'] = self.df_inflows.index.tolist()
        return dict_idx

    def swmm_objs(self):
        """ Create SWMM input file Sections. """

        ##### ONLY UPDATING CONSTANTS
        Title   = "\n".join(['Brett Buzzanga', time.strftime('%A %B %d, %Y'),
                             'Input File Created: ' + time.strftime('%X'),
                             'SLR: {}'.format(self.params.get('slr'))])
        title    = SwmmPd(title=Title).final_dict()
        evap     = EvapInp().final_dict()
        temp     = TempInp(op.join(self.path_data, 'climate_full.DAT')).final_dict()
        rain     = RainInp(op.join(self.path_data, 'rain_full.DAT'), [0]).final()
        report   = ReportInp().final_dict()
        opts     = OptInp(**self.params).final_dict()
        aq       = AqInp(**self.params).final()
        outs     = OutInp(self.dict_idx['outs'], **self.params).final()
        stors    = StorInp(self.dict_idx['stors'], **self.params).final()

        ##### WILL OVERWRITE COLUMNS
        sub      = SubInp(self.dict_idx['subs'], **self.params)
        area     = AreasInp(self.dict_idx['subs'], **self.params)
        infil    = InfInp(self.dict_idx['subs'], **self.params)
        gw       = GwInp(self.dict_idx['subs'], **self.params)
        juncts   = JnctInp(self.dict_idx['jcts'], **self.params)
        streams  = LinkInp(self.dict_idx['links'], **self.params)
        tsects   = TsectsInp(self.dict_idx['links'], **self.params)
        inflows  = InflowsInp(self.dict_idx['inflows'], **self.params)
        coords   = CoordInp(self.dict_idx['nodes'])
        polys    = PolyInp(self.dict_idx['subs'])

        ### SUBCATCHMENTS
        sub.df['Outlet']    = self.df_subs.Outlet.astype(int)
        sub.df['perImperv'] = self.df_subs.perImperv
        sub.df['perSlope']  = self.df_subs.perSlope
        sub = sub.final()

        ### SUBAREAS
        area.df['N-Perv'] =  self.df_subs.N_Perv
        area.df['S-Perv'] =  self.df_subs.S_Perv
        area = area.final()

        ### INFILATRATION
        infil.df['MinRate'] = self.df_subs.Min_Inf
        inf = infil.final()

        ### GROUNDWATER
        gw.df['Esurf'] = self.df_subs.Esurf
        # From MODFLOW step 1 of Trans (SS)
        gw.df['Egw']   = self.df_mf.HEAD
        gw.df['Umc']   = self.df_mf.THETA
        gw = gw.final()

        ### JUNCTIONS
        # put bottom of node to the bottom of the stream; assume land surface is top of stream
        juncts.df['Elevation'] = self.df_jcts['invert_z'] - self.params.get('Depth', 0)
        juncts.df.MaxDepth = self.params.get('Depth', 0) - self.params.get('surf', 0)
        jncts = juncts.final()

        ### STREAMS
        # no init flow; it would double tidal inflow to nodes, which isnt right
        #streams.df['InitFlow']  = swmm_df['links'].conv2cms * (0.3048 + params.get('slr', 0))
        streams.df['From_Node'] = self.df_links.From_Node
        streams.df['To_Node']   = self.df_links.To_Node
        streams.df['InOffset']  = self.df_links.InOffset
        conduits = streams.final()

        ### STREAM GEOMETRY
        tsects.df.WIDTH = self.df_links.Geom2_width
        geom = tsects.final()

        ### INFLOWS
        inflows.df['Baseline'] = self.df_inflows.conv2cms * (0.3048 + float(self.params.get('slr')))
        #inflows.df.loc[STOR[:-2], 'Baseline'] = 200
        inflow = inflows.final()

        ### NODE COORDINATES
        coords.df['X-Coord'] = self.df_nodes.CENTROID_X
        coords.df['Y-Coord'] = self.df_nodes.CENTROID_Y
        xynodes = coords.final()

        ### POLY COORDINATES
        polys.df['X-Coord']  = self.df_subs.CENTROID_X
        polys.df['Y-Coord']  = self.df_subs.CENTROID_Y
        xypolys = polys.final()
        return [title, opts, evap, temp, rain, sub, area, inf, aq, gw, jncts, outs,
                            stors, conduits, inflow, geom, report, xynodes, xypolys]

    def swmm_write(self):
        """
        Purpose:
            Write the swmm.inp file.
        Arguments:
            path_root = (+ /SWMM/) is location file will be written
            to_write  = list of swmmput objects
            swmm_days = SET FROM PARAMETERS (prob ignore following 3 lines)
                        ACTUAL amount of days (1 is be added automatically for SWMM)
                        One less than coupled step.
                        Equal to MF Kper (check). Set
        """
        to_write  = self.swmm_objs()
        swmm_days = self.params.get('days', -1)

        opts  = [i for i, items in enumerate(to_write) if '[OPTIONS]' in items][0]
        tstep =[each for each in to_write[opts]['[OPTIONS]'] if 'REPORT_STEP' in each][0][1]

        if isinstance(to_write, list):
            inp_name      = '{}.inp'.format(self.params.get('name', 'missing_name'))
            inp_file      = op.join(self.path, 'SWMM', inp_name)
            writedicts(inp_file, to_write)
        else:
            raise TypeError('Must pass a list of swmmput objects')

        if self.params.get('Verbose'):
            print '  ***********************************', \
                  '\n  SWMM Input File Created at: ', \
                  '\n  ...',"/".join(BB.splitall(inp_file)[-4:]), \
                  '\n  SWMM Time Step:', tstep, '(hr:min:sec)', \
                  '\n  SWMM simulation length:', swmm_days, 'days', \
                  '\n  ***********************************\n'

def main(path_root, write=True, **params):
    ## will need all the params from Coupled.py
    wnc_obj = WNC_SWMM_Inps(path_root, **params)
    if write:
        wnc_obj.swmm_write()
    else:
        print wnc_obj.swmm_objs()[0]
