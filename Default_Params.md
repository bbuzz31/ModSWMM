DEFAULT MODSWMM PARAMATERS
2017-05-15

def mf_params(self):
       MF_params  = OrderedDict([
                   ('slr', self.slr), ('name' , self.slr_name),
                   ('days' , self.days), ('START_DATE', '06/29/2011'),
                   ('path_exe', op.join('/', 'opt', 'local', 'bin')),
                   ('ss', False), ('ss_rate', 0.0002783601), ('strt', 1),
                   ('extdp', 3.0), ('extwc', 0.101), ('eps', 3.75),
                   ('thts', 0.3), ('sy', 0.25), ('vks', 0.12), ('surf', 0.3048),
                   ('noleak', False), ('diff', 0.05),
                   ('coupled', self.coupled), ('Verbose', self.verbose)
                   ])
       return MF_params

   def swmm_params(self):
       # to adjust rain/temp or perviousness change the actual csv files
       SWMM_params = OrderedDict([
              #OPTIONS
              ('START_DATE', self.mf_parms['START_DATE']),
              ('name', self.slr_name), ('days', self.days),
              # SUBCATCHMENTS
              ('Width', 200),
              # SUBAREAS
              ('N-Imperv', 0.011), ('N-Perv', 0.015),
              ('S-Imperv', 0.05), ('S-Perv', 2.54),
              # INFILTRATION
              ('MaxRate', 50), ('MaxInfil', 0),# ('MinRate', 0.635),
              ('Decay', 4), ('DryTime', 7),
              # AQUIFER
              ('Por', self.mf_parms['thts']), ('WP', self.mf_parms['extwc']- 0.001),
              ('FC', self.mf_parms['sy']), ('Ksat' , self.mf_parms['vks']),
              ('Kslope', 25), ('Tslope', 0.00),
              ('ETu', 0.50), ('Ets', self.mf_parms['extdp']),
              ('Seep', 0), ('Ebot' ,  0), ('Egw', 0),
              ### GROUNDWATER
              ('Node', 13326),
              ('a1' , 0.00001), ('b1', 0), ('a2', 0), ('b2', 0), ('a3', 0),
              ('Dsw', 0), ('Ebot', 0),
              ### JUNCTIONS
              # note elevation and maxdepth maybe updated and overwrittten
              ('Elevation', ''), ('MaxDepth', ''), ('InitDepth', 0),
              ('Aponded', 40000), ('surf', self.mf_parms['surf']*0.5),
              ### STORAGE
              # ksat = steady state infiltration
              ('A1', 0), ('A2', 0), ('A0', 40000),
              ### LINKS
              ('Roughness', 0.02), ('InOffset', 0), ('OutOffset' , 0),
              ### TRANSECTS (shape)
              ('Depth', 2.0), ('WIDTH', 10),
              ('Side1', 0.5), ('Side2', 0.5), ('Barrels', 1),
              ### INFLOWS (slr)
              ('slr', self.slr),
              ('coupled', self.coupled), ('Verbose', self.verbose)
              ])
