import os.path as op
from components import bcpl, swmmSOL as swmm

def run_swmm():
    """ Run SWMM Using SOL """
    path_child = op.join('/', 'Users', 'bb', 'Dropbox', 'Temp')
    NAME_swmm  = 'Dummy.inp'
    elapsed = []
    swmm.initialize(op.join(path_child, NAME_swmm))
    steps = 4*24
    cells = [15001, 15002]
    leak  = 0.02 # m
    for timesteps in range(0,23):
        elapsed.append (swmm.get_time())
        swmm.run_step()
        # print swmm.get(str(15002), swmm.SUB_AREA, swmm.SI)
        # print swmm.get(str(15002), swmm.GET_LEAK, swmm.SI)
        swmm.setGW(str(15002), swmm.LEAK, swmm.SI,  1000)
        # print swmm.get(str(15002), swmm.GET_LEAK, swmm.SI)
        print swmm.get(str(15002), swmm.POND, swmm.SI)
        # swmm.setGW(str(15001), swmm.LEAK, swmm.SI, 3000)
    for timesteps in range(23,steps):
        elapsed.append (swmm.get_time())
        swmm.run_step()
        # # print swmm.get(str(15002), swmm.SUB_AREA, swmm.SI)
        swmm.setGW(str(15002), swmm.LEAK, swmm.SI, 000)
        print swmm.get(str(15002), swmm.POND, swmm.SI)
        pass

# runoff_addLeak(j, runoffStep);

    errors = swmm.finish()

run_swmm()
