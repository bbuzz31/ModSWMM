import os.path as op
from components import bcpl, swmmSOL as swmm

def run_swmm():
    """ Run SWMM Using SOL """
    path_child = op.join('/', 'Users', 'bb', 'Dropbox', 'Temp')
    NAME_swmm  = 'Dummy.inp'
    elapsed = []
    swmm.initialize(op.join(path_child, NAME_swmm))
    steps = 3*24
    cells = [15001, 15002]
    leak  = 0.02 # m
    for timesteps in range(0,steps):
        if swmm.is_over():
            print '\n   ### ERROR: SWMM has less steps than MF  ### \n'
            break

        elapsed.append (swmm.get_time())
        swmm.run_step()
        print swmm.get(str(15002), swmm.SUB_AREA, swmm.SI)
        swmm.setGW(str(15002), swmm.LEAK, swmm.SI, 10000)
        # print swmm.get(str(15002), swmm.POND, swmm.SI)

# runoff_addLeak(j, runoffStep);

    errors = swmm.finish()

run_swmm()
