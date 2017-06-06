from __future__ import print_function
import os.path as op
def _debug(steps, tstep=11, verb_level=1, errs=[1, 2], path_root=''):
       """
       Control Messages Printing.
       Verb_level 0 == silent
       Verb_level 1 == to console
       Verb_level 2 == to Coupled_today.log
       Verb_level 3 == to console and log
       Verb_level 4 == to console, make sure still working
       Errs = final SWMM Errors.
       """
       message=[]
       if not isinstance(steps, list):
           steps=[steps]
       if 'new' in steps:
           message.extend(['','  *************', '  NEW DAY: {}'.format(tstep), '  *************',''])
       if 'last' in steps:
           message.extend(['','  #############', '  LAST DAY: {}'.format(tstep), '  #############', ''])
       if 'from_swmm' in steps:
           message.extend(['Pulling Applied INF and GW ET', 'From SWMM: {} For MF-Trans: {}'.format(tstep,tstep), '             .....'])
       if 'for_mf' in steps:
           message.extend(['FINF & PET {} arrays written to: {}'.format(tstep+1, op.join(path_root, 'mf', 'ext'))])
       if 'mf_run' in steps:
           message.extend(['', 'Waiting for MODFLOW Day {} (Trans: {})'.format(tstep+1, tstep)])
       if 'mf_done' in steps:
           message.extend(['---MF has finished.---', '---Calculating final SWMM steps.---'])
       if 'for_swmm' in steps:
           message.extend(['Pulling head & soil from MF Trans: {} for SWMM day: {}'.format(tstep, tstep+1)])
       if 'set_swmm' in steps:
           message.extend(['', 'Setting SWMM values for new SWMM day: {}'.format(tstep + 1)])
       if 'swmm_done' in steps:
           message.extend(['', '  *** SIMULATION HAS FINISHED ***','  Runoff Error: {}'.format(errs[0]),
           '  Flow Routing Error: {}'.format(errs[1]),
           '  **************************************'])
       if 'swmm_run' in steps:
           message.extend(['', 'Waiting for SWMM Day: {}'.format(tstep+1)])

       if verb_level == 1 or verb_level == 3:
           print ('\n'.join(message))
       if verb_level == 2 or verb_level == 3:
           with open(op.join(self.init.path_child, 'Coupled_{}.log'.format(
                       time.strftime('%m-%d'))), 'a') as fh:
                       [fh.write('{}\n'.format(line)) for line in message]

       if verb_level == 4 and 'swmm_run' in steps or 'mf_run' in steps:
           print ('\n'.join(message))

       return message


# steps = ['new', 'last', 'from_swmm', 'for_mf', 'mf_run', 'mf_done', 'for_swmm', 'set_swmm', 'swmm_done', 'swmm_run']
# _debug(steps)

print ('\n  ###################################################################',
      '\n  MODFLOW {} FAILED to converge at Trans Day: {}'.format('a path', 10),
      '\n  ###################################################################\n')
