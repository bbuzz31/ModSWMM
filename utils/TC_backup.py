#!/usr/bin/env python

"""
Backup Pickles and Coupled File to Time Capsule

Usage:
  TC_backup.py PATH_SRC

Arguments:
  PATH_SRC  Results_MMDD Directory to move to Time Capsule
"""

import os
import os.path as op
import shutil

from docopt import docopt
from schema import Schema, Use

def move(path_src):
    date         = op.basename(path_src).split('_')[1]
    path_pickles = op.join(path_src, 'Pickles')
    path_log     = op.join(path_src, 'SLR-0.0_{}'.format(date),
                                     'Coupled_{}.log'.format(date))
    path_dest    = op.join('/', 'Volumes', 'BB_Cap', 'Thesis_Backups', date)

    try:
        shutil.copy(path_log, op.join(path_dest, op.basename(path_log)))
        print "Backed up Log"
    except:
        print "Log not found. Check Dates of SLR Dirs"

    try:
        shutil.copy(op.join(op.dirname(path_src), 'README'), op.join(path_dest, 'README'))
        print 'README Overwritten'
    except:
        pass

    shutil.copytree(path_pickles, op.join(path_dest, 'Pickles'))
    print "Backed up Pickles"


if __name__ == '__main__':
    arguments   = docopt(__doc__)
    typecheck   = Schema({'PATH_SRC' : os.path.exists}, ignore_extra_keys=True)
    PATH_SRC    = op.abspath(typecheck.validate(arguments)['PATH_SRC'])

    move(PATH_SRC)
