#!/usr/bin/env python

"""
Backup Pickles and Coupled File to Time Capsule

Usage:
  TC_backup.py PATH_DEST

Arguments:
  PATH_DEST  Time Capsule Thesis_Backups Directoy

Examples:
  Call from script:
  TC_backup.py /Volumes/BB_Cap/Thesis_Backups

  Call from terminal:
  TC_backup.py .

Notes:
  The path to the TimeCapsule may be messed up.
  Better to call this from terminal.
"""

from __future__ import print_function
import os
import os.path as op
import tarfile

from docopt import docopt
from schema import Schema, Use

def make_tarfile(output_dest, source_dir):
    with tarfile.open('{}.tar.gz'.format(output_dest), "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def main(path_dest):
    PATH_src    = op.join('/', 'Volumes', 'BB_4TB', 'Thesis')
    if not op.exists(PATH_src):
        raise OSError ('Path to Source Incorrect')

    # get list of folders that are already backed up
    exists_full = os.listdir(path_dest)
    # strip off the ending for comparison
    exists      = [dirs.split('.')[0] for dirs in exists_full
                                      if dirs.endswith('tar.gz')]
    # make only the directories not already existing
    for d in os.listdir(PATH_src):
        d_full = op.join(PATH_src, d)
        if op.isdir(d_full) and d_full in exists:
            print (d, 'already exists, skipping...')
        elif op.isdir(d_full):
            make_tarfile(op.join(path_dest, d), d_full)

if __name__ == '__main__':
    arguments   = docopt(__doc__)
    typecheck   = Schema({'PATH_DEST' : os.path.exists}, ignore_extra_keys=True)

    PATH_tcap   = op.abspath(typecheck.validate(arguments)['PATH_DEST'])

    main(PATH_tcap)
