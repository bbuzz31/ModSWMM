#!/usr/bin/env python
"""
Create a Pandas DataFrame from modflow .FHD (formatted head file)
Prints to screen.

Usage:
  fhd2df.py FHD KPER [options]

Examples:
   Print whole dataframe
   fhd2df.py FHD
   Print head (or tail)
   fhd2df.py --head 1

Arguments:
  FHD  fhd file
  KPER stress period to fetch

Options:
  --head Print top 5 rows                 [default: 0]
  --tail Print bottom 5 rows              [default: 0]
  --plot Plot 2d Grid                     [default: 0]
"""

from General import BB
import os
import flopy.utils.formattedfile as ff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from docopt import docopt
from schema import Schema, Use

def main(head_file, kper):
    try:
        hds  = ff.FormattedHeadFile(head_file, precision='single')
    except:
        hds  = ff.FormattedHeadFile(head_file, precision='double')
    heads    = hds.get_alldata(mflay=0)
    mat_heads = heads[kper]
    df_heads  = pd.DataFrame(mat_heads.flatten(), index=range(10001, 10001+3774),
                                                               columns=['Head'])
    rows = []
    for col in range(mat_heads.shape[1]):
        rows.extend(range(1, mat_heads.shape[0]+1))
    cols = np.repeat(range(1, mat_heads.shape[1] + 1), mat_heads.shape[0])

    df_heads['ROW'] = rows
    df_heads['COL'] = cols

    return df_heads

if __name__ == '__main__':
    arguments = docopt(__doc__)
    typecheck = Schema({'KPER': Use(int),      'FHD': os.path.exists,
                        '--head': Use(int), '--tail' : Use(int),
                        '--plot': Use(int)}, ignore_extra_keys=True)

    args = typecheck.validate(arguments)
    df_heads = main(args['FHD'], args['KPER'])

    if args['--head']:
        print df_heads.head()
    elif args['--tail']:
        print df_heads.tail()
    else:
        BB.print_Df(df_heads)

    if args['--plot']:
        mat_heads = np.where(mat_heads < -500, np.nan, mat_heads)
        plt.imshow(mat_heads)
        plt.colorbar
        plt.show()
    print 'Stress Period: {}'.format(args['KPER'])
