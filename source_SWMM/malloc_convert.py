# python script to replace for compiling SWMM on mac
# mac has 'malloc' function in stdlib.h, not in malloc.h

import os
import fileinput
from BB_PY import BB

path = os.path.join(op.expanduser('~'), 'Software_Thesis', 'SWMM', 'swmm_engine',
                    'source5_1_011')

filelist = BB.filelist(path, 'c')

lookfor = '#include '
old_word = '<malloc.h>'
new_word = '<stdlib.h>'

fh = fileinput.input(files = filelist, inplace=1)
for line in fh:
    line = line.rstrip("\n")
    if lookfor + '<malloc.h>' in line:

    check =  (BB.check_next_chars(lookfor, line, old_word))
    if check == 'PASS':
        print (BB.insert_word (new_word, line, lookfor))
    else:
        print (line)
fh.close()

grep -rl 'malloc.h' ./ | xargs sed -i 's/malloc.h/stdlib.h/g'
find . -type f -name '*.c' -exec sed -i '' s/malloc.h/stdlib.h/ {} +
