answered = False
while not answered:
   overwrite = raw_input('Overwrite pickles? (y/n) ')
   if overwrite == 'y':
       print ('chose yes')
       answered = True
   elif overwrite == 'n':
       print ('Not overwriting pickles')
       answered = True
   else:
       print ('Choose y or n')
