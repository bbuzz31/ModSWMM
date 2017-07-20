### BB Steps for Compiling MFNWT - 04-29-17
# Run Python Script, which I've edited to include a makefile, making sure to keep folders
# make clean and get rid of everything but src_temp and makefile
# move makefile to source directory and run
# edit makefile to not run the make_bin function
# edit makefile to save an object directory OUTSIDE of the source folder
# add make install to copy program to opt/local/bin

### TEST
# Copy test-run from MF-2005
# run the test scenarios, compare with properly compiled MF list file (~/Software_Thesis/MODFLOW/MF_2005/test-out)
	# compare with version of MF-NWT Compiled with pymake


### CHANGES
# grep BB or hopefully --BB
# there are 3 flushes (grep them)
	# 2 in bas, one in uzf
# a call to swmm_is_done and modflow_is_done
# turned off the print for what file file it's looking for, can easily turn on again right before the call to sleep in Coupling.f
# turned off the print of Solving ... in MF_NWT.f lines 279, 280, 281

######
src_coupled has all the files that are necessary for a coupled simulation

######
orig is non-coupled, simply made with the pymake script
