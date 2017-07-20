Compiling Instructions
As command line executale:
	Uncomment Cle in swmm5.c
	In make file delete all spaces before cc -o and then tab
	Change cc to gcc
	Comment include direct.h (os functionss like os.remove, windows only)
	Change malloc to stdlib.h by:
	find . -type f -name '*.c' -exec sed -i '' s/malloc.h/stdlib.h/ {} +

	 sudo cp /opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include/omp.h .f
		might have to get from different swmm source code dir

	Where . is src directory of swmm files
	Enclose omp.h in dynwave.c in" " instead of <>
	Enclose omp.h in project.c  in " " instead of <>
	Enclose omp.h in stats.c  in " " instead of <>

As Shared library:
	Do all that ^
	Remove object files (clean): Rm * .o
	Comment cle, uncomment shared line
	Possibly need:
	gcc -shared -o swmm5 $(objs) -fopenmp -fPIC -lpthread -lm

For Coswm:
	Python changes ctype DLL to CDLL

	Copy and paste functions into swmm5.c from swmm5.c in matswmm_BB
		Also include cosim.h
	Change malloc in cosimulation.c to stdlib.h

	--problem in swmm.5 defining coswm functions
	- Edit makefile to include coswm objects/headers
	- Remove DLLEXPORT from functions (though not necessary ;  SOL all functions are automatically exported)
	- The coswmm_orig directory has everything set as the minimum needed

	- For my additions, be careful copy everythign properly from headers and cosim.c
	- Addtiionaly consdieration for getting infiltraiton without gw (my functions)
	- Subarea in (swmm_get) has to be specifically declared as Subcatch[j].subArea
		○ because there subArea is an array of 3 structures
		○ structure three is the pervious one [subarea[2]]
	- Make sure my cosimulation functions are defined in swmm.h and coswmm.h
	- add function to get leakage
		○ add a variable to groundwater strucutre in objects
		○ only works if you put in gw structure
		○ initialize to 0.0 in gw init
		○ unit and time conversions
