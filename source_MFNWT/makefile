# makefile created on 2017-04-29 07:58:11.894702
# by pymake (version 1.1.0)
# using the gfortran fortran and gcc c/c++ compilers.

# Define the directories for the object and module files,
# the executable, and the executable name and path.
OBJDIR = ../objs
BINDIR =
PROGRAM = NWT_BB

SOURCEDIR1=MODFLOW-NWT_1.1.2/src

VPATH = \
${SOURCEDIR1}

.SUFFIXES: .c .cpp .f .f90 .F90 .fpp .o

# Define the Fortran compile flags
F90 = gfortran
F90FLAGS = -O2 -fbacktrace -ffpe-summary=overflow

# Define the C compile flags
CC = gcc
CFLAGS = -O3 -D_UF

# Define the libraries
SYSLIBS = -lc

OBJECTS = \
$(OBJDIR)/Coupling.o \
$(OBJDIR)/gwfsfrmodule_NWT.o \
$(OBJDIR)/mach_mod.o \
$(OBJDIR)/NWT1_xmdlib.o \
$(OBJDIR)/modules.o \
$(OBJDIR)/utl7.o \
$(OBJDIR)/gwf2swr7util.o \
$(OBJDIR)/gwflakmodule_NWT.o \
$(OBJDIR)/gwf2bas7_NWT.o \
$(OBJDIR)/gwf2str7.o \
$(OBJDIR)/gwf2evt7.o \
$(OBJDIR)/de47_NWT.o \
$(OBJDIR)/gwf2ghb7_NWT.o \
$(OBJDIR)/gwf2riv7_NWT.o \
$(OBJDIR)/gwf2huf7.o \
$(OBJDIR)/gwf2fhb7.o \
$(OBJDIR)/gwfuzfmodule_NWT.o \
$(OBJDIR)/sip7_NWT.o \
$(OBJDIR)/gwf2hfb7_NWT.o \
$(OBJDIR)/gwf2gag7.o \
$(OBJDIR)/NWT1_module.o \
$(OBJDIR)/gwf2drn7_NWT.o \
$(OBJDIR)/gwf2drt7.o \
$(OBJDIR)/gwf2lpf7.o \
$(OBJDIR)/gwf2rch7.o \
$(OBJDIR)/gwf2swt7.o \
$(OBJDIR)/parutl7.o \
$(OBJDIR)/gwf2chd7.o \
$(OBJDIR)/mhc7.o \
$(OBJDIR)/gwf2bcf7.o \
$(OBJDIR)/gwf2ets7.o \
$(OBJDIR)/gwf2ibs7.o \
$(OBJDIR)/gwf2res7.o \
$(OBJDIR)/pcg7_NWT.o \
$(OBJDIR)/NWT1_gmres.o \
$(OBJDIR)/obs2bas7.o \
$(OBJDIR)/hufutl7.o \
$(OBJDIR)/nogmg.o \
$(OBJDIR)/NWT1_ilupc_mod.o \
$(OBJDIR)/gwf2sub7_NWT.o \
$(OBJDIR)/gwf2upw1.o \
$(OBJDIR)/gwf2wel7_NWT.o \
$(OBJDIR)/gwf2uzf1_NWT.o \
$(OBJDIR)/gwf2hydmod7.o \
$(OBJDIR)/NWT1_xmd.o \
$(OBJDIR)/obs2chd7.o \
$(OBJDIR)/obs2riv7.o \
$(OBJDIR)/MF_NWT.o \
$(OBJDIR)/NWT1_solver.o \
$(OBJDIR)/obs2drn7.o \
$(OBJDIR)/gwf2mnw17_NWT.o \
$(OBJDIR)/obs2ghb7.o \
$(OBJDIR)/obs2str7.o \
$(OBJDIR)/gwf2mnw27_NWT.o \
$(OBJDIR)/gwf2swr7.o \
$(OBJDIR)/gwf2swi27.o \
$(OBJDIR)/lmt8_NWT.o \
$(OBJDIR)/gwf2mnw2i7.o \
$(OBJDIR)/gwf2lak7_NWT.o \
$(OBJDIR)/gwf2sfr7_NWT.o

# Define task functions

# Create the bin directory and compile and link the executable
all: makebin | mfnwt

# Make the bin directory for the executable
makebin :
	# mkdir -p $(BINDIR)

# Define the objects that make up mfnwt
mfnwt: $(OBJECTS)
	-$(F90) $(F90FLAGS) -o $(PROGRAM) $(OBJECTS) $(SYSLIBS) -I$(OBJDIR) -J$(OBJDIR)

$(OBJDIR)/%.o : %.f
	@mkdir -p $(@D)
	$(F90) $(F90FLAGS) -c $< -o $@ -I$(OBJDIR) -J$(OBJDIR)

$(OBJDIR)/%.o : %.f90
	@mkdir -p $(@D)
	$(F90) $(F90FLAGS) -c $< -o $@ -I$(OBJDIR) -J$(OBJDIR)

$(OBJDIR)/%.o : %.F90
	@mkdir -p $(@D)
	$(F90) $(F90FLAGS) -c $< -o $@ -I$(OBJDIR) -J$(OBJDIR)

$(OBJDIR)/%.o : %.fpp
	@mkdir -p $(@D)
	$(F90) $(F90FLAGS) -c $< -o $@ -I$(OBJDIR) -J$(OBJDIR)

$(OBJDIR)/%.o : %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o : %.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean the object and module files and the executable
.PHONY : clean
clean :
	-rm -rf $(OBJDIR)
	# -rm -rf $(BINDIR)

# Clean the object and module files
.PHONY : cleanobj
cleanobj :
	-rm -rf $(OBJDIR)

# Install, copy to bin dir
.PHONY : install
install :
	-sudo rm -f /opt/local/bin/$(PROGRAM)
	-sudo ln -s $(shell pwd)/$(PROGRAM) /opt/local/bin/$(PROGRAM)
