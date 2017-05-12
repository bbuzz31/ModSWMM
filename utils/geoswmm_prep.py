# ---------------------------------------------------------------------------
# geoSWMM_prep.py
# -*- coding: utf-8 -*-
# Created: 06.07.2016
# Current: 11.01.2016
# Author: Brett Buzzanga, bbuzz001@odu.edu

# ---------------------------------------------------------------------------
# CREATE GRID_SWMM, A SHAPEFILE OF STUDY AREA ALIGNED TO MODFLOW GRID WITH ALL
# SWMM ATTRIBUTES; EXCEPT FOR GW AND INFILTRATION PARAMETERS
# ADD GW AND INFILTRATION PARAMETRS TO SHAPEFILE CREATED IN GEOSWMM (within Arc)

# !!! REQUIRES ARCGIS !!! #

# ---------------------------------------------------------------------------

import BB
import os
import os.path as op
import arcpy as ap

### Check out any necessary licenses
ap.CheckOutExtension("spatial")

### Set File Location Variables
gis_path  = op.join('C:\\', 'Users', 'bbuzzang', 'Google_Drive', 'WNC', 'GIS', '2016')
swmm_path = op.join('C:\\', 'Users', 'bbuzzang', 'Google_Drive', 'WNC', 'SWMM')

SWMM_prep = op.join(gis_path, 'SWMM_prep_new.gdb')
Base      = op.join(gis_path, 'Base.gdb')
Data      = op.join(gis_path, 'Data.gdb')
MF_SS     = op.join(gis_path, 'FloPy')
RAM       = op.join("in_memory")

### Set Geoprocessing environments
ap.env.workspace        = RAM
ap.env.scratchWorkspace = RAM

ap.env.outputCoordinateSystem    = ("PROJCS['NAD_1983_2011_UTM_Zone_18N',"
            "GEOGCS['GCS_NAD_1983_2011', DATUM['D_NAD_1983_2011',"
            "SPHEROID['GRS_1980',6378137.0,298.257222101]],"
            "PRIMEM['Greenwich',0.0], UNIT['Degree', 0.0174532925199433]],"
            "PROJECTION['Transverse_Mercator'], PARAMETER['False_Easting',500000.0],"
            "PARAMETER['False_Northing',0.0], PARAMETER['Central_Meridian',-75.0],"
            "PARAMETER['Scale_Factor',0.9996], PARAMETER['Latitude_Of_Origin',0.0],UNIT['Meter',1.0]]")
ap.env.geographicTransformations = ""
ap.env.overwriteOutput           = True
ap.Delete_management("in_memory")

### Raw input files:
    # grid vert:
        # exported from ModelMuse; Model_top was rejoined by ID to correct (also, 1 cell is altered from -7 to 0);
        # zone as str added for ISAT
grid_vertical = op.join(Base, 'Grid_vertical')
DEM_raw       = op.join(Data, 'LiDAR_WNC')
grid_raw      = op.join(Data, 'grid_poly_raw')
AOI           = op.join(Base, 'AOI')
CCAP          = op.join(Data, 'CCAP')
try:
    ap.CopyRaster_management(DEM_raw, op.join(SWMM_prep, 'DEM_UTM'))
except:
    print ('Warning: DEM already in working directory')

### Modflow input files:
# Steady state, for joining to final and defining outflows and active cells
grid_Active   = op.join(MF_SS, 'NON_CHD.shp')

### Intermediate files
grid_rast     = op.join(RAM, 'rast')
grid_points   = op.join(RAM, 'points')
join_dir      = op.join(RAM, 'direction')
int_dir       = op.join(RAM, 'int_dir')

ccap_points   = op.join(RAM,'ccap_points')
int_manning   = op.join(RAM, 'int_manning')

slope         = op.join(SWMM_prep, 'slope')
int_slope     = op.join(RAM, 'int_slope')
slope_points  = op.join(RAM, 'slope_points')
mean_slope    = op.join(SWMM_prep, 'slope_mean')

int_gw        = op.join(RAM, 'int_GW')
int_ISAT      = op.join(RAM, 'int_ISAT')

### Output files
# only grid_SWMM_prep (final) is needed; rest for debugging/sectioning code
grid_Dir      = op.join(SWMM_prep, 'grid_Dir')
grid_Manning  = op.join(SWMM_prep, 'grid_Manning')
grid_Slope    = op.join(SWMM_prep, 'grid_Slope')
grid_ISAT     = op.join(SWMM_prep, 'grid_ISAT')
grid_GW       = op.join(SWMM_prep, 'grid_GW')

grid_SWMM     = op.join(SWMM_prep, 'grid_SWMM')
table_SWMM    = op.join(swmm_path, 'Data', 'table_SWMM.csv')

### Processing parameters
cellsize = 200
coeff = op.join(swmm_path, 'ISAT', 'WNC', 'LC_ISvalues.csv') # coefficients for ISAT tool
# ---------------------------------------------------------------------------- #
# Field Updating Functions
def manning (Majority):
    if (Majority < 4):
        return 0.015
    elif (Majority < 6 and  Majority > 3):
        return 0.025
    elif (Majority < 7 and Majority > 5):
        return 0.10
    elif (Majority < 8 and Majority > 5):
        return 0.13
    elif (Majority < 14 and Majority > 12):
        return 0.25
    else:
        return 1

def dep_stor (Majority):
    if (Majority < 4):
        return 2.54
    elif (Majority < 6 and  Majority > 3):
        return 3.81
    elif (Majority < 7 and Majority > 5):
        return 5.08
    elif (Majority < 8 and Majority > 5):
        return 5.08
    elif (Majority < 14 and Majority > 12):
        return 6.35
    else:
        return 1

def flowID (OBJECTID, Direction):
    if (Direction == 1):
        return  (int(OBJECTID) + 1)
    elif (Direction == 2):
        return (int(OBJECTID) + 52)
    elif (Direction == 4):
        return (int(OBJECTID) + 51)
    elif (Direction == 8):
        return (int(OBJECTID) + 50)
    elif (Direction == 16):
        return (int(OBJECTID) - 1)
    elif (Direction == 32):
        return (int(OBJECTID) - 52)
    elif (Direction == 64):
        return (int(OBJECTID) - 51)
    elif (Direction == 128):
        return (int(OBJECTID) - 50)
    else:
        return 0.5
# ---------------------------------------------------------------------------- #

# **************************************************************************** #
# Set Script arguments
# flow_direction         # assign directional integers to grid based on elevation
# landcover              # assign ccap values to grid based on majority to calculate manning and storage depth
# slope                  # calculate internal slope of each grid cell as % rise
# ISAT                   # calculate impervious surface of each grid cell as % of cell
# flowID                 # assign the correct grid cell based on flow direction integer
# cleanup                # delete extranneous output grids
# **************************************************************************** #
# don't run all at once; runs out of memory
#runsteps = ['flow_direction', 'landcover', 'slope', 'ISAT', 'flowID', 'final']
#runsteps = ['flow_direction', 'landcover', 'slope', 'ISAT', 'final']
runsteps = ['final', 'export']
# **************************************************************************** #

if 'flow_direction' in runsteps:
    ap.Delete_management("in_memory")

    try:
        ap.Delete_management(grid_Dir)
        #ap.Delete_management(op.join(SWMM_prep,'rotated'))
    except:
        print 'Warning: Unable to delete original grid_Dir'

    ### Flow Direction
    ap.PolygonToRaster_conversion(grid_vertical, "Model_Top", grid_rast, cellsize=cellsize)
    grid_filled  = ap.sa.Fill(grid_rast, '1')
    outFlow = ap.sa.FlowDirection(grid_filled)
    ap.RasterToPoint_conversion(outFlow, grid_points, "VALUE")
    #ap.Rotate_management(accumFlow, op.join(SWMM_prep,'rotated'), '-19', '407463.996844 4067100.061781')

    # convert rasters to points, & join to vertical grid
    ap.SpatialJoin_analysis(grid_vertical, grid_points, join_dir, "JOIN_ONE_TO_ONE",
    "KEEP_COMMON", "", "INTERSECT", "", "")

    # Attach to Grid
    ap.CopyFeatures_management(grid_raw, int_dir)
    ap.JoinField_management(int_dir, "OBJECTID", join_dir, "OBJECTID", ["Zone", "MODEL_TOP", "grid_CODE"])
    ap.DeleteField_management (int_dir, ["ID", "Index"])
    ap.AlterField_management(int_dir, "grid_CODE", "Direction", "Direction")

    ap.AddField_management(int_dir, "Flow_ID", "double")

    # Map Flow Direction to Subcatch ID
    with ap.da.UpdateCursor(int_dir, ['OBJECTID', 'Direction', 'Flow_ID']) as cur:
        for row in cur:
            flowid = flowID(row[0], row[1])
            new_row = [row[0], row[1], flowid]
            #print new_row
            cur.updateRow(new_row)

    ### Flow Accumulation
    accumFlow = ap.sa.FlowAccumulation (outFlow)
    ap.RasterToPoint_conversion(accumFlow, grid_points, "VALUE")
    ap.SpatialJoin_analysis(grid_vertical, grid_points, join_dir, "JOIN_ONE_TO_ONE",
    "KEEP_COMMON", "", "INTERSECT", "", "")
    ap.JoinField_management(int_dir, "OBJECTID", join_dir, "OBJECTID", ["grid_CODE"])
    ap.AlterField_management(int_dir, "grid_CODE", "Accum", "Accum")

    ### Export to disc for viewing
    ap.FeatureClassToFeatureClass_conversion (int_dir, SWMM_prep, "grid_Dir")

if 'landcover' in runsteps:
    ap.Delete_management("in_memory")

    try:
        ap.Delete_management(grid_Manning)
    except:
        print 'Warning: Unable to delete original grid_Manning'

    ap.CopyFeatures_management(grid_raw, int_manning)

    # ----------------------- majority in ZonalStatisticsasTable is not working
    ap.FeatureToPoint_management(int_manning, grid_points, "CENTROID")

    # find majority landcover type in each grid cell, convert it to a point
    ccap_maj = ap.sa.ZonalStatistics(int_manning, "Zone", CCAP, "Majority", "DATA")
    ap.sa.ExtractValuesToPoints(grid_points, ccap_maj, ccap_points, "NONE", "VALUE_ONLY")

    # create list of desired fields to pass to final object
    fieldmappings = ap.FieldMappings()
    fieldmappings.addTable(int_manning)
    keepfields = [field.name for field in (ap.ListFields(int_manning))]
    keepfields.append("RASTERVALU")
    keepfields.append('Shape_Area')
    keepfields.append ('Shape_Length')

    # join all fields by location of points to grid cells
    ap.SpatialJoin_analysis(int_manning, ccap_points, grid_Manning, "JOIN_ONE_TO_ONE", "KEEP_ALL")

    # clean up fields in attribute table
    fieldmappings.addTable(grid_Manning)
    unwanted_fields = [field.name for field in fieldmappings.fields if field.name not in keepfields]
    ap.DeleteField_management (grid_Manning, unwanted_fields)
    ap.AlterField_management(grid_Manning, "RASTERVALU", "Majority", "Majority")

    # compute manning number based on landcover (SWMM_manual2015, pg 182)
    ap.AddField_management(grid_Manning, "Manning", "double")
    # compute depression storage based on landcover(SWMM_manual 2015, pg 181)
    ap.AddField_management(grid_Manning, "dep_stor", "double")
    # update fields
    with ap.da.UpdateCursor(grid_Manning, ['Majority', 'Manning', 'dep_stor']) as cur:
        for row in cur:
            man_num = manning(row[0])
            storage_depth = dep_stor(row[0])
            new_row = [row[0], man_num, storage_depth]
            cur.updateRow(new_row)

if 'slope' in runsteps:

    ap.Delete_management("in_memory")
    # initialize new files
    ap.CopyFeatures_management(grid_raw, int_slope)
    ap.FeatureToPoint_management(int_slope, grid_points, "CENTROID")
    try:
        ap.Delete_management (grid_Slope)
    except:
        print 'Warning: Unable to delete original grid_Slope'

    # create list of desired fields to pass to final object
    fieldmappings = ap.FieldMappings()
    fieldmappings.addTable(int_slope)
    keepfields = [field.name for field in (ap.ListFields(int_slope))]
    keepfields.append('RASTERVALU')
    keepfields.append('Shape_Area')
    keepfields.append('Shape_Length')

    # calculate slope as percent rise
    arcpy.gp.Slope_sa(DEM_raw, slope, 'PERCENT_RISE', '1')

    # Take average slope in each grid cell (to table, and new syntax don't work)
    ap.gp.ZonalStatistics_sa(int_slope, 'Zone', slope, mean_slope, 'MEAN', 'DATA')

    # Extract values to points
    ap.gp.ExtractValuesToPoints_sa(grid_points, mean_slope, slope_points, "#", "VALUE_ONLY")

    # Join all fields
    ap.SpatialJoin_analysis(int_slope, slope_points, grid_Slope, "JOIN_ONE_TO_ONE", "KEEP_ALL")

    # Remove unwanted fields and rename
    fieldmappings.addTable(grid_Slope)
    remove = [field.name for field in fieldmappings.fields if field.name not in keepfields]
    ap.DeleteField_management (grid_Slope, remove)
    ap.AlterField_management(grid_Slope, "RASTERVALU", "Slope", "% Slope")

    # change 0 values to NULL
    with ap.da.UpdateCursor(grid_Slope, 'Slope') as cur:
        # ------------------------------ May be an issue; gets rid of flat cells
        [cur.updateRow([None]) for row in cur if row[0] <= 0]

    # delete intermediate files
    remove = [mean_slope, slope]
    [ap.Delete_management (layer) for layer in remove]

if 'ISAT' in runsteps:
    ap.Delete_management("in_memory")
    ap.CopyFeatures_management(grid_raw, int_ISAT)
    ap.ImportToolbox(op.join(swmm_path, 'ISAT', 'ISAT_Toolkit.tbx'))

    #  assign toolbox an alias (ISAT)
    ap.ISATWithReclass2_ISAT(CCAP, "", "", "", "", "false", int_ISAT, "Zone", coeff, "Med", grid_ISAT, "", "", "", "")

if 'GW' in runsteps:
    try:
        ap.Delete_management("in_memory")
        ap.Delete_management(grid_GW)
    except:
        print 'Warning: Unable to delete original grid_Dir'
    # Set dummy variables for groundwater
    ap.CopyFeatures_management(grid_raw, int_ISAT)

if 'final' in runsteps:
    print ('The following must exist: grid_Dir, grid_Manning, grid_Slope'
            ' and grid_ISAT in:', SWMM_prep)
    try:
        ap.Delete_management (grid_SWMM)
    except:
        print 'Warning: Unable to delete original grid_SWMM'

    ap.FeatureClassToFeatureClass_conversion (grid_Dir, SWMM_prep, "grid_SWMM")
    # shapefiles to combine
    to_combine = [grid_Manning, grid_Slope, grid_ISAT]
    # create list of desired fields to pass to final object
    desired = ['Manning', 'dep_stor', 'Majority', 'Slope', 'Accum', 'MEAN_IS']
    # combine desired fields into one shape file
    [ap.JoinField_management(grid_SWMM, 'OBJECTID', to_combine[idx], 'OBJECTID',
                             desired) for idx in range(len(to_combine))]

    print "\n"
    print "Success!"
    print "Check grid_SWMM is correct"
    print "\n"
    # Compare with original grid_SWMM_old in SWMM_prep_old.gdb

if 'export' in runsteps:
    print 'grid_SWMM must exist at:', grid_SWMM
    try:
        os.remove (table_SWMM)
    except:
        print 'Creating new table'
    # export to csv
    keep_fields = ['Zone', 'Flow_ID', 'Accum', 'MEAN_IS', 'Slope', 'Manning',
                   'dep_stor', 'MODEL_TOP', 'COLUMN_', 'ROW']
    ap.ExportXYv_stats(grid_SWMM, keep_fields, 'COMMA', table_SWMM, 'ADD_FIELD_NAMES')

ap.CheckInExtension("Spatial")
