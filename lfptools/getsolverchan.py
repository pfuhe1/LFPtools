#!/usr/bin/env python

# inst: university of bristol
# auth: Peter Uhe
# mail: peter.uhe@bristol.ac.uk / pete.uhe@gmail.com

import os
import sys
import getopt
import subprocess
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import gdalutils


def getbedelevs_shell(argv):

    myhelp = '''
LFPtools v0.1

Name
----
getbedelevs

Description
-----------
Get channels calculated by the Neal channel solver, then output to shp and tif files

Usage
-----
>> lfp-getbedelevs -i config.txt

Content in config.txt
---------------------
[getbedelevs]
csvf   = Csv file output by channel solver script
templatetif  = Used to determine the geometry of the output tif files
proj   = Output projection in Proj4 format
bnkf   = Shapefile output bank
bedf   = Shapefile output depth
wthf   = Shapefile output width
'''

    try:
        opts, args = getopt.getopt(argv, "i:")
        for o, a in opts:
            if o == "-i":
                inifile = a
    except:
        print(myhelp)
        sys.exit(0)

    config = configparser.SafeConfigParser()
    config.read(inifile)

    csvf = str(config.get('getsolverchan', 'csvfile'))
    templatetif = str(config.get('getsolverchan', 'templatetif'))
    proj = str(config.get('getsolverchan', 'proj'))
    # Output
    bnkf = str(config.get('getsolverchan', 'bnkf'))
    bedf = str(config.get('getsolverchan', 'bedf'))
    wthf = str(config.get('getsolverchan', 'wthf'))

    getsolverchan(csvfile,templatetif,proj,bnkf,bedf,wthf)

def getsolverchan(csvfile,templatetif,proj,bnkf,bedf,wthf):

    print("    running getsolverchan.py...")

    # Reading channel solver output csv file
    # Cols are: point,lon,lat,reach,link,zh_est,z_est,width_smooth
    varnames = ['zh_est','z_est','width_smooth']
    outfiles = [bnkf,bedf,wthf]
    df = pd.read_csv(csvfile)
    # Reverse order of dataframe so smallest reaches are written first
    # As nodes have multiple values, the rasterising will override the first values in the shapefile
    df  = df.iloc[::-1]

    for var,outname in zip(varnames,outfiles):
        outshp = outname+'.shp'
        outtif = outname+'.tif'

        mybed = gpd.GeoDataFrame(df[var], crs={'init': 'epsg:4326'}, geometry=[
                             Point(xy) for xy in zip(df.lon.astype(float), df.lat.astype(float))])
        mybed.to_file(outshp)


        if var == 'width_smooth': # Hack, somehow var name has been clipped
            var = 'width_smoo'
        nodata = -9999
        fmt = "GTiff"
        mygeo = gdalutils.get_geo(templatetif)
        subprocess.call(["gdal_rasterize", "-a_nodata", str(nodata), "-of", fmt, "-co", "COMPRESS=DEFLATE", "-tr", str(mygeo[6]), str(mygeo[7]), "-a",
                         var, "-a_srs", proj, "-te", str(mygeo[0]), str(mygeo[1]), str(mygeo[2]), str(mygeo[3]), outshp, outtif])

if __name__ == '__main__':
    getbedelevs_shell(sys.argv[1:])
