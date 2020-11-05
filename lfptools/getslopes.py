#!/usr/bin/env python

# inst: university of bristol
# auth: jeison sosa
# mail: j.sosa@bristol.ac.uk / sosa.jeison@gmail.com

import sys
import getopt
import subprocess
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import gdalutils
from lfptools import shapefile
from lfptools import misc_utils
from osgeo import osr
from sklearn import linear_model


def getslopes_shell(argv):

    myhelp = '''
LFPtools v0.1

Name
----
getslopes

Description
-----------
Estimate slopes from a bank file (e.g. from lfp-fixelevs), slope
is estimated by fitting a 1st order model on the elevations. The
number of elevations to take is based on the parameter `step` in the
config.txt file

Usage
-----
>> lfp-getslopes -i config.txt

Content in config.txt
---------------------
[getslopes]
source = File from which get the slopes e.g. resulting file from lfp-fixelevs
output = Output file
netf = Target mask file path
recf = `Rec` file path
proj = Output projection is Proj4 format
step = steps to count, upstream and downstream
method = 'elev' (default) or 'taudem' (use slopes calculated by taudem per reach)
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

    source = str(config.get('getslopes', 'source'))
    output = str(config.get('getslopes', 'output'))
    netf = str(config.get('getslopes', 'netf'))
    recf = str(config.get('getslopes', 'recf'))
    proj = str(config.get('getslopes', 'proj'))
    step = int(config.get('getslopes', 'step'))
    method = config.get('getslopes', 'method','elev')
    getslopes(source,output,netf,recf,proj,step,method)

# Wrapper function to call either getslopes_elev or getslopes_taudem
def getslopes(source,output,netf,recf,proj,step,method):
    if method == 'elev':
        getslopes_elev(source,output,netf,recf,proj,step)
    elif method == 'taudem':
        getslopes_taudem(source,output,netf,recf,proj)

def getslopes_taudem(source,output,netf,recf,proj):

    print("    runnning getslopes.py (from taudem)...")

    # Reading XXX_rec.csv file
    rec = pd.read_csv(recf)

    # Reading streamnet file
    snet = gpd.read_file(source)
    print('snet',snet['LINKNO'])

    # Loop over rec items
    slopes = np.zeros([len(rec.index)])
    for i in rec.index:
        link  = rec['link'][i]
        try:
            # Match link between streamnet and recf, then extract slope
            slopes[i] = snet.loc[snet['LINKNO']==link, 'Slope'].values[0]
        except:
            raise Exception('Cant match link: '+str(link))
    # Set minimum slope to be 1e-6
    slopes[slopes<1.e-6]=1.e-6
    # Add slope to dataframe
    rec['slope'] = slopes

    write_output(rec,output,netf,proj)

def getslopes_elev(source,output,netf,recf,proj,step):

    print("    runnning getslopes.py...")

    # Reading XXX_rec.csv file
    rec = pd.read_csv(recf)

    # Reading bank file (adjusted bank)
    elev = np.array(shapefile.Reader(source).records(), dtype='float64')

    # Retrieving adjusted bank elevations from XXX_bnkfix.shp file
    # Values are stored in rec['bnk']
    bnkadj = []
    for i in rec.index:
        dis, ind = misc_utils.near_euc(elev[:, 0], elev[:, 1], (rec['lon'][i],
                                                                rec['lat'][i]))
        bnkadj.append(elev[ind, 2])
    rec['bnkadj'] = bnkadj

    # Calculating slopes
    # coordinates are grouped by REACH number
    rec['slope'] = 0
    recgrp = rec.groupby('reach')
    for reach, df in recgrp:
        ids = df.index
        dem = df['bnkadj']
        # calc slopes
        slopes_vals = calc_slope_step(
            dem, df['lon'].values, df['lat'].values, step)
        rec['slope'][ids] = slopes_vals

    write_output(rec,output,netf,proj)


def calc_slope_step(dem, x, y, step):

    myslp = np.ones(dem.size)*-9999

    # calculate distance by using haversine equation
    dis = calc_dis_xy(x, y)

    # fit a linear regression by using Scikit learn, other more sophistcated
    # methods can be used to estimate the slope check on Linear regression methods
    # on Scikit learn .ebsite

    for i in range(len(dem)):

        left = max(0, i-step)
        right = min(len(dem), i+step+1)  # +1 because inclusive slicing
        # *1000 -> to convert from kilometers in meters, reshape -> to requirement scikitlearn
        X_train = dis[left:right].reshape(-1, 1)*1000
        Y_train = dem[left:right]
        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        slp = abs(regr.coef_)

        if slp <= 0.000001:
            slp = 0.0001
        myslp[i] = slp

        # # DEBUG DEBUG DEBUG
        # plt.scatter(X_train, Y_train,  color='black')
        # plt.scatter(X_train[step],Y_train[step], color='red')
        # plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)
        # plt.show()

    return myslp


def haversine(point1, point2, miles=False):
    """
    Calculate the great-circle distance bewteen two points on the Earth surface.
    Uses Numpy functions

    """
    AVG_EARTH_RADIUS = 6371  # in km

    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat*0.5)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(lng*0.5)**2
    h = 2*AVG_EARTH_RADIUS*np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers


def calc_dis_xy(x, y):
    dis = np.zeros(x.size)
    for i in range(len(dis)):
        if i > 0:
            dis[i] = haversine([y[i], x[i]], [y[i-1], x[i-1]])
        discum = np.cumsum(dis)
    return discum

def write_output(rec,output,netf,proj):

    # Writing .shp resulting file
    # Initiate output shapefile
    #w = shapefile.Writer(shapefile.POINT)
    #w.field('x')
    #w.field('y')
    #w.field('slope')

    #for i in rec.index:
    #    w.point(rec['lon'][i], rec['lat'][i])
    #    w.record(rec['lon'][i], rec['lat'][i], rec['slopes'][i])
    #w.save("%s.shp" % output)

    print('Writing out data')
    df = gpd.GeoDataFrame(rec[['lon','lat','slope']], crs={'init': 'epsg:4326'}, geometry=[
                                 Point(xy) for xy in zip(rec.lon.astype(float), rec.lat.astype(float))])
    df.to_file(output+'.shp')

    # write .prj file
    prj = open("%s.prj" % output, "w")
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj)
    prj.write(srs.ExportToWkt())
    prj.close()

    # Writing .tif file
    # Reading XXX_net.tif file for coordinates
    geo = gdalutils.get_geo(netf)
    nodata = -9999
    fmt = "GTiff"
    name1 = output+".shp"
    name2 = output+".tif"
    subprocess.call(["gdal_rasterize", "-a_nodata", str(nodata), "-of", fmt,  "-co", "COMPRESS=DEFLATE","-tr",
                     str(geo[6]), str(geo[7]), "-a", "slope", "-a_srs", proj, "-te", str(geo[0]), str(geo[1]), str(geo[2]), str(geo[3]), name1, name2])

if __name__ == '__main__':
    getslopes_shell(sys.argv[1:])
