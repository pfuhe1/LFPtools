#!/usr/bin/env python

# inst: university of bristol
# auth: jeison sosa
# mail: j.sosa@bristol.ac.uk / sosa.jeison@gmail.com

import os
import sys
import subprocess
import configparser
import getopt
import numpy as np
import gdalutils
from lfptools import shapefile
from lfptools import misc_utils
from osgeo import osr
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve


def getdepths_shell(argv):

    myhelp = '''
LFPtools v0.1

Name
----
getdepths

Description
-----------
Get river depths, three methods availables:

1) depth_raster
    Get depths from a raster of depths

2) depth_geometry
    Get depths by using hydraulic geometry equation
    depth = r * width ^ p

3) depth_mannings
    Get depths by using simplified mannings equation
    ((bankfull_flow*manning_coef)/(slope**0.5*width))**(3/5.)

Usage
-----
>> lfp-getdepths -i config.txt

Content in config.txt
---------------------
[getdepths]
recf   = `Rec` file path
proj   = Output projection in Proj4 format
netf   = Target mask file path
method = depth_raster, depth_geometry, depth_mannings
output = Shapefile output file path

# If depth_raster
fdepth = Depth raster source file GDAL format projection EPSG:4326
thresh = Serching threshold in degrees

# If depth_geometry
wdtf   = Shapefile width from lfp-getwidths
r      = Constant number
p      = Constant number

# If depth_mannings
n      = Manning's coefficient 
wdtf   = Shapefile width from lfp-getwidths
slpf   = Shapefile slope from lfp-getslopes
qbnkf  = Shapefile q bank full

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

    proj = str(config.get('getdepths', 'proj'))
    netf = str(config.get('getdepths', 'netf'))
    method = str(config.get('getdepths', 'method'))
    output = str(config.get('getdepths', 'output'))

    try:
        fdepth = str(config.get('getdepths', 'fdepth'))
        thresh = np.float64(config.get('getdepths', 'thresh'))
        kwargs = {'fdepth':fdepth,'thresh':thresh}
    except:
        pass

    try:
        wdtf = str(config.get('getdepths', 'wdtf'))
        r = np.float64(config.get('getdepths', 'r'))
        p = np.float64(config.get('getdepths', 'p'))
        kwargs = {'wdtf':wdtf,'r':r,'p':p}
    except:
        pass

    try:
        n = np.float64(config.get('getdepths', 'n'))
        wdtf = str(config.get('getdepths', 'wdtf'))
        slpf = str(config.get('getdepths', 'slpf'))
        qbnkf = str(config.get('getdepths', 'qbnkf'))
        kwargs = {'n':n,'wdtf':wdtf,'slpf':slpf,'qbnkf':qbnkf}
    except:
        pass

    getdepths(proj,netf,method,output,**kwargs)

def getdepths(proj,netf,method,output,**kwargs):

    print("    runnning getdepths.py...")

    fname = output

    w = shapefile.Writer(shapefile.POINT)
    w.field('x')
    w.field('y')
    w.field('depth')

    if method == "depth_raster":
        depth_raster(w, netf, **kwargs)
    elif method == "depth_geometry":
        depth_geometry(w, **kwargs)
    elif method == "depth_manning":
        depth_manning(w, **kwargs)
    else:
        sys.exit("ERROR method not recognised")

    # write final value in a shapefile
    w.save("%s.shp" % fname)

    # write .prj file
    prj = open("%s.prj" % fname, "w")
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj)
    prj.write(srs.ExportToWkt())
    prj.close()

    nodata = -9999
    fmt = "GTiff"
    name1 = output+".shp"
    name2 = output+".tif"
    mygeo = gdalutils.get_geo(netf)
    subprocess.call(["gdal_rasterize", "-a_nodata", str(nodata), "-of", fmt, "-tr", str(mygeo[6]), str(mygeo[7]), "-co", "COMPRESS=DEFLATE",
                     "-a", "depth", "-a_srs", proj, "-te", str(mygeo[0]), str(mygeo[1]), str(mygeo[2]), str(mygeo[3]), name1, name2])


def depth_raster(w, netf, fdepth, thresh):
    """
    From a raster of depths this subroutine finds nearest depth to every river pixel in grid
    """

    # Reading river network file
    dat_net = gdalutils.get_data(netf)
    geo_net = gdalutils.get_geo(netf)
    iy, ix = np.where(dat_net > 0)
    xx = geo_net[8][ix]
    yy = geo_net[9][iy]

    # Reading depth source file
    dat = gdalutils.get_data(fdepth)
    geo = gdalutils.get_geo(fdepth)
    iy, ix = np.where(dat > -9999)
    xdat = geo[8][ix]
    ydat = geo[9][iy]

    depth = []
    for x, y in zip(xx, yy):
        try:
            dis, ind = misc_utils.near_euc(xdat, ydat, (x, y))
            if dis <= thresh:
                val = dat[iy[ind], ix[ind]]
                depth.append(val)
            else:
                depth.append(np.nan)
        except ValueError:
            depth.append(np.nan)

    for x,y,mydepth in zip(xx,yy,depth):
        w.point(x,y)
        w.record(x,y,mydepth)

    return w


def depth_geometry(w, r, p, wdtf):
    """
    Uses hydraulic geoemtry equation to estimate requires width, r and p coeffs.
    """

    width = np.array(shapefile.Reader(wdtf).records(), dtype='float64')
    x = width[:, 0]
    y = width[:, 1]

    for i in range(width.shape[0]):

        print("getdepths.py - " + str(width.shape[0]-i))

        mydepth = r*width[i, 2]**p

        w.point(x[i], y[i])
        w.record(x[i], y[i], mydepth)

    return w


def depth_manning(f, n, qbnkf, slpf, wdtf):
    """
    Uses manning's equation to estimate depth requires bankfull flow,
    slope, width and manning coefficient
    """

    # load width shapefile
    width = np.array(shapefile.Reader(wdtf).records(), dtype='float64')
    xw = width[:, 0]
    yw = width[:, 1]

    qbnk = np.array(shapefile.Reader(qbnkf).records(), dtype='float64')
    xq = qbnk[:, 0]
    yq = qbnk[:, 1]

    slope = np.array(shapefile.Reader(slpf).records(), dtype='float64')
    xs = slope[:, 0]
    ys = slope[:, 1]

    # iterate over every width x-y pair in the shapefile
    for i in range(width.shape[0]):

        # get index for Q and S based on W coordinates
        iiq = near(yq, xq, np.array([[xw[i], yw[i]]]))
        iis = near(ys, xs, np.array([[xw[i], yw[i]]]))

        if (np.float32(xw[i]) != np.float32(xq[iiq])) | (np.float32(xw[i]) != np.float32(xs[iis])) | (np.float32(yw[i]) != np.float32(yq[iiq])) | (np.float32(yw[i]) != np.float32(ys[iis])):

            print(xw[i], yw[i])
            print(xq[iiq], yq[iiq])
            print(xs[iis], ys[iis])
            sys.exit("Coordinates are not equal")

        w = width[i, 2]
        q = qbnk[iiq, 2]
        s = slope[iis, 2]

        data = (q, w, s, n)

        # # depth by using a full version of the mannings equation (solve numerically)
        # mydepth = fsolve(manning_depth,0,args=data)
        # f.point(xw[i],yw[i])
        # f.record(xw[i],yw[i],mydepth[0])

        # depth by using a simplified version of the mannings equation
        mydepth = manning_depth_simplified(data)

        f.point(xw[i], yw[i])
        f.record(xw[i], yw[i], mydepth)

    return f


def nearpixel(array, ddsx, ddsy, XA):
    """
    Find nearest pixel

    array: array with sourcedata
    ddsx: 1-dim array with longitudes of array
    ddsy: 1-dim array with latitudes of array
    XA: point
    """
    _ds = np.where(array > 0)

    # if there are river pixels in the window
    if _ds[0].size > 0:
        XB = np.vstack((ddsy[_ds[0]], ddsx[_ds[1]])).T
        ind = np.int(cdist(XA, XB, metric='euclidean').argmin())
        res = array[_ds[0][ind], _ds[1][ind]]
    else:
        res = -9999

    return res


def manning_depth(d, *data):
    q, w, s, n = data
    return q*n/s**0.5-w*d*(w*d/(2*d+w))**(2/3)


def manning_depth_simplified(data):
    q = data[0]
    w = data[1]
    s = data[2]
    n = data[3]
    return ((q*n)/(s**0.5*w))**(3/5.)


def near(ddsx, ddsy, XA):

    XB = np.vstack((ddsy, ddsx)).T
    dis = cdist(XA, XB, metric='euclidean').argmin()

    return dis


if __name__ == '__main__':
    getdepths_shell(sys.argv[1:])
