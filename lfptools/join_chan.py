#!/usr/bin/env python

# inst: university of bristol
# auth: jeison sosa
# mail: j.sosa@bristol.ac.uk / sosa.jeison@gmail.com

import os
import sys
import getopt
import subprocess
import configparser
import numpy as np
import pandas as pd
from lfptools import shapefile
from lfptools import misc_utils
import statsmodels.api as sm
import gdalutils
from osgeo import osr


def joinchan_shell(argv):

    myhelp = '''
LFPtools v0.1

Name
----
join_chan

Description
-----------
join channel parameters into single file

Usage
-----
>> lfp-joinchan -i config.txt

Content in config.txt
---------------------
[joinchan]
output = csv/shapefile output file path
recf   = `Rec` file path
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


    """
    This function takes the `Rec` csv file which contains records of
    each point in the river network and
    shapefiles for other channel parameters (produced by other LFPtools scripts)
    This assumes that the shapefiles have identical records to the `rec` file
    with the same order.

    It combines these datasets, extracts a subset defined by the `extent`
    Then outputs a new csv file to be used with the Neal channel solver
    Output records are sorted from downstream to upstream, by reach
    """

    output = str(config.get('joinchan', 'output'))
    recf = str(config.get('joinchan', 'recf'))
    bankelevf = str(config.get('joinchan', 'bankelevf'))
    bankfixf = str(config.get('joinchan', 'bankfixf'))
    widthf = str(config.get('joinchan', 'widthf'))
    slopef = str(config.get('joinchan', 'slopef'))
    bfqf= str(config.get('joinchan', 'bfqf'))
    depthf = str(config.get('joinchan', 'depthf'))
    extent = config.get('joinchan','extent').split(',') #TODO, set default

    join_chan(output,recf,bankelevf,widthf,slopef,bfqf,depthf,extent)

def join_chan(output,recf,bankelevf,bankfixf,widthf,slopef,bfqf,depthf,extent):

    print("    running joinchan.py...")

    # Reading XXX_rec.csv file
    rec = pd.read_csv(recf)

    # Load shapefiles, and add to rec file
    rec['elev'] = np.array(shapefile.Reader(bankelevf).records(), dtype='float')[:,2]
    rec['elevfix'] = np.array(shapefile.Reader(bankfixf).records(), dtype='float')[:,2]
    rec['width'] = np.array(shapefile.Reader(widthf).records(), dtype='float')[:,2]
    rec['bfq'] = np.array(shapefile.Reader(bfqf).records(), dtype='float')[:,2]
    rec['slope'] = np.array(shapefile.Reader(slopef).records(), dtype='float')[:,2]
    rec['depth'] = np.array(shapefile.Reader(depthf).records(), dtype='float')[:,2]

    print(rec.head())
    rec = rec.rename(columns = {'Unnamed: 0':'point'})
    print('rows',len(rec))
    if extent is not None:
        print('Clipping to extent',extent)
        dropvals = np.logical_or(
        np.logical_or(rec.lon<float(extent[0]),rec.lon > float(extent[1])), np.logical_or(rec.lat<float(extent[2]),rec.lat>float(extent[3])))
        rec = rec.drop(rec[dropvals].index)

    rec = rec.drop(columns=['strahler'])#,'Unnamed: 0'])

    # In this script sort so values go from downstream to upstream
    rec.sort_values(by=['reach','distance'],inplace=True)

    # work out us points of each link
    recgrp = rec.groupby('link')
    # Index for the upstream point of each reach
    usindex = {}
    # List of (duplicate) points to drop from dataframe
    droppoints = []
    # Lists of links and downstream links
    dslinks = []
    links   = []
    # First make list of ds dslinks
    for link, df in recgrp:
        dslinks.append(df.iloc[0]['dslink'])
        links.append(df.iloc[0]['link'])
    for link, df in recgrp:
        if link in dslinks:
            usindex[link] = df.iloc[-1].name
        # Drop downstream point
        droppoints.append(df.iloc[0].name)

    # Actually dont drop downstream points for each reach
    recgrp = rec.groupby('reach')
    for reach,df in recgrp:
        print(reach)
        dspoint = df.iloc[0].name
        droppoints.remove(dspoint)

    print('Dropping duplicate points exept at start of reaches')
    rec = rec.drop(droppoints)

    print(droppoints)
    # Create dataframes for dspoints and dx
    dspoints = np.zeros([len(rec)])
    rec['dspoint']=dspoints
    dx       = np.zeros([len(rec)])
    rec['dx']= dx

    lastdist = -1
    lastreach = -1
    lastpoint = -1

    # Now loop over items
    for index,val in rec.iterrows():
        if lastreach==val.reach:
            rec.loc[index,'dx'] = val.distance-lastdist
            # set downstream point for previous index
            rec.loc[index,'dspoint']=lastpoint
        else: # we are on a new reach
            # Need to work out ds point
            if val.dslink in usindex:
                # downstream point is the upstream point of the downstream link
                rec.loc[index,'dspoint']=rec.loc[usindex[val.dslink],'point']
                #rec.loc[index,'dx'] = val.distance - usindex[val.dslink]
                # The first point in the reach is a duplicate of the upstream point of the reach it drains into. Copy dx
                rec.loc[index,'dx'] = rec.loc[usindex[val.dslink],'dx']
            else:
                rec.loc[index,'dspoint']=-1
                rec.loc[index,'dx'] = 250.
        # Update lastpoint,dist,link,reach
        lastpoint = val.point
        lastdist = val.distance
        lastreach = val.reach

    # test write output
    rec.to_csv(output,index=False)

if __name__ == '__main__':
    joinchan_shell(sys.argv[1:])
