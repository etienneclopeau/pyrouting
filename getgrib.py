#!/usr/bin/python
"""Download GRIB files from NOAA.

Timestamps of downloaded files are preserved. Timestamps are saved in localtime.
"""
import os
import sys
import subprocess
import time
import datetime
import urllib2
import logging

import wgrib
import forecasts
from constants import *
import updater
import logger

def getTimeOfFile(file_name):
    """Get time of downloaded forecast in the forecasts directory.

    >>> getTimeOfFile('gfs.t06z.pgrb2f00')
    datetime.datetime(2009, 3, 24, 10, 27, 35)

    Args:
        file_name: name of file in downloads directory
    
    Returns:
        return time UTC 
    """
    gfs_file_path = '%s/%s/%s' % (PATH,DOWNLOADS_DIR, file_name)
    try:
        secs_since_epoch = os.path.getmtime(getForecastFilePath(file_name))
        time_struct = time.gmtime(secs_since_epoch)
        return datetime.datetime(*time_struct[:6])
    except OSError:
        return None

def isGRIBOnServerUpdated(run):
    """Check if the forecasts at the server is newer than the local version.

    The check is only made on the last forecast hour. Because this is the
    latest generated file. If the local forecasts doesn't exist, True is
    returned.

    Args:
        calculation_hour: hour of gfs calculation to check freshness of
    Returns:
        Boolean indicating if forecast at server is newer than the local.
    """
    file_name = getForecastFileName(run,'48')
    time, the_run = getLatestModifiedTimeNOAA()
    logging.info('isGribOnServerUpdated: checking %s' % file_name)
    try:
        local_time = getTimeOfFile(file_name)
        logging.info('isGribOnServerUpdated: NOAA time: %s, Local time: %s' % (time,local_time))
        # not local_time: if the file does not exist
        if(not local_time or time > local_time):
            logging.info('grib is updated')
            return True
        logging.info('grib is not updated')
        return False
    except (OSError, TypeError):
        # file does not exist locally
        return True

def getLatestModifiedTimeNOAA():
    """Get most recent forecast calculation hour at NOAA.
    """
    latest_run = CALC_HOURS[0]
    latest_time = getLastModifiedTime(getForecastFileName(latest_run,'48'))

    for run in CALC_HOURS[1:]:
        # Look at the last file in every run
        time = getLastModifiedTime(getForecastFileName(run,'48'))
        if time > latest_time:
            latest_time = time
            latest_run = run
    return latest_time, latest_run


def getLatestRunLocal():
    """Get most recent forecast calculation hour locally.

    Precondition: Last weather file available in each run.
    FIXME: CLEAN UP
    """
    # If the files are not yet downloaded
    latest_run = CALC_HOURS[0]
    latest_time = getLastModifiedTime(getForecastFileName(latest_run,'48'))

    for run in CALC_HOURS[1:]:
        # Look at the last file in every run
        time = getTimeOfFile(getForecastFileName(run,'48'))
        if time > latest_time:
            latest_time = time
            latest_run = run
    return latest_run

def getLastModifiedTime(file_name):
    """Get last modified time of...

    http://nomad3.ncep.noaa.gov/pub/gfs/rotating-0.5/<<file_name>>

    Args:
        file_name: file on server
    Returns:
        the time as a datetime
    """
    url_obj = urllib2.urlopen('http://nomad3.ncep.noaa.gov/pub/gfs/rotating-0.5/' + file_name)
    time_str = url_obj.headers.getheader('Last-Modified')
    # Python 2.4: datetime.strptime not supported
    time_struct = time.strptime(time_str, '%a, %d %b %Y %H:%M:%S GMT')
    return datetime.datetime(*time_struct[:6])

def downloadInventory(file_name):
    """Download inventory.

    Args:
        file_name: name of inventory to download    
    """
    cmd = '%s/get_inv.pl %s%s > %s' % (PATH,NOAA_URL,file_name,getForecastFilePath(file_name))
    runBash(cmd)
    
def downloadForecast(file_name):
    """Download grib forecast from the given url.

    Applies the filter string, to get relevant GRIB data.
    
    Args:
        file_name: name of file to download
    """
    logging.info('Downloading: %s%s' % (NOAA_URL,file_name))
    grib_output = '%s/%s/%s' % (PATH, DOWNLOADS_DIR, file_name)
    inv_name = '%s.inv' % (file_name)
    inv_path = '%s.inv' % (grib_output)
    downloadInventory(inv_name)
    get_grib_cmd = '%s/get_grib.pl %s%s %s' % (PATH, NOAA_URL,file_name,grib_output)
    filter = 'egrep "%s"' % (GRIB_FILTER)
    cmd = filter + ' < ' + inv_path + ' | ' + get_grib_cmd
    runBash(cmd)
    logging.info('Finished downloading: %s%s' % (NOAA_URL,file_name))

def downloadForecasts(file_names):
    for file_name in file_names:
        downloadForecast(file_name)    

def getFileNamesInRun(calc_hour):
    file_names = []
    for hour in FORECAST_HOURS:
        file_name = getForecastFileName(calc_hour, hour)
        file_names.append(file_name)
    return file_names

def getForecastFileName(calculation_hour, forecast_hour):
    return 'gfs.t' + calculation_hour + 'z.pgrb2f' + forecast_hour 

def getForecastFilePath(file_name):
    return '%s/%s/%s' % (PATH, DOWNLOADS_DIR, file_name)

def runBash(cmd):
    """Executes the given bash command.

    Args:
        cmd: command string to execute
    Returns:
        The stdout from the command.
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out = p.communicate()[0].strip()
    return out  #This is the stdout from the shell command 

def update_grib():
    """Update the grib files with the latest on the server.
    Update the GAE server.

    Files are only fetched if the local copy is stale.

    Returns:
        a boolean indicating wheter the files were updated.
    """
    time, run = getLatestModifiedTimeNOAA()
    logging.info('Last-Modified at NOAA: %s Run: %s' % (time,run))
    file_names = getFileNamesInRun(run)
    
    if isGRIBOnServerUpdated(run):
        downloadForecasts(file_names)
        return True
    return False

def main():
    try:
        sys.argv[1]
        logger.set_log_file_name('/users/dam/cronlogs/downloader.log')
        logger.enable_file_log()
    except Exception, e:
        logger.enable_console_log()
    logging.info('update gfs started')
    update_grib()
    logging.info('update gfs finised')

if __name__ == '__main__':
    main()
