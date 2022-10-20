#!/usr/bin/env python

import os
import sys
import re
import datetime
import subprocess

from optparse import OptionParser
from collections import OrderedDict

from CRABClient.UserUtilities import setConsoleLogLevel
from CRABClient.UserUtilities import getUsernameFromCRIC
from CRABClient.ClientUtilities import LOGLEVEL_MUTE
from CRABClient.UserUtilities import getConsoleLogLevel

from CRABAPI.RawCommand import crabCommand

# Usage: multicrab.py <analysis skim> <dataversion>
# Example: multicrab.py NanoAOD_Hplus2taunuAnalysisSkim.py 2017UL
# More: multicrab.py -h

datasetpath = os.path.join(os.environ['CMSSW_BASE'],"src/NanoAnalysis/NanoAODSkim/python")
sys.path.append(datasetpath)
workpath = os.path.join(os.environ['CMSSW_BASE'],"src/NanoAnalysis/NanoAODSkim/test")
sys.path.append(workpath)

skimConfig  = ""
dataVersion = ""
analysisName = ""
if os.path.exists(sys.argv[1]):
    skimConfig = sys.argv[1]
    analysis_re = re.compile("NanoAOD_(?P<name>\S+Analysis)Skim")
    match = analysis_re.search(sys.argv[1])
    analysisName = match.group("name")
if len(sys.argv) >= 3 and "20" in sys.argv[2]:
    dataVersion = sys.argv[2]

#================================================================================================
# Global Definitions
#================================================================================================
ss  = "\033[92m"
ns  = "\033[0;0m"
ts  = "\033[0;35m"
hs  = "\033[1;34m"
ls  = "\033[0;33m"
es  = "\033[1;31m"
cs  = "\033[0;44m\033[1;37m"
cys = "\033[1;36m"

PBARLENGTH = 20

######################################################
# DATASETS
######################################################
datasets = []
#if dataVersion:
import importlib

if not skimConfig == "":

    analysisDatasets = importlib.import_module("datasets_%s"%analysisName)
    datasets = analysisDatasets.getDatasets(dataVersion)

    skimModule = importlib.import_module(skimConfig.replace(".py",""))
    if hasattr(skimModule, 'ANALYSISNAME'):
        analysisName = skimModule.ANALYSISNAME
print analysisName

JSONPATH = os.path.realpath(os.path.join(os.getcwd(),"../data"))
print "JSONPATH",JSONPATH

alldatasets = datasets

def FindDataset(path):
    name_re = re.compile("multicrab\S+/(?P<name>\S+)")
    match = name_re.search(path)
    if match:
        name = match.group("name")
        for dset in alldatasets:
            dsetname = dset.getName()
            if len(dsetname) == 0:
                dsetname = GetRequestName(dset)
            if dsetname == name:
                return dset
    print path,"not found"



######################################################

#================================================================================================
# Class Definition
#================================================================================================
class colors:
    '''
    \033[  Escape code, this is always the same
    1 = Style, 1 for normal.
    32 = Text colour, 32 for bright green.
    40m = Background colour, 40 is for black.

    WARNING:
    Python doesn't distinguish between 'normal' characters and ANSI colour codes, which are also characters that the terminal interprets.
    In other words, printing '\x1b[92m' to a terminal may change the terminal text colour, Python doesn't see that as anything but a set of 5 characters.
    If you use print repr(line) instead, python will print the string literal form instead, including using escape codes for non-ASCII printable characters
    (so the ESC ASCII code, 27, is displayed as \x1b) to see how many have been added.

    You'll need to adjust your column alignments manually to allow for those extra characters.
    Without your actual code, that's hard for us to help you with though.

    Useful Links:
    http://ozzmaker.com/add-colour-to-text-in-python/
    http://stackoverflow.com/questions/15580303/python-output-complex-line-with-floats-colored-by-value
    '''
    colordict = {
                'RED'     :'\033[91m',
                'GREEN'   :'\033[92m',
                'BLUE'    :'\033[34m',
                'GRAY'    :'\033[90m',
                'WHITE'   :'\033[00m',
                'ORANGE'  :'\033[33m',
                'CYAN'    :'\033[36m',
                'PURPLE'  :'\033[35m',
                'LIGHTRED':'\033[91m',
                'PINK'    :'\033[95m',
                'YELLOW'  :'\033[93m',
                }
    if sys.stdout.isatty():
        RED      = colordict['RED']
        GREEN    = colordict['GREEN']
        BLUE     = colordict['BLUE']
        GRAY     = colordict['GRAY']
        WHITE    = colordict['WHITE']
        ORANGE   = colordict['ORANGE']
        CYAN     = colordict['CYAN']
        PURPLE   = colordict['PURPLE']
        LIGHTRED = colordict['LIGHTRED']
        PINK     = colordict['PINK']
        YELLOW   = colordict['YELLOW']
    else:
        RED, GREEN, BLUE, GRAY, WHITE, ORANGE, CYAN, PURPLE, LIGHTRED, PINK, YELLOW = '', '', '', '', '', '', '', '', '', '', ''

#================================================================================================
# Class Definition
#================================================================================================
class Report:
    def __init__(self, name, allJobs, idle, retrieved, running, finished, failed, transferring, retrievedLog, retrievedOut, eosLog, eosOut, status, dashboardURL):
        '''
        Constructor
        '''
        self.name            = name
        self.allJobs         = str(allJobs)
        self.retrieved       = str(retrieved)
        self.running         = str(running)
        self.dataset         = self.name.split("/")[-1]
        self.dashboardURL    = dashboardURL
        self.status          = self.GetTaskStatusStyle(status)
        self.finished        = str(len(finished))
        self.failed          = failed
        self.idle            = idle
        self.transferring    = transferring
        self.retrievedLog    = str(len(retrievedLog))
        self.retrievedOut    = str(len(retrievedOut))
        self.eosLog          = eosLog
        self.eosOut          = eosOut
        return

    def Print(self, printHeader=True):
        '''
        Simple function to print report.
        '''
        name = os.path.basename(self.name)
        while len(name) < 30:
            name += " "

        fName = GetSelfName()
        cName = self.__class__.__name__
        name  = fName + ": " + cName
        if printHeader:
            print "=== ", name
        msg  = '{:<20} {:<40}'.format("\t %sDataset"           % (colors.WHITE) , ": " + self.dataset)
        msg += '\n {:<20} {:<40}'.format("\t %sRetrieved Jobs" % (colors.WHITE) , ": " + self.retrieved + " / " + self.allJobs)
        msg += '\n {:<20} {:<40}'.format("\t %sStatus"         % (colors.WHITE) , ": " + self.status)
        msg += '\n {:<20} {:<40}'.format("\t %sDashboard"      % (colors.WHITE) , ": " + self.dashboardURL)
        print msg
        return


    def GetURL():
        return self.dashboardURL

    def GetTaskStatusStyle(self, status):
        '''
        NEW, RESUBMIT, KILL: Temporary statuses to indicate the action ('submit', 'resubmit' or 'kill') that has to be applied to the task.
        QUEUED: An action ('submit', 'resubmit' or 'kill') affecting the task is queued in the CRAB3 system.
        SUBMITTED: The task was submitted to HTCondor as a DAG task. The DAG task is currently running.
        SUBMITFAILED: The 'submit' action has failed (CRAB3 was unable to create a DAG task).
        FAILED: The DAG task completed all nodes and at least one is a permanent failure.
        COMPLETED: All nodes have been completed
        KILLED: The user killed the task.
        KILLFAILED: The 'kill' action has failed.
        RESUBMITFAILED: The 'resubmit' action has failed.
        '''
        # Remove all whitespace characters (space, tab, newline, etc.)
        status = ''.join(status.split())
        if status == "NEW":
            status = "%s%s%s" % (colors.BLUE, status, colors.WHITE)
        elif status == "RESUBMIT":
            status = "%s%s%s" % (colors.BLUE, status, colors.WHITE)
        elif status == "QUEUED":
            status = "%s%s%s" % (colors.GRAY, status, colors.WHITE)
        elif status == "SUBMITTED":
            status = "%s%s%s" % (colors.BLUE, status, colors.WHITE)
        elif status == "SUBMITFAILED":
            status = "%s%s%s" % (colors.RED, status, colors.WHITE)
        elif status == "FAILED":
            status = "%s%s%s" % (colors.RED, status, colors.WHITE)
        elif status == "COMPLETED":
            status = "%s%s%s" % (colors.GREEN, status, colors.WHITE)
        elif status == "KILLED":
            status = "%s%s%s" % (colors.ORANGE, status, colors.WHITE)
        elif status == "KILLFAILED":
            status = "%s%s%s" % (colors.ORANGE, status, colors.WHITE)
        elif status == "RESUBMITFAILED":
            status = "%s%s%s" % (colors.ORANGE, status, colors.WHITE)
        elif status == "?":
            status = "%s%s%s" % (colors.PINK, status, colors.WHITE)
        elif status == "UNDETERMINED":
            status = "%s%s%s" % (colors.CYAN, status, colors.WHITE)
        elif status == "UNKNOWN":
            status = "%s%s%s" % (colors.LIGHTRED, status, colors.WHITE)
        else:
            raise Exception("Unexpected task status %s." % (status) )
        return status

def PrintProgressBar(taskName, iteration, total, suffix=""):
    '''
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    barLength   - Optional  : character length of bar (Int)
    '''
    Verbose("PrintProgressBar()")

    iteration      += 1 # since enumerate starts from 0
    prefix          = taskName
    decimals        = 1
    barLength       = PBARLENGTH
    txtSize         = 50
    fillerSize      = txtSize - len(taskName)
    if fillerSize < 0:
        fillerSize = 0
    filler          = " "*fillerSize
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '=' * filledLength + '-' * (barLength - filledLength)
    if iteration == 1:
        sys.stdout.write('\n')
    sys.stdout.write('\r\t%s%s |%s| %s%s %s' % (prefix, filler, bar, percents, '%', suffix)),
    sys.stdout.flush()
    return

def ClearProgressBar():
    Verbose("ClearProgressBar()")
    # The \r is the carriage return. (Option: You need the comma at the end of the print statement to avoid automatic newline)
    print '\r%s' % (" "*185)
    return


def FinishProgressBar():
    Verbose("FinishProgressBar()")
    sys.stdout.write('\n')
    return

def EnsurePathDoesNotExist(taskDirName, requestName):
    '''
    Ensures that file does not already exist
    '''
    filePath = os.path.join(taskDirName, requestName)

    if not os.path.exists(filePath):
        return
    else:
        msg = "File '%s' already exists!" % (filePath)
        Print(msg + "\n\tProceeding to overwrite file.")
    return

def GetCMSSW():
    '''
    Get a command-line-friendly format of the CMSSW version currently use.
    https://docs.python.org/2/howto/regex.html
    '''

    # Get the current working directory
    pwd = os.getcwd()

    # Create a compiled regular expression object
    cmssw_re = re.compile("/CMSSW_(?P<version>\S+?)/")

    # Scan through the string 'pwd' & look for any location where the compiled RE 'cmssw_re' matches
    match = cmssw_re.search(pwd)

    # Return the string matched by the RE. Convert to desirable format
    version = ""
    if match:
        version = match.group("version")
        version = version.replace("_","")
        version = version.replace("pre","p")
        version = version.replace("patch","p")
    return version

def CreateTaskDir(pset,dirName):
    '''
    Create the CRAB task directory and copy inside it the PSET to be used for the CRAB job.
    '''
    # Copy file to be used (and others to be tracked) to the task directory
    cmd = "cp %s %s" %(pset, dirName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)

        os.system(cmd)
    else:
        pass

####    # Write the commit id, "git status", "git diff" command output the directory created for the multicrab task
####    gitFileList = git.writeCodeGitInfo(dirName, False)
    return

def GetRequestName(dataset):
    '''
    Return the file name and path to an (empty) crabConfig_*.py file where "*"
    contains the dataset name and other information such as tune, COM, Run number etc..
    of the Data or MC sample used
    '''
    #print "check GetRequestName",dataset.getName()
    if len(dataset.getName()) > 0:
        return dataset.getName()

    # Create compiled regular expression objects
    datadataset_re = re.compile("^/(?P<name>\S+?)/(?P<run>Run\S+?)/")
    mcdataset_re   = re.compile("^/(?P<name>\S+?)/")
    tune_re        = re.compile("(?P<name>\S+)_Tune")
    tev_re         = re.compile("(?P<name>\S+)_13TeV")
    ext_re         = re.compile("(?P<name>_ext\d+)-")
    runRange_re    = re.compile("Cert_(?P<RunRange>\d+-\d+)_")
    # runRange_re    = re.compile("Cert_(?P<RunRange>\d+-\d+)_13TeV_PromptReco_Collisions15(?P<BunchSpacing>\S*)_JSON(?P<Silver>(_\S+|))\."\)
    # runRange_re    = re.compile("Cert_(?P<RunRange>\d+-\d+)_13TeV_PromptReco_Collisions15(?P<BunchSpacing>\S*)_JSON")
    # runRange_re    = re.compile("Cert_(?P<RunRange>\d+-\d+)_13TeV_PromptReco_Collisions15_(?P<BunchSpacing>\d+ns)_JSON_v")

    # Scan through the string 'dataset.URL' & look for any location where the compiled RE 'mcdataset_re' matches
    #print "check url",dataset.URL
    match = mcdataset_re.search(dataset.URL)
    if dataset.isData():
        match = datadataset_re.search(dataset.URL)

    # Append the dataset name
    requestName = "analysis"
    if match:
        requestName = match.group("name")
    firstName = requestName
    #print "check name",requestName
    # Append the Run number (for Data samples only)
    #print "check isData",dataset.isData()
    if dataset.isData():
        requestName+= "_"
        requestName+= match.group("run")
    #print "check run",requestName
    # Append the MC-tune (for MC samples only)
    tune_match = tune_re.search(requestName)
    if tune_match:
        requestName = tune_match.group("name")

    # Append the COM Energy (for MC samples only)
    tev_match = tev_re.search(requestName)
    if tev_match:
        requestName = tev_match.group("name")

    # Simple hack to prevent overwrite of special TT samples
####    requestName = GetTTbarSystematicsName(firstName, requestName)

    # Append the Ext
    ext_match = ext_re.search(dataset.URL)
    if ext_match:
        requestName+=ext_match.group("name")

    # Append the Run Range (for Data samples only)
    if dataset.isData():
        runRangeMatch = runRange_re.search(dataset.lumiMask)
        if runRangeMatch:
            runRange= runRangeMatch.group("RunRange")
            runRange = runRange.replace("-","_")
            #bunchSpace = runRangeMatch.group("BunchSpacing")
            requestName += "_" + runRange #+ bunchSpace
            #Ag = runRangeMatch.group("Silver")
            #if Ag == "_Silver": # Use  chemical element of silver (Ag)
            #    requestName += Ag

    # Finally, replace dashes with underscores
    requestName = requestName.replace("-","_")
    return requestName
"""
def CreateCfgFile(dataset, taskDirName, requestName, infilePath, opts):
    '''
    Creates a CRAB-specific configuration file which will be used in the submission
    of a job. The function uses as input a generic cfg file which is then customised
    based on the dataset type used.

    infilePath = "crabConfig.py"
    '''

    outfilePath = os.path.join(taskDirName, "crabConfig_" + requestName + ".py")

    # Check that file does not already exist
    EnsurePathDoesNotExist(taskDirName, outfilePath)

    # Open input file (read mode) and output file (write mode)
    fIN  = open(infilePath , "r")
    fOUT = open(outfilePath, "w")

    # Create compiled regular expression objects
    crab_requestName_re     = re.compile("config.General.requestName")
    crab_workArea_re        = re.compile("config.General.workArea")
    crab_transferOutputs_re = re.compile("config.General.transferOutputs")
    crab_transferLogs_re    = re.compile("config.General.transferLogs")
    crab_pset_re            = re.compile("config.JobType.psetName")
    crab_psetParams_re      = re.compile("config.JobType.pyCfgParams")
    crab_dataset_re         = re.compile("config.Data.inputDataset")
    crab_split_re           = re.compile("config.Data.splitting")
    crab_splitunits_re      = re.compile("config.Data.unitsPerJob")
    crab_dbs_re             = re.compile("config.Data.inputDBS")
    crab_storageSite_re     = re.compile("config.Site.storageSite")
    crab_outLFNDirBase_re   = re.compile("config.Data.outLFNDirBase")

    # For-loop: All line of input fine
    for line in fIN:

        # Skip lines which are commented out
        if line[0] == "#":
            continue

        # Set the "inputDataset" field which specifies the name of the dataset. Can be official CMS dataset or a dataset produced by a user.
        match = crab_dataset_re.search(line)
        if match:
            line = "config.Data.inputDataset = '" + dataset.URL + "'\n"

        # Set the "requestName" field which specifies the request/task name. Used by CRAB to create a project directory (named crab_<requestName>)
        match = crab_requestName_re.search(line)
        if match:
            line = "config.General.requestName = '" + requestName + "'\n"

        # Set the "workArea" field which specifies the (full or relative path) where to create the CRAB project directory.
        match = crab_workArea_re.search(line)
        if match:
            line = "config.General.workArea = '" + taskDirName + "'\n"

        # Set the "psetName" field which specifies the name of the CMSSW pset_cfg.py file that will be run via cmsRun.
        match = crab_pset_re.search(line)
        if match:
            line = "config.JobType.psetName = '" + opts.pset +"'\n"
"""
def CreateCfgFile(dataset, taskDirName, requestName, infilePath, opts):
    '''
    Creates a CRAB-specific configuration file which will be used in the submission
    of a job. The function uses as input a generic cfg file which is then customised
    based on the dataset type used.

    infilePath = "crabConfig.py"
    '''

    outfilePath = os.path.join(taskDirName, "crabConfig_" + requestName + ".py")

    # Check that file does not already exist
    EnsurePathDoesNotExist(taskDirName, outfilePath)

    # Open input file (read mode) and output file (write mode)
    fIN  = open(infilePath , "r")
    fOUT = open(outfilePath, "w")

    # Create compiled regular expression objects
    crab_requestName_re     = re.compile("config.General.requestName")
    crab_workArea_re        = re.compile("config.General.workArea")
    crab_transferOutputs_re = re.compile("config.General.transferOutputs")
    crab_transferLogs_re    = re.compile("config.General.transferLogs")
    crab_pset_re            = re.compile("config.JobType.psetName")
    crab_psetParams_re      = re.compile("config.JobType.pyCfgParams")
    crab_dataset_re         = re.compile("config.Data.inputDataset")
    crab_split_re           = re.compile("config.Data.splitting")
    crab_splitunits_re      = re.compile("config.Data.unitsPerJob")
    crab_dbs_re             = re.compile("config.Data.inputDBS")
    crab_storageSite_re     = re.compile("config.Site.storageSite")
    crab_outLFNDirBase_re   = re.compile("config.Data.outLFNDirBase")

    # For-loop: All line of input fine
    for line in fIN:

        # Skip lines which are commented out
        if line[0] == "#":
            continue

        # Set the "inputDataset" field which specifies the name of the dataset. Can be official CMS dataset or a dataset produced by a user.
        match = crab_dataset_re.search(line)
        if match:
            line = "config.Data.inputDataset = '" + dataset.URL + "'\n"

        # Set the "requestName" field which specifies the request/task name. Used by CRAB to create a project directory (named crab_<requestName>)
        match = crab_requestName_re.search(line)
        if match:
            line = "config.General.requestName = '" + requestName + "'\n"

        # Set the "workArea" field which specifies the (full or relative path) where to create the CRAB project directory.
        match = crab_workArea_re.search(line)
        if match:
            line = "config.General.workArea = '" + taskDirName + "'\n"

        # Set the "psetName" field which specifies the name of the CMSSW pset_cfg.py file that will be run via cmsRun.
        match = crab_pset_re.search(line)
        if match:
            line = "config.JobType.psetName = '" + opts.pset +"'\n"

        # Set the "pyCfgParams" field which contains list of parameters to pass to the pset_cfg.py file.
        match = crab_psetParams_re.search(line)
        if match:
            line = "config.JobType.pyCfgParams = ['dataVersion=" + dataset.dataVersion +"']\n"

        # Set the "inputDBS" field which specifies the URL of the DBS reader instance where the input dataset is published
        match = crab_dbs_re.search(line)
        if match:
            line = "config.Data.inputDBS = '" + dataset.DBS + "'\n"

        # Set the "storageSite" field which specifies the destination site for submission [User MUST have write access to destination site!]
        match = crab_storageSite_re.search(line)
        if match:
            line = "config.Site.storageSite = '" + opts.storageSite + "'\n"

        match = crab_outLFNDirBase_re.search(line)
        if match:
            if opts.dirName.endswith("/"):
                mcrabDir = os.path.basename(opts.dirName[:-1])  # exclude last "/", either-wise fails
            else:
                mcrabDir = os.path.basename(opts.dirName)
            fullDir  = "/store/user/%s/CRAB3_TransferData/%s" % (getUsernameFromCRIC(), mcrabDir) # NOT getpass.getuser()
            line     = "config.Data.outLFNDirBase = '" + fullDir + "'\n"

        # Only if dataset is real data
        if dataset.isData():

            # Set the "splitting" field which specifies the mode to use to split the task in jobs ('FileBased', 'LumiBased', or 'EventAwareLumiBased')
            match = crab_split_re.search(line)
            if match:
####                line = "config.Data.splitting = 'LumiBased'\n"
                line+= "config.Data.lumiMask = '"+ dataset.lumiMask + "'\n"

            # Set the "unitsPerJob" field which suggests (but not impose) how many files, lumi sections or events to include in each job.
            match = crab_splitunits_re.search(line)
####            if match:
####                line = "config.Data.unitsPerJob = 250\n"
#                line = "config.Data.unitsPerJob = 1\n"
        else:
            pass

        # Write line to the output file
        fOUT.write(line)

    # Close input and output files
    fOUT.close()
    fIN.close()

    return

def SubmitTaskDir(taskDirName, requestName):
    '''
    Submit a given CRAB task using the specific cfg file.
    '''

    outfilePath = os.path.join(taskDirName, "crabConfig_" + requestName + ".py")

    # Submit the CRAB task
    cmd_submit = "crab submit " + outfilePath
    os.system(cmd_submit)

    # Rename the CRAB task directory (remove "crab_" from its name)
    cmd_mv = "mv " + os.path.join(taskDirName, "crab_" + requestName) + " " + os.path.join(taskDirName, requestName)
    #print "check cmd_mv",cmd_mv
    os.system(cmd_mv)
    return

def GetTaskDirName(analysis, version, datasets, opts):
    '''
    Get the name of the CRAB task directory to be created. For the user's benefit this
    will include the CMSSW version and possibly important information from
    the dataset used, such as the bunch-crossing time.
    '''
    # Constuct basic task directory name
    dirName = "multicrab"
    dirName+= "_"  + analysis
    dirName+= "_v" + version

    # Add dataset-specific info, like bunch-crossing info
    bx_re = re.compile("\S+(?P<bx>\d\dns)_\S+")
    match = bx_re.search(datasets[0].URL)
    if match:
        dirName+= "_"+match.group("bx")

    run_re = re.compile("(?P<run>Run20\d\d)(?P<letter>\S)")
    runs = ""
    runletters = []
    for d in datasets:
        if not d.isData():
            continue
        match = run_re.search(d.URL)
        if match and len(runs) == 0:
            runs = match.group("run")
        if match and match.group("letter") not in runletters:
            runletters.append(match.group("letter"))
    for l in runletters:
        runs+=l
    dirName+= "_"+runs

    # Append the creation time to the task directory name
    # time = datetime.datetime.now().strftime("%d%b%Y_%Hh%Mm%Ss")
    time = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    dirName+= "_" + time

    # If directory already exists (resubmission)
    if os.path.exists(opts.dirName) and os.path.isdir(opts.dirName):
        dirName = opts.dirName

    return dirName

def CreateJob(opts, args):
    '''
    Create & submit a CRAB task, using the user-defined PSET and list of datasets.
    '''

    # Get general info
    version      = GetCMSSW()
    analysis     = opts.pset.replace('.py','')
    if opts.name:
        analysis     = opts.name.replace('.py','')
    #datasets     = datasets #DatasetGroup(analysis).GetDatasetList()
    taskDirName  = GetTaskDirName(analysis, version, datasets, opts)
    opts.dirName = taskDirName

    # Create CRAB task diractory
    CreateTaskDir(opts.pset,taskDirName)

    # For-loop: All datasets
    for dataset in datasets:
        requestName = GetRequestName(dataset)
        fullDir     = taskDirName + "/" + requestName

        if os.path.exists(fullDir) and os.path.isdir(fullDir):
            continue

        CreateCfgFile(dataset, taskDirName, requestName, "crab_cfg.py", opts)
        SubmitTaskDir(taskDirName, requestName)
        if dataset.isData():
            cmd = "cp %s %s"%(dataset.lumiMask,os.path.join(fullDir,'inputs'))
            print(cmd)
            os.system(cmd)

    print taskDirName
    return 0

def CheckJob(opts, args):
    '''
    Check status, retrieve, resubmit, kill CRAB tasks.
    '''

    # Force crabCommand to stay quite
    setConsoleLogLevel(LOGLEVEL_MUTE)

    # Retrieve the current crabCommand console log level:
    crabConsoleLogLevel = getConsoleLogLevel()

    # Get the paths for the datasets (absolute paths)
    datasets = GetDatasetsPaths(opts)
    if len(datasets) < 1:
        Print("Found %s CRAB tasks under %s! Exit .." % (opts.dirName) )
        return

    # Create a dictionary to map TaskName <-> CRAB Report
    reportDict = GetCrabReportDictionary(datasets,opts)

    # Print a summary table with information on each CRAB Task
    PrintTaskSummary(reportDict)
    return

def GetDatasetsPaths(opts):
    '''
    Return the absolute path for each task/dataset located inside
    the working multi-CRAB directory. If the --inlcudeTask or the
    --excludeTask options are used, they are taken into consideration
    accordingly.
    '''

    # Get the multi-CRAB working dir
    multicrabDirPath = [opts.dirName]

    # Get the absolute path for each task(=dataset)
    datasetsDirPaths = GetDatasetAbsolutePaths(multicrabDirPath)

    # Check include/exclude options to get final datasets list
    datasets = GetIncludeExcludeDatasets(datasetsDirPaths, opts)

    return datasets

def GetDatasetAbsolutePaths(datasetdirs):

    datasets = []
    # For-loop: All CRAB dirs (absolute paths)
    for d in datasetdirs:

        if os.path.exists( os.path.join(d, "results") ):
            datasets.append(d)

        # Get the contents of this directory
        cands = Execute("ls -tr %s"%d)

        # For-loop: All directory contents
        for c in cands:
            path = os.path.join(d, c)
            # Get all dataset directories
            if os.path.exists( os.path.join(path, "results") ):
                datasets.append(path)
    return datasets

def Execute(cmd):
    '''
    Executes a given command and return the output.
    '''
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)

    stdin  = p.stdout
    stdout = p.stdout
    ret    = []
    for line in stdout:
        ret.append(line.replace("\n", ""))

    stdout.close()
    return ret

def GetIncludeExcludeDatasets(datasets, opts):
    '''
    Does nothing by default, unless the user specifies a dataset to include (--includeTasks <datasetNames>) or
    to exclude (--excludeTasks <datasetNames>) when executing the script. This function filters for the inlcude/exclude
    datasets and returns the lists of datasets and baseNames to be used further in the program.
    '''

    # Initialise lists
    newDatasets  = []

    # Exclude datasets
    if opts.excludeTasks != "None":
        tmp = []
        exclude = GetRegularExpression(opts.excludeTasks)

        for d in datasets:
            task  = GetBasename(d)
            found = False

            for e_re in exclude:
                if e_re.search(task):
                    found = True
                    break
            if found:
                continue
            newDatasets.append(d)
        return newDatasets

    # Include datasets
    if opts.includeTasks != "None":
        tmp = []
        include = GetRegularExpression(opts.includeTasks)

        for d in datasets:
            task  = GetBasename(d)
            found = False

            for i_re in include:
                if i_re.search(task):
                    found = True
                    break
            if found:
                newDatasets.append(d)
        return newDatasets

    return datasets

def GetCrabReportDictionary(datasets,opts):
    '''
    Loops over all datasets paths.
    Retrieves the report object for the given task
    and saves it into a dictionary, thus mapping the
    task name (basename of dataset path) to the CRAB
    report for that task.
    '''

    reportDict = {}
    # For-loop: All (absolute) paths of the datasets
    for index, d in enumerate(datasets):

        # Check if task is in "DONE" state
        if GetTaskStatusBool(d):
            continue

        # Get the CRAB task report & add to dictionary (retrieves job output!)
        report = GetTaskReports(d, opts)
        reportDict[d.split("/")[-1]] = report

    return reportDict

def PrintTaskSummary(reportDict):
    '''
    Print a summary table of all submitted tasks with their information.
    The purpose it to easily determine which jobs are done, running and failed.
    '''
    Verbose("PrintTaskSummary()")

    reports  = []
    msgAlign = "{:<3} {:<60} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16}"
    header   = msgAlign.format("#", "Task",
                               "%s%s" % (colors.GRAY  , "Idle"    ),
                               "%s%s" % (colors.RED   , "Failed"  ),
                               "%s%s" % (colors.ORANGE, "Running" ),
                               "%s%s" % (colors.ORANGE, "Transfer"),
                               "%s%s" % (colors.WHITE , "Done"    ),
                               "%s%s" % (colors.PURPLE, "Logs"    ),
                               "%s%s" % (colors.BLUE  , "Out"     ),
                               "%s%s" % (colors.CYAN  , "Logs"    ),
                               "%s%s" % (colors.CYAN  , "Out"     ),
                               "%s%s" % (colors.WHITE , "Status"  ),
                               )
    hLine = colors.WHITE + "="*175
    reports.append(hLine)
    reports.append(header)
    reports.append(hLine)

    # Alphabetical sorting of tasks
    ReportDict = OrderedDict(sorted(reportDict.items(), key=lambda t: t[0]))
    # For-loop: All datasets (key) and corresponding status (value)
    for i, dataset in enumerate(ReportDict):
        report     = reportDict[dataset]
        index      = i+1
        task       = dataset
        status     = report.status
        idle       = '{0: >3}'.format(report.idle)
        allJobs    = '{0: <3}'.format(report.allJobs)
        running    = '{0: >3}'.format(report.running)
        finished   = '{0: >3}'.format(report.finished)
        transfer   = '{0: >3}'.format(report.transferring)
        failed     = '{0: >3}'.format(len(report.failed))
        rLogs      = '{0: >3}'.format(report.retrievedLog)
        rOutput    = '{0: >3}'.format(report.retrievedOut)
        rLogsEOS   = '{0: >3}'.format(report.eosLog)
        rOutputEOS = '{0: >3}'.format(report.eosOut)
        line = msgAlign.format(index, task,
                               "%s%s/%s" % (colors.GRAY  , idle      , allJobs),
                               "%s%s/%s" % (colors.RED   , failed    , allJobs),
                               "%s%s/%s" % (colors.ORANGE, running   , allJobs),
                               "%s%s/%s" % (colors.ORANGE, transfer  , allJobs),
                               "%s%s/%s" % (colors.WHITE , finished  , allJobs),
                               "%s%s/%s" % (colors.PURPLE, rLogs     , allJobs),
                               "%s%s/%s" % (colors.BLUE  , rOutput   , allJobs),
                               "%s%s/%s" % (colors.CYAN  , rLogsEOS  , allJobs),
                               "%s%s/%s" % (colors.CYAN  , rOutputEOS, allJobs),
                               "%s"   % (status), #already with colour
                               )
        reports.append(line)
    reports.append(hLine)

    # For-loop: All lines in report table
    print
    for r in reports:
        print r
    print
    return

def JobList(jobs):
    Verbose("JobList()")

    joblist = ""
    for i,e in enumerate(sorted(jobs)):
        joblist += str(e)
        if i < len(jobs)-1:
            joblist += ","
    return joblist

def RetrievedFiles(taskDir, crabResults, dashboardURL, printTable, opts):
    '''
    Determines whether the jobs Finished (Success or Failure), and whether
    the logs and output files have been retrieved. Returns all these in form
    of lists. The list of tuple crabResults contains the jobId and its status.
    For example:
    crabResults = [['finished', 1], ['finished', 2], ['finished', 3] ] #obsolete
    '''
    Verbose("RetrievedFiles()", True)

    # Initialise variables
    retrievedLog       = []
    retrievedOut       = []
    retrievedOutMerged = []
    eosLog             = 0
    eosOut             = 0
    eosOutMerged       = 0
    finished           = []
    failed             = []
    transferring       = 0
    running            = 0
    idle               = 0
    unknown            = 0
    dataset            = taskDir.split("/")[-1]
    nJobs              = GetTotalJobsFromStatus(crabResults)
    missingOuts        = []
    missingLogs        = []

    # For-loop:All CRAB results
    for index, jobId in enumerate(crabResults['jobs']):

        stateDict = crabResults['jobs'][jobId]

        # Inform user of progress (especially if opts.filesInEOS is enabled)
        PrintProgressBar(os.path.basename(taskDir), index, nJobs )

        # Get the job ID and status
        jobStatus = stateDict['State']

        Verbose("Investigating jobId=%s with status=%s" % (jobId, jobStatus))
        # Assess the jobs status individually
        if jobStatus == 'finished':
            finished.append(jobId)
            # Count Output & Logfiles (EOS)
            if opts.filesInEOS:
                taskDirEOS  = GetEOSDir(taskDir, opts)
                foundLogEOS = ExistsEOS(taskDirEOS, "log", "cmsRun_%s.log.tar.gz" % jobId, opts)
                foundOutEOS = ExistsEOS(taskDirEOS, ""   , "miniaod2tree_%s.root" % jobId, opts)
                Verbose("foundLogEOS=%s , foundOutEOS=%s" % (foundLogEOS, foundOutEOS))
                if foundLogEOS:
                    eosLog += 1
                if foundOutEOS:
                    eosOut += 1
            else:
                eosLog = "?"
                eosOut = "?"
                pass

            # Count Output & Logfiles (local)
            foundLog = Exists(taskDir, "cmsRun_%s.log.tar.gz" % jobId)
            foundOut = Exists(taskDir, "\S+_%s.root" % jobId)
            if foundLog:
                retrievedLog.append(jobId)
                exitCode = CheckTaskReport(taskDir, jobId, opts)
                if not exitCode == 0:
                    Verbose("Found failed job for task=%s with jobId=%s and exitCode=%s" % (taskDir, jobId, exitCode) )
                    failed.append( jobId )
            if foundOut:
                retrievedOut.append(jobId)
            if foundLog and not foundOut:
                missingOuts.append( jobId )
            if foundOut and not foundLog:
                missingLogs.append( jobId )
        elif jobStatus == 'failed':
            failed.append( jobId )
        elif jobStatus == 'transferring':
            transferring += 1
        elif jobStatus == 'idle':
            idle += 1
        elif jobStatus == 'running':
            running+= 1
        else:
            unknown+= 1
    failed = list(set(failed))

    # Remove the progress bar once finished
    ClearProgressBar()

    # Count merged files (local)
    retrievedOutMerged = [f for f in os.listdir(os.path.join(taskDir, "results"))]
    # Count merged files (EOS)
    if opts.filesInEOS:
        taskDirEOS = GetEOSDir(taskDir, opts)
        cmd = ConvertCommandToEOS("ls", opts) + " " + taskDirEOS + " | grep histograms- | wc -l"
        Verbose(cmd, True)
        # Check if directory does not exist!
        if "Unable to stat" in cmd:
            eosOutMerged = 0
        else:
            eosOutMerged = Execute(cmd)[0] # just a number

    # Print results in a nice table
    reportTable = GetReportTable(taskDir, nJobs, running, transferring, finished, unknown, failed, idle, retrievedLog, retrievedOut, retrievedOutMerged, eosLog, eosOut, eosOutMerged)
    if printTable:
        for r in reportTable:
            Print(r, False)

    # Sanity check
    status = GetTaskStatus(taskDir).replace("\t", "")
    if opts.verbose and status == "COMPLETED":
        if len(missingLogs) > 0:
            Print( "Missing log file(s) job ID: %s" % missingLogs)
        if len(missingOuts) > 0:
            Print( "Missing output files(s) job ID: %s" % missingOuts)

    # Print the dashboard url
    if opts.url:
        Print(dashboardURL, False)
    return idle, running, finished, transferring, failed, retrievedLog, retrievedOut, retrievedOutMerged, eosLog, eosOut, eosOutMerged

def GetReportTable(taskDir, nJobs, running, transferring, finished, unknown, failed, idle, retrievedLog, retrievedOut, retrievedOutMerged, eosLog, eosOut, eosOutMerged):
    '''
    Takes various info on the status of a CRAB job and return a neat table.
    '''
    Verbose("GetReportTable()", True)

    nTotal    = str(nJobs)
    nRun      = str(running)
    nTransfer = str(transferring)
    nFinish   = str(len(finished))
    nUnknown  = str(unknown)
    nFail     = str(len(failed))
    nIdle     = str(idle)
    nLogs     = str(len(retrievedLog))#''.join( str(retrievedLog).split() )
    nOut      = str(len(retrievedOut))#''.join( str(retrievedOut).split() )
    nOutM     = str(len(retrievedOutMerged))
    nLogsEOS  = ''.join( str(eosLog).split() )
    nOutEOS   = ''.join( str(eosOut).split() )
    nOutEOSM  = eosOutMerged
    txtAlign  = "{:<31} {:>4} {:<1} {:<4}"
    dataset   = taskDir.split("/")[-1]
    length    = 65 #len(dataset)
    hLine     = "="*length
    status    = GetTaskStatus(taskDir).replace("\t", "")
    txtAlignB = "{:<%s}" % (length)
    header    = txtAlignB.format(dataset)

    table = []
    table.append(hLine)
    table.append(header)
    table.append(hLine)
    table.append( txtAlign.format("%sIdle"                  % (colors.GRAY  ), nIdle    , "/", nTotal ) )
    table.append( txtAlign.format("%sUnknown"               % (colors.GRAY  ), nUnknown , "/", nTotal ) )
    table.append( txtAlign.format("%sFailed"                % (colors.RED   ), nFail    , "/", nTotal ) )
    table.append( txtAlign.format("%sRunning"               % (colors.ORANGE), nRun     , "/", nTotal ) )
    table.append( txtAlign.format("%sTransferring"          % (colors.BLUE  ), nTransfer, "/", nTotal ) )
    table.append( txtAlign.format("%sDone"                  % (colors.WHITE ), nFinish  , "/", nTotal ) )
    table.append( txtAlign.format("%sRetrieved Logs"        % (colors.PURPLE), nLogs    , "/", nTotal ) )
    table.append( txtAlign.format("%sRetrieved Outputs"     % (colors.PURPLE), nOut     , "/", nTotal ) )
    table.append( txtAlign.format("%sRetrieved Outputs (M)" % (colors.PURPLE), ""       , "" , nOutM ) )
    table.append( txtAlign.format("%sEOS Logs"              % (colors.CYAN  ), nLogsEOS , "/", nTotal ) )
    table.append( txtAlign.format("%sEOS Outputs"           % (colors.CYAN  ), nOutEOS  , "/", nTotal ) )
    table.append( txtAlign.format("%sEOS Outputs (M)"       % (colors.CYAN  ), ""       , "" , nOutEOSM ) )
    table.append( "{:<100}".format("%s%s"                   % (colors.WHITE, hLine) ) )
    return table

def Exists(dataset, filename):
    '''
    Checks that a dataset filename exists by executing the ls command for its full path.
    '''
    Verbose("Exists()", False)

#    fileName = os.path.join(dataset, "results", filename)
    fileName = os.path.join(dataset, "results")
    cmd      = "ls " + fileName

    Verbose(cmd)
    files     = Execute("%s" % (cmd) ) #not used
    file_re = re.compile(filename)
    for f in files:
        match = file_re.search(f)
        if match:
            return True
    #firstFile = files[0] #not used
    return os.path.exists(os.path.join(dataset, "results", filename))

def GetTaskStatusBool(datasetPath):
    '''
    Check the crab.log for the given task to determine the status.
    If the the string "Done" is found inside skip it.
    '''
    crabLog      = os.path.join(datasetPath,"crab.log")
    if not os.path.exists(crabLog):
        return False
    stringToGrep = "COMPLETED" #"Done"
    cmd          = "grep '%s' %s" % (stringToGrep, crabLog)

    ret = Execute(cmd)
    if len(ret) > 0:
        message = os.path.basename(datasetPath)
        while len(message) < 80:
            message += ' '
        message += stringToGrep
        print(message)
        return True

    return False

def GetTaskDashboardURL(datasetPath):
    '''
    Call the "grep" command to look for the dashboard URL from the crab.log file
    of a given dataset. It uses as input parameter the absolute path of the task dir (datasetPath)
    '''

    # Variable Declaration
    crabLog      = os.path.join(datasetPath, "crab.log")
    grepFile     = os.path.join(datasetPath, "grep.tmp")
    stringToGrep = "Dashboard monitoring URL"
    cmd          = "grep '%s' %s > %s" % (stringToGrep, crabLog, grepFile )
    dashboardURL = "UNKNOWN"

    # Execute the command
    if os.system(cmd) == 0:

        if os.path.exists( grepFile ):
            results      = [i for i in open(grepFile, 'r').readlines()]
            dashboardURL = FindBetween( results[0], "URL:\t", "\n" )
            os.system("rm -f %s " % (grepFile) )
        else:
            raise Exception("File %s not found!" % (grepFile) )
    else:
        raise Exception("Could not execute command %s" % (cmd) )
    return dashboardURL

def GetTaskStatus(datasetPath):
    '''
    Call the "grep" command to look for the "Task status" from the crab.log file
    of a given dataset. It uses as input parameter the absolute path of the task dir (datasetPath)
    '''

    # Variable Declaration
    crabLog      = os.path.join(datasetPath, "crab.log")
    grepFile     = os.path.join(datasetPath, "grep.tmp")
    #stringToGrep = "Task status:"
    #stringToGrep = "Status on the CRAB server:"
    #stringToGrep = "Jobs status:"
    stringToGrep  = "Status on the scheduler:"
    cmd          = "grep '%s' %s > %s" % (stringToGrep, crabLog, grepFile )
    status       = "UNKNOWN"

    if not os.path.exists( crabLog ):
        raise Exception("File %s not found!" % (crabLog) )

    # Execute the command
    if os.system(cmd) == 0:

        if os.path.exists( grepFile ):
            results = [i for i in open(grepFile, 'r').readlines()]
            status  = FindBetween( results[-1], stringToGrep, "\n" )
            os.system("rm -f %s " % (grepFile) )
        else:
            raise Exception("File %s not found!" % (grepFile) )
    else:
        raise Exception("Could not execute command %s" % (cmd) )
    return status

def GetTaskReports(datasetPath, opts):
    '''
    Execute "crab status", get task logs and output.
    Resubmit or kill task according to user options.
    '''
    report = None

    # Get all files under <dataset_dir>/results/
    files = Execute("ls %s" % os.path.join( datasetPath, "results") )
    #print "check datasetPath",datasetPath
    try:
        d = GetBasename(datasetPath)

        # Execute "crab status --dir=datasetPath"
        result = crabCommand('status', dir=datasetPath)

        # Get CRAB task status
        status = GetTaskStatus(d).replace("\t", "")

        # Get CRAB task dashboard URL
        dashboardURL = GetTaskDashboardURL(d)

        # Assess JOB success/failure for task
        idle, running, finished, transferring, failed, retrievedLog, retrievedOut, retrievedOutMerged, eosLog, eosOut, eosOutMerged = RetrievedFiles(datasetPath, result, dashboardURL, True, opts)

        # Get the task logs & output ?
        GetTaskLogs(datasetPath, retrievedLog, finished)

        # Get the task output
        GetTaskOutput(datasetPath, retrievedOut, finished)

        # Resubmit task if failed jobs found
        ResubmitTask(datasetPath, failed)

        # Kill task which are active
        KillTask(datasetPath)
        # Count retrieved/all jobs
        retrieved = min(finished, retrievedLog, retrievedOut)
        alljobs   = GetTotalJobsFromStatus(result)

        # Append the report
        report = Report(datasetPath, alljobs, idle, retrieved, running, finished, failed, transferring, retrievedLog, retrievedOut, eosLog, eosOut, status, dashboardURL)

        # Determine if task is DONE or not
        if retrieved == alljobs and retrieved > 0:
            absolutePath = os.path.join(datasetPath, "crab.log")
            os.system("sed -i -e '$a\DONE! (Written by multicrabCheck.py)' %s" % absolutePath )
    # Catch exceptions (Errors detected during execution which may not be "fatal")
    except:
        msg = sys.exc_info()[1]
        report = Report(datasetPath, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?")
        Print("crab status failed with message %s. Skipping ..." % ( msg ), True)
    return report

def GetTotalJobsFromStatus(status):
    '''
    reads output of "crab status" command and determines
    the total number of jobs for a given CRAB task
    '''
    Verbose("GetTotalJobsFromStatus()", True)

    # dictKey = 'jobList'  # obsolete after May 2017
    dictKey = 'jobs'
    nJobs = len(status[dictKey])
    return nJobs


def CheckTaskReport(taskDir, jobId,  opts):
    '''
    Probes the log-file tarball for a given jobId to
    determine the job status or exit code.
    '''
    Verbose("CheckTaskReport()", True)

    filePath    = os.path.join(taskDir, "results", "cmsRun_%i.log.tar.gz" % jobId)
    exitCode_re = re.compile("process\s+id\s+is\s+\d+\s+status\s+is\s+(?P<exitcode>\d+)")

    # Ensure file is indeed a tarfile
    if tarfile.is_tarfile(filePath):

        # Open the tarball
        fIN = tarfile.open(filePath)
        log_re = re.compile("cmsRun-stdout-(?P<job>\d+)\.log")

        # For-loop: All files inside tarball
        for member in fIN.getmembers():

            # Extract the log file
            logfile = fIN.extractfile(member)
            match   = log_re.search(logfile.name)

            # Regular Expression match for log-file
            if match:
                # For-loop: All lines of log-file
                for line in reversed(logfile.readlines()):

                    # Search for exit code
                    exitMatch = exitCode_re.search(line)

                    # If exit code found, return the value
                    if exitMatch:
                        return int(exitMatch.group("exitcode"))
    return -1

def GetTaskLogs(taskPath, retrievedLog, finished):
    '''
    If the number of retrieved logs files is smaller than the number of finished jobs,
    execute the CRAB command "getlog" to retrieve all unretrieved logs files.
    '''
    Verbose("GetTaskLogs()")

    if retrievedLog == finished:
        return

    if opts.get or opts.log:
        Verbose("Retrieved logs (%s) < finished (%s). Retrieving CRAB logs ..." % (retrievedLog, finished) )
        Touch(taskPath)
        if "fnal" in GetHostname():
            # "crab getoutput <task>" only works if the "-K ADLER32 option" is  removed (enabled by default)
            # the checksum validation seems to be causing trouble - CRAB can't autodetect whether to use it or not
            result = crabCommand('getlog', checksum="no", dir=taskPath)
        else:
            result = crabCommand('getlog', dir=taskPath)
    else:
        Verbose("Retrieved logs (%s) < finished (%s). To retrieve CRAB logs relaunch script with --get option." % (retrievedLog, finished) )
    return

def GetTaskOutput(taskPath, retrievedOut, finished):
    '''
    If the number of retrieved output files is smaller than the number of finished jobs,
    execute the CRAB command "getoutput" to retrieve all unretrieved output files.
    '''
    Verbose("GetTaskOutput()")

    if retrievedOut == finished:
        return

    if opts.get or opts.out:
        if opts.ask:
            if AskUser("Retrieved output (%s) < finished (%s). Retrieve CRAB output?" % (retrievedOut, finished) ):
                if "fnal" in GetHostname():
                    # "crab getoutput <task>" only works if the "-K ADLER32 option" is  removed (enabled by default)
                    # the checksum validation seems to be causing trouble - CRAB can't autodetect whether to use it or not
                    result = crabCommand('getoutput', checksum="no", dir=taskPath)
                else:
                    result = crabCommand('getoutput', dir=taskPath)
                Touch(taskPath)
            else:
                return
        else:
            Verbose("Retrieved output (%s) < finished (%s). Retrieving CRAB output ..." % (retrievedOut, finished) )

            if "fnal" in GetHostname():
                # "crab getoutput <task>" only works if the "-K ADLER32 option" is  removed (enabled by default)
                # the checksum validation seems to be causing trouble - CRAB can't autodetect whether to use it or not
                result = crabCommand('getoutput', checksum="no", dir=taskPath)
            else:
                finished_tmp = finished
                if len(retrievedOut) > 0:
                    finished_tmp = list(set(finished) - set(retrievedOut))
                while len(finished_tmp) > 0:
                    retrieveThese = finished_tmp[:500]
                    finished_tmp = finished_tmp[500:]
                    retrieveThese_str = ""
                    for a in retrieveThese:
                        retrieveThese_str+=a+','
                    retrieveThese_str = retrieveThese_str[:len(retrieveThese_str)-1]
                    result = crabCommand("getoutput", dir=taskPath, jobids=retrieveThese_str)

                #result = crabCommand("getoutput", dir=taskPath)
            Touch(taskPath)
    else:
        Verbose("Retrieved output (%s) < finished (%s). To retrieve CRAB output relaunch script with --get option." % (retrievedOut, finished) )

    return

def ResubmitTask(taskPath, failed):
    '''
    If the number of failed jobs is greater than zero,
    execute the CRAB command "resubmit" to resubmit all failed jobs.
    '''
    Verbose("ResubmitTask()")

    if failed == 0:
        return

    if not opts.resubmit:
        return

    joblist = JobList(failed)
    nFailed = 0
    if "," in joblist:
        nFailed = len(joblist.split(","))
    else:
        nFailed = 1

    # Sanity check
    if len(joblist) < 1:
        return

    taskName = os.path.basename(taskPath)
    Print("Found %d failed jobs! Resubmitting ..." % (nFailed) )
    Print("crab resubmit %s --jobids %s" % (taskName, joblist) )
#    result = crabCommand('resubmit', jobids=joblist, dir=taskPath)
    result = crabCommand('resubmit', dir=taskPath)
    Verbose("Calling crab \"resubmit %s --jobids %s\" returned %s" % (taskName, joblist, result ) )

    return

def KillTask(taskPath):
    '''
    If the number of failed jobs is greater than zero,
    execute the CRAB command "resubmit" to resubmit all failed jobs.
    '''
    Verbose("KillTask()")

    if not opts.kill:
        return

    taskStatus = GetTaskStatus(taskPath)
    taskStatus = taskStatus.replace("\t", "")
    forbidden  = ["KILLED", "UNKNOWN", "DONE", "COMPLETED", "QUEUED"]
    if taskStatus in forbidden:
        Print("Cannot kill a task if it is in the %s state. Skipping ..." % (taskStatus) )
        return
    else:
        Print("Killing jobs ...")

    if opts.ask:
        if AskUser("Kill task %s?" % (GetLast2Dirs(taskPath)) ):
            dummy = crabCommand('kill', dir=taskPath)
        else:
            pass
    else:
        dummy = crabCommand('kill', dir=taskPath)
    return

def Verbose(msg, printHeader=False):
    '''
    Calls Print() only if verbose options is set to true.
    '''
    if not opts.verbose:
        return
    Print(msg, printHeader)
    return

def Print(msg, printHeader=True):
    '''
    Simple print function. If verbose option is enabled prints, otherwise does nothing.
    '''
    fName = __file__.split("/")[-1]
    if printHeader:
        print "=== ", fName
    print "\t", msg
    return

def GetBasename(fullPath):
    return os.path.basename(fullPath)

def FindBetween(myString, first, last ):
    try:
        start = myString.index( first ) + len( first )
        end   = myString.index( last, start )
        return myString[start:end]
    except ValueError:
        return ""

if __name__ == "__main__":

    # Default Values
    VERBOSE = False
    PSET    = "PSet.py"
    SITE    = "T2_FI_HIP"
    DIRNAME = ""

    parser = OptionParser(usage="Usage: %prog [options]")
    parser.add_option("--create", dest="create", default=len(skimConfig)>0, action="store_true",
                      help="Flag to create a CRAB job [default: False")

    parser.add_option("--status", dest="status", default=False, action="store_true",
                      help="Flag to check the status of all CRAB jobs [default: False")

    parser.add_option("--get", dest="get", default=False, action="store_true",
                      help="Get output and log files of finished jobs [defaut: False]")

    parser.add_option("--log", dest="log", default=False, action="store_true",
                      help="Get log files of finished jobs [defaut: False]")

    parser.add_option("--out", dest="out", default=False, action="store_true",
                      help="Get output files of finished jobs [defaut: False]")

    parser.add_option("--resubmit", dest="resubmit", default=False, action="store_true",
                      help="Resubmit all failed jobs [defaut: False]")

    parser.add_option("--kill", dest="kill", default=False, action="store_true",
                      help="Kill all submitted jobs [defaut: False]")

    parser.add_option("-v", "--verbose", dest="verbose", default=VERBOSE, action="store_true",
                      help="Verbose mode for debugging purposes [default: %s]" % (VERBOSE))

    parser.add_option("-p", "--pset", dest="pset", default=PSET, type="string",
                      help="The python cfg file to be used by cmsRun [default: %s]" % (PSET))

    parser.add_option("-n", "--name", dest="name", default=analysisName, type="string",
                      help="The python cfg file to be used by cmsRun [default: \"\"]")

    parser.add_option("-d", "--dir", dest="dirName", default=DIRNAME, type="string",
                      help="Custom name for CRAB directory name [default: %s]" % (DIRNAME))

    parser.add_option("-s", "--site", dest="storageSite", default=SITE, type="string",
                      help="Site where the output will be copied to [default: %s]" % (SITE))

    parser.add_option("-i", "--includeTasks", dest="includeTasks", default="None", type="string",
                      help="Only perform action for this dataset(s) [default: \"\"]")

    parser.add_option("-e", "--excludeTasks", dest="excludeTasks", default="None", type="string",
                      help="Exclude this dataset(s) from action [default: \"\"]")

    parser.add_option("--filesInEOS", dest="filesInEOS", default=False, action="store_true",
                      help="The CRAB files are in a local EOS. Do not use files from the local multicrab directory [default: 'False']")

    parser.add_option("-u", "--url", dest="url", default=False, action="store_true",
                      help="Print the dashboard URL for the CARB task [default: False]")

    (opts, args) = parser.parse_args()

    if opts.create == False and opts.dirName == "":
        opts.dirName = os.getcwd()

    if opts.create == True and opts.status == True:
        raise Exception("Cannot both create and check a CRAB job!")

    if opts.create == True:
        sys.exit( CreateJob(opts, args) )
    elif opts.status == True or opts.get == True or opts.out == True or opts.log == True or opts.resubmit == True or opts.kill == True:
        if opts.dirName == "":
            raise Exception("Must provide a multiCRAB dir with the -d option!")
        else:
            sys.exit( CheckJob(opts, args) )
    else:
        raise Exception("Must either create or check a CRAB job!")


