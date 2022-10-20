#!/usr/bin/env python

import os
import sys
import re
import json
import subprocess

#from multicrab import Dataset,alldatasets,GetRequestName

NormTagJSON     = "/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json"

def usage():
    print
    print "### Usage:  ",sys.argv[0]," <multicrabdir>"
    print

def CallBrilcalc(InputFile,brilcalc_out):
    # brilcalc lumi -u /pb -i JSON-file
    home = os.environ['HOME']
    path = os.path.join(home, ".local/bin")
    exe  = os.path.join(path, "brilcalc")
    
    BeamStatus = '"STABLE BEAMS"'
    CorrectionTag=NormTagJSON
    LumiUnit="/pb"

    # Ensure brilcal executable exists
    if not os.path.exists(exe):
        print "brilcalc not found, have you installed it?"
        print "http://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html"
        sys.exit()
    
    # Execute the command
    cmd     = [exe,"lumi", "-b", BeamStatus, "--normtag", CorrectionTag, "-u", LumiUnit, "-i", InputFile]
    sys_cmd = " ".join(cmd) + " > %s" %brilcalc_out

    ret    = os.system(sys_cmd)

    output = [i for i in open(brilcalc_out, 'r').readlines()]
    lumi = GetLumiAndUnits(output)
    return lumi

def GetLumiAndUnits(output):
    '''
    Reads output of "brilcalc" command
    and finds and returns the lumi and units
    '''
        
    # Definitions
    lumi = -1.0
       
    # Regular expressions
    lumi_re = re.compile("\|\s+(?P<recorded>\d+\.*\d*)\s+\|\s*$")
    
    #For-loop: All lines in "crab report <task>" output
    for line in output:
        m = lumi_re.search(line)
        if m:
            lumi = float(m.group("recorded")) # lumiCalc2.py returns pb^-1
          
    return lumi

def Execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    ret    = []
    for line in p.stdout:
        line = line.decode('utf-8')
        ret.append(line.replace("\n", ""))
    return ret

def isData(dsetname):
    if "Run201" in dsetname:
        return True
    return False

def main():

    if len(sys.argv) == 1:
        usage()
        sys.exit()

    lumidata = {}
    multicrabdir = sys.argv[1]
    sumlumi = 0
    for dataset in alldatasets:
        if not dataset.isData():
            continue

        dsetname = GetRequestName(dataset)
        path = os.path.join(multicrabdir,dsetname)
        #print path
        if os.path.exists(path):
            print dsetname
            """
            fOUT = os.path.join(multicrabdir,dsetname,"results","PileUp.root")
            inputLumiJSON = PileUpJSON
            inputFile = dataset.lumiMask # crab report not working for nanoaod yet, assuming 100% jobs successfull
            hName = pileupHistName
            minBiasXsec = minBiasXsecNominal
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec,fOUT,hName)
            """
            inputFile = dataset.lumiMask # crab report not working for nanoaod yet, assuming 100% jobs successfull
            fOUT = os.path.join(multicrabdir,dsetname,"results","brilcalc.log")
            lumidata[dsetname] = CallBrilcalc(inputFile,fOUT)
            sumlumi += lumidata[dsetname]
            lumijson = os.path.join(multicrabdir,"lumi.json")
            f = open(lumijson, "wb")
            json.dump(lumidata, f, sort_keys=True, indent=2)
            f.close()
    print "Sum lumi",sumlumi

def main2():
    
    if len(sys.argv) == 1:
        usage()
        sys.exit()

    lumidata = {}
    multicrabdir = sys.argv[1]
    sumlumi = 0

    dirs = Execute("ls %s"%multicrabdir)
    for d in dirs:
        if not isData(d):
            continue

        if os.path.isdir(os.path.join(multicrabdir,d)):
            cmd = "ls %s"%(os.path.join(multicrabdir,d,'inputs','Cert_*.txt'))
            inputFile = Execute(cmd)[0]
            fOUT = os.path.join(multicrabdir,d,"results","brilcalc.log")
            lumidata[d] = CallBrilcalc(inputFile,fOUT)
            sumlumi += lumidata[d]
            lumijson = os.path.join(multicrabdir,"lumi.json")
            f = open(lumijson, "wb")
            json.dump(lumidata, f, sort_keys=True, indent=2)
            f.close()
    print "Sum lumi",sumlumi

if __name__ == "__main__":
    main2()
