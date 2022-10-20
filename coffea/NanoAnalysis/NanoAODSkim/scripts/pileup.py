#!/usr/bin/env python

import os
import sys
import re
import subprocess
import ROOT

#from multicrab import Dataset,FindDataset,GetDatasetsPaths,GetRequestName
#from optparse import OptionParser

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData
calcMode       = "true"
maxPileupBin   = "200" 
numPileupBins  = "200"
pileupHistName = "pileup"
PileUpJSON_2016 = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/pileup_latest.txt"
PileUpJSON_2017 = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt"
PileUpJSON_2018 = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/pileup_latest.txt"
#PileUpJSON = PileUpJSON_2016
#PileUpJSON = PileUpJSON_2017
#PileUpJSON = PileUpJSON_2018
#
# Recommended minimum bias xsection                                                                                                                                         
minBiasXsecNominal = 69200 #from https://twiki.cern.ch/twiki/bin/viewauth/CMS/POGRecipesICHEP2016
minBiasXsec = minBiasXsecNominal
puUncert    = 0.05 

pupath = os.path.join(os.environ['CMSSW_BASE'],"src/NanoAnalysis/NanoAODSkim/python")

def usage():
    print
    print "### Usage:  ",sys.argv[0]," <multicrabdir>"
    print

def Execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    ret    = []
    for line in p.stdout:
        line = line.decode('utf-8')
        ret.append(line.replace("\n", ""))
    return ret

def CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec,fOUT,pileupHistName):
    cmd = [os.path.join(pupath,"pileupCalc.py"), "-i", inputFile, "--inputLumiJSON", inputLumiJSON, "--calcMode", calcMode,
           "--minBiasXsec", minBiasXsec, "--maxPileupBin", maxPileupBin, "--numPileupBins", numPileupBins,
           "--pileupHistName", pileupHistName, fOUT]
    sys_cmd = " ".join([str(c) for c in cmd])
    print sys_cmd
    os.system(sys_cmd)

def sumPU(multicrabdir):
    pu_files = Execute("find %s -name 'PileUp.root'"%multicrabdir)
    #print(pu_files)

    first = True
    for f in pu_files:
        if first:
            first = False
            fIN = ROOT.TFile.Open(f)
            fIN.ls()
            pu      = fIN.Get("pileup")
            pu.SetDirectory(0)
            pu_up   = fIN.Get("pileup_up")
            pu_up.SetDirectory(0)
            pu_down = fIN.Get("pileup_down")
            pu_down.SetDirectory(0)
            fIN.Close()
        else:
            fIN = ROOT.TFile.Open(f)
            pu1      = fIN.Get("pileup")
            pu.Add(pu1)
            pu_up1   = fIN.Get("pileup_up")
            pu_up.Add(pu_up1)
            pu_down1 = fIN.Get("pileup_down")
            pu.Add(pu_down1)
            fIN.Close()
    fOUT = ROOT.TFile(os.path.join(multicrabdir,"pileup.root"),"recreate")
    pu.Write()
    pu_up.Write()
    pu_down.Write()
    fOUT.Close()

def getYear(multicrabdir):
    year_re = re.compile("_Run(?P<year>201\d)\S+_")
    match = year_re.search(multicrabdir)
    if match:
        return match.group("year")
    return "YearNotFound"

def isData(dsetname):
    if "Run201" in dsetname:
        return True
    return False

def main():

    if len(sys.argv) == 1:
        usage()
        sys.exit()

    multicrabdir = os.path.realpath(sys.argv[1])
    parser = OptionParser(usage="Usage: %prog [options]")
    parser.add_option("-d", "--dir", dest="dirName", default=multicrabdir, type="string",
                      help="Custom name for CRAB directory name [default: %s]" % (multicrabdir))

    parser.add_option("-i", "--includeTasks", dest="includeTasks", default="None", type="string",
                      help="Only perform action for this dataset(s) [default: \"\"]")

    parser.add_option("-e", "--excludeTasks", dest="excludeTasks", default="None", type="string",
                      help="Exclude this dataset(s) from action [default: \"\"]")

    (opts, args) = parser.parse_args()
    datasetpaths = GetDatasetsPaths(opts)
    for datasetpath in datasetpaths:
        dataset = FindDataset(datasetpath)
        if not dataset.isData():
            continue

        dsetname = GetRequestName(dataset)
        path = os.path.join(multicrabdir,dsetname)
        if os.path.exists(path):
            fOUT = os.path.join(multicrabdir,dsetname,"results","PileUp.root")
            if dataset.getYear() == "2016":
                PileUpJSON = PileUpJSON_2016
            if dataset.getYear() == "2017":
                PileUpJSON = PileUpJSON_2017
            if dataset.getYear() == "2018":
                PileUpJSON = PileUpJSON_2018

            inputLumiJSON = PileUpJSON
            inputFile = dataset.lumiMask # crab report not working for nanoaod yet, assuming 100% jobs successfull
            hName = pileupHistName
            minBiasXsec = minBiasXsecNominal
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec,fOUT,hName)

            minBiasXsec_up = minBiasXsec*(1+puUncert)
            fOUT_up        = fOUT.replace(".root","_up.root")
            hName_up       = pileupHistName+"_up"
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec_up,fOUT_up,hName_up)

            minBiasXsec_down = minBiasXsec*(1-puUncert)
            fOUT_down        = fOUT.replace(".root","_down.root")
            hName_down       = pileupHistName+"_down"
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec_down,fOUT_down,hName_down)

            hadd_cmd = "hadd -a %s %s %s"%(fOUT,fOUT_up,fOUT_down)
            os.system(hadd_cmd)

    sumPU(multicrabdir)

def main2():

    if len(sys.argv) == 1:
        usage()
        sys.exit()

    multicrabdir = os.path.realpath(sys.argv[1])
    year = getYear(multicrabdir)

    dirs = Execute("ls %s"%multicrabdir)
    for dsetname in dirs:
        if os.path.isdir(os.path.join(multicrabdir,dsetname)):
            if not isData(dsetname):
                continue

            cmd = "ls %s"%(os.path.join(multicrabdir,dsetname,'inputs','Cert_*.txt'))
            inputFile = Execute(cmd)

            fOUT = os.path.join(multicrabdir,dsetname,"results","PileUp.root")
            if year == "2016":
                PileUpJSON = PileUpJSON_2016
            if year == "2017":
                PileUpJSON = PileUpJSON_2017
            if year == "2018":
                PileUpJSON = PileUpJSON_2018

            inputLumiJSON = PileUpJSON
            inputFile = inputFile[0] # crab report not working for nanoaod yet, assuming 100% jobs successfull                                                                                                                               
            hName = pileupHistName
            minBiasXsec = minBiasXsecNominal
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec,fOUT,hName)

            minBiasXsec_up = minBiasXsec*(1+puUncert)
            fOUT_up        = fOUT.replace(".root","_up.root")
            hName_up       = pileupHistName+"_up"
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec_up,fOUT_up,hName_up)

            minBiasXsec_down = minBiasXsec*(1-puUncert)
            fOUT_down        = fOUT.replace(".root","_down.root")
            hName_down       = pileupHistName+"_down"
            CallPileupCalc(inputFile,inputLumiJSON,minBiasXsec_down,fOUT_down,hName_down)

            hadd_cmd = "hadd -a %s %s %s"%(fOUT,fOUT_up,fOUT_down)
            os.system(hadd_cmd)

    sumPU(multicrabdir)
    
if __name__ == "__main__":
    main2()
