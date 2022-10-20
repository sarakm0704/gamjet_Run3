#!/usr/bin/env python

import os
import sys
import re
import subprocess
import json

import uproot

import ROOT

from aux import execute

root_re = re.compile("(?P<rootfile>([^/]*events_\d+\.root))")
json_re = re.compile("(?P<jsonfile>files_\S+\.json$)")

class Dataset:
    def __init__(self,path,run):
        self.path = path
        self.name = os.path.basename(path)
        self.run = run
        self.year = getYear(path)
        self.type = getType(path)
        self.runRange = getRunRange(path)
        self.isData = False
        if "Run20" in self.name:
            self.isData = True
        self.lumi = 0
        self.files = []
        cands = execute("ls %s"%os.path.join(path,"results"))
        for c in cands:
            match = root_re.search(c)
            if match:
                self.files.append(os.path.join(path,"results",match.group("rootfile")))
            jsonmatch = json_re.search(c)
            if jsonmatch:
                f = open(os.path.join(path,"results",jsonmatch.group("jsonfile")))
                filelist = json.load(f)
                f.close()
                self.files = filelist["files"]

        if len(self.files) == 0:
            print("Dataset %s contains no root files"%path)
            return

        self.histograms = {}
        #self.fPU = self.files[0]
        self.fPU = os.path.join(path,"results","PileUp.root")
        self.fRF = ROOT.TFile.Open(self.files[0])

        self.skimCounter = self.fRF.Get("configInfo/SkimCounter")
        if self.skimCounter:
            self.skimCounter = self.skimCounter.Clone("SkimCounter")
            self.skimCounter.Reset()
            for fname in self.files:
                rf = ROOT.TFile.Open(fname)
                s = rf.Get("configInfo/SkimCounter").Clone("SkimCounter")
                self.skimCounter.Add(s)
                rf.Close()
            obj = self.skimCounter.Clone("skimCounter")
            obj.SetTitle("skim")
            self.histograms["skimCounter"] = obj

    def getPileup(self):
        if self.isData:
            return None
        fPU = self.getPileupfile()
        with uproot.open(fPU) as fIN:
            return fIN['pileup']
            #return fIN['configInfo/pileup']
        return None

    def getPileupfile(self):
        return self.fPU

    def getFileNames(self):
        return self.files

    def getSkimCounter(self):
        return self.skimCounter

    
    def Print(self):
        print( self.name )
        print( "    is data",self.isData )
        print( "    number of files",len(self.files) )

#def execute(cmd):
#    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
#                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
#    (s_in, s_out) = (p.stdin, p.stdout)
#
#    f = s_out
#    ret=[]
#    for line in f:
#        line = str(line,'utf-8')
#        ret.append(line.replace("\n", ""))
#    f.close()
#    return ret

def getRun(multicrabdir,dsetnames):
    datasetnames = dsetnames
    datasetnames.append(multicrabdir)
    run = ""
    apv = ""
    run_re = re.compile("Run(?P<run>20\d\d\S+?)_")
    for n in datasetnames:
        if 'HIPM' in n:
            apv = "APV"
        match = run_re.search(n)
        if match:
            run = match.group("run")
            break
    return run+apv

def getDatasets(multicrabdir,whitelist=[],blacklist=[]):
    datasets = []
    cands = execute("ls %s"%multicrabdir)
#    for c in cands:                                                                                                                                            
#        resultdir = os.path.join(multicrabdir,c,"results")                                                                                                     
#        if os.path.exists(resultdir):                                                                                                                          
#            datasets.append(Dataset(os.path.join(multicrabdir,c)))                                                                                             

    if len(whitelist) > 0:
        #print "check whitelist 1 ",whitelist,blacklist                                                                                                         
        datasets_whitelist = []
        for d in cands:
            for wl in whitelist:
                wl_re = re.compile(wl)
                match = wl_re.search(d)
                if match:
                    datasets_whitelist.append(d)
                    break
        #print "check whitelist",datasets_whitelist                                                                                                             
        cands = datasets_whitelist

    if len(blacklist) > 0:
        #print "check blacklist 1 ",whitelist,blacklist                                                                                                         
        datasets_blacklist = []
        for d in cands:
            found = False
            for bl in blacklist:
                bl_re = re.compile(bl)
                match = bl_re.search(d)
                if match:
                    found = True
                    break
            if not found:
                datasets_blacklist.append(d)
        cands = datasets_blacklist

    run = getRun(multicrabdir,cands)

    for c in cands:
        resultdir = os.path.join(multicrabdir,c,"results")
        if os.path.exists(resultdir):
            datasets.append(Dataset(os.path.join(multicrabdir,c),run))

    return datasets

def getDataPileupROOT(datasets):
    pileup_data = ROOT.TH1F("pileup_data","",100,0,100)
    for d in datasets:
        if d.isData:
            if hasattr(d, 'pileup'):
                pileup_data.Add(d.pileup)
    return pileup_data

def getDataPileup(datasets):
    hMC = None
    for d in datasets:
        if d.isData:
            fPU = d.getPileupfile()
            with uproot.open(fPU) as fIN:
                if hMC == None:
                    hMC = fIN['pileup'].values()
                else:
                    hMC.sum(fIN['pileup'].values())
                print("check pileup",hMC)
    return hMC

def getDataPileupMulticrab(multicrabdir):
    with uproot.open(os.path.join(multicrabdir,"pileup.root")) as fIN:
        return fIN['pileup']
    return None
    """
    print("check getDataPileupMulticrab",os.path.join(multicrabdir,"pileup.root"))
    fIN = ROOT.TFile.Open(os.path.join(multicrabdir,"pileup.root"))
    pileup_data = fIN.Get("pileup").Clone("pileup_data")
    pileup_data.SetDirectory(0)
    print("check pu data histo",pileup_data)
    return pileup_data
    """
def loadLuminosity(multicrabdir,datasets):
    lumisum = 0
    lumijson = open(os.path.join(multicrabdir,"lumi.json"),'r')
    data = json.load(lumijson)
    for d in datasets:
        if d.name in data.keys():
            d.lumi = data[d.name]
            lumisum += d.lumi
    lumijson.close()
    return lumisum

def getYear(multicrabdir):
    year_re = re.compile("Run(?P<year>20\d\d)\S+_")
    match = year_re.search(multicrabdir)
    if match:
        return match.group("year")
    return "-1"

def getType(multicrabdir):
    # DoubleMuon_Run2017B_UL2017_MiniAODv2_NanoAODv9_v1_297050_299329
    type_re = re.compile("\S+Run20\d\d\S+?_(?P<type>\S*?20\S+?)_")
    match = type_re.search(multicrabdir)
    if match:
        return match.group("type")
    return "-1"

def getRunRange(multicrabdir):
    # DoubleMuon_Run2017B_UL2017_MiniAODv2_NanoAODv9_v1_297050_299329
    rr_re = re.compile("_(?P<rr>\d\d\d\d\d\d_\d\d\d\d\d\d)")
    match = rr_re.search(multicrabdir)
    if match:
        return match.group("rr")
    return "-1"
