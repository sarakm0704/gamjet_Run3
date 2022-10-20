#!/usr/bin/env python

import sys
import os,re
import datetime
import numpy
import multiprocessing

import awkward as ak
from coffea import nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea import lookup_tools
from coffea import analysis_tools

from typing import Iterable, Callable, Optional, List, Generator, Dict, Union
import collections
import hist
import uproot

import ROOT


basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python"))


import multicrabdatasets
import hist2root
import Counter
import PileupWeight
from parsePileUp import *
import RochesterCorrections
import JetCorrections
import LumiMask
import JetvetoMap
import Btag
import QGL
import aux

import ZbSelection as selection
import ZbHistograms as Histograms

#LEPTONFLAVOR = 11
LEPTONFLAVOR = 13

MAX_WORKERS = 1 #max(multiprocessing.cpu_count()-1,1)

CHUNKSIZE = 100000
MAXCHUNKS = None

MAXEVENTS = -1
#MAXEVENTS = 1000
if MAXEVENTS > 0:
    CHUNKSIZE = MAXEVENTS
    MAXCHUNKS = 1


class Analysis(processor.ProcessorABC):
    def __init__(self,dataset,pu_data): ####run,isData, pu_data, pu_mc):
        self.run = dataset.run
        self.year = dataset.run[:4]
        print("Analyzing year",self.year)
        self.isData = dataset.isData

        if self.isData:
            parsePileUpJSON2(self.year)

        self.pu_data = pu_data
        if not self.isData:
            self.pu_mc   = dataset.getPileup()
            self.pu_weight = PileupWeight.PileupWeight(self.pu_data,self.pu_mc)

        self.lumimask = LumiMask.LumiMask(dataset)

        self.histo = Histograms.AnalysisHistograms(self.isData)

        self.counter = Counter.Counters()
        if "skimCounter" in dataset.histograms.keys():
            self.counter.book(dataset.histograms["skimCounter"])

        self.book_histograms()
        x,y = self.getArrays(self.pu_data)
        self.histograms['pu_data'].fill('pu_data', value=x, weight=y)

        self.rochesterCorrections = RochesterCorrections.Roccor(self.run,self.isData)
        self.jetCorrections = JetCorrections.JEC(self.run,self.isData)
        self.jetvetoMap = JetvetoMap.JetvetoMap(self.year,self.isData)

        self.btag = Btag.Btag('btagDeepB',self.year)
        self.ctag = Btag.Btag('btagDeepC',self.year)
        #self.qgl = QGL.qgl(self.year)

    def book_histograms(self):
        self.histograms = {}
        self.histograms.update(self.histo.book())

        self.addHistogram('pu_orig', 100, 0, 100)
        if not self.isData:
            self.cloneHistogram('pu_orig', 'pu_corr')
        self.cloneHistogram('pu_orig', 'pu_data')
        print("Booked",len(self.histograms),"histograms")

    def addHistogram(self, name, nbins, binmin, binmax):
        self.histograms[name] = hist.Hist(
            hist.axis.StrCategory([], growth=True, name=name, label="label"),
            hist.axis.Regular(nbins, binmin, binmax, name="value", label="x value"),
            storage="weight",
            name="Counts"
        )
        
    def cloneHistogram(self, nameOrig, nameClone):
        edges = self.histograms[nameOrig].axes['value'].edges
        nbins = len(edges)
        binmin = edges[0]
        binmax = edges[-1]
        self.addHistogram(nameClone,nbins,binmin,binmax)

    def getArrays(self, histo):
        x = []
        axis  = histo.axis()
        edges = axis.edges()
        for i in range(0,len(edges)-1):
            bincenter = edges[i] + 0.5*(edges[i+1]-edges[i])
            x.append(bincenter)
        y = histo.values()
        return ak.from_numpy(numpy.array(x)),y

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        out = {}

        self.counter.setSkimCounter()

        # Weights
        eweight = analysis_tools.Weights(len(events),storeIndividual=True)
        if not self.isData:
            genw = events['genWeight']
            eweight.add('pileup',genw)

            pu = self.pu_weight.getWeight(events.Pileup.nTrueInt)
            eweight.add('pileup',pu)
        events["weight"] = eweight.weight()
        self.counter.increment('all events',events)

        if self.isData:
            events = events[selection.lumimask(events,self.lumimask)]
        self.counter.increment('JSON filter',events)

        events = events[selection.triggerSelection(events, self.year, LEPTONFLAVOR)]
        self.counter.increment('passed trigger',events)

        events = events[selection.METCleaning(events, self.year)]
        self.counter.increment('MET cleaning',events)

        events = selection.leptonSelection(events, LEPTONFLAVOR)
        self.counter.increment('lepton selection',events)

        events['Muon'] = self.rochesterCorrections.apply(events)                                                                                              
        events = selection.leptonPtCut(events, LEPTONFLAVOR)
        self.counter.increment('lepton pt cut',events)

        events = selection.Zboson(events,LEPTONFLAVOR)
        self.counter.increment('Z boson',events)

        events = selection.JetSelection(events)
        self.counter.increment('jet selection',events)

        events = selection.JetCorrections(events,self.jetCorrections)

        events["leadingJet"] = events.Jet[:, 0]
        events = events[(events.leadingJet.pt > 12)]
        self.counter.increment('leading jet pt',events)

        events = selection.Jetvetomap(events,self.jetvetoMap)
        self.counter.increment('jetvetomap',events)

        events = selection.PhiBB(events)
        self.counter.increment('phiBB',events)

        events = events[selection.Jetid(events.leadingJet)]
        self.counter.increment('jet jetId',events)

        # Constructing variables
        mu = None
        if self.isData:
            mu = getAvgPU(events.run,events.luminosityBlock)
        else:
            mu = events.Pileup.nTrueInt
        events["mu"] = mu

        events["METtype1"] = self.jetCorrections.recalculateMET2(events)

        events["leadingJet"]["qgl"] = np.zeros_like(events["leadingJet"]) ####self.qgl.compute(events["leadingJet"])
        lj = events["leadingJet"]
        lj["qgl"] = np.zeros_like(events["leadingJet"].pt) ####self.qgl.compute(events["leadingJet"])
        events["leadingJet"] = lj

        events["btag"] = (events["leadingJet"].btagDeepB > self.btag.tight())
        events["ctag"] = ((events["leadingJet"].btagDeepB <= self.btag.tight()) & (events["leadingJet"].btagDeepCvL > self.ctag.tight()))
        events["quarktag"] = (
            (events["leadingJet"].btagDeepB <= self.btag.tight()) &
            (events["leadingJet"].btagDeepCvL <= self.ctag.tight()) &
            (events["leadingJet"].qgl > 0.5)
        )
        events["gluontag"] = (
            (events["leadingJet"].btagDeepB <= self.btag.tight()) &
            (events["leadingJet"].btagDeepCvL <= self.ctag.tight()) &
            (events["leadingJet"].qgl >= 0) &
            (events["leadingJet"].qgl < 0.5)
        )
        events["notag"] = (
            (events["btag"] == False) &
            (events["ctag"] == False) &
            (events["quarktag"] == False) &
            (events["gluontag"] == False)
        )

        # leading jet gen flavor
        if not self.isData:
            ljetGenFlavor = np.absolute(events["leadingJet"].partonFlavour)
            events["ljetGenFlavor"] = ljetGenFlavor

        # Plotting
        out = self.histo.fill(events,out)
        self.histograms.update(self.histo.fill(events,out))

        # Plot PU distributions to see that the pu reweighting works
        if not self.isData:
            self.histograms['pu_orig'].fill('pu_orig', value=mu)
            self.histograms['pu_corr'].fill('pu_corr', value=mu, weight=events["weight"])

        out["counter"] = self.counter
        out.update(self.histograms)
        return out

    def postprocess(self, accumulator):
        pass

def usage():
    print
    print( "### Usage:  ",os.path.basename(sys.argv[0]),"<multicrab skim>" )
    print

def main():

    multicrabdir = os.path.abspath(sys.argv[1])
    if not os.path.exists(multicrabdir) or not os.path.isdir(multicrabdir):
        usage()
        sys.exit()
    year = multicrabdatasets.getYear(multicrabdir)

    starttime = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    lepton = "_"

    blacklist = []
    whitelist = []

    if LEPTONFLAVOR == 11:
        lepton+="Electron"
        whitelist = ["DoubleEG","EGamma","DYJets","TTJet"]

    if LEPTONFLAVOR == 13:
        lepton+="Muon"
#        whitelist = ["Muon","DYJets","TTJet"]
#        whitelist = ["Muon"]
#        whitelist = ["DYJets"]

    datasets = multicrabdatasets.getDatasets(multicrabdir,whitelist=whitelist,blacklist=blacklist)
    pileup_data = multicrabdatasets.getDataPileupMulticrab(multicrabdir)
    lumi = multicrabdatasets.loadLuminosity(multicrabdir,datasets)

    outputdir = os.path.basename(os.path.abspath(multicrabdir))+"_processed"+starttime+lepton
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    import time
    t0 = time.time()

    print("Number of cores used",MAX_WORKERS)
    if len(datasets) == 0:
        print("No datasets to be processed")
        print("  whitelist:",whitelist)
        print("  blacklist:",blacklist)
        sys.exit()

    for i,d in enumerate(datasets):
        print("Dataset %s/%s %s"%(i+1,len(datasets),d.name))
        t00 = time.time()
        subdir = os.path.join(outputdir,d.name)
        if not os.path.exists(subdir):
            os.mkdir(subdir)
            os.mkdir(os.path.join(subdir,"results"))

        samples = {d.name: d.getFileNames()}

        job_executor = processor.FuturesExecutor(workers = MAX_WORKERS)
        #job_executor = processor.IterativeExecutor()
        run = processor.Runner(
            executor = job_executor,
            schema=nanoevents.NanoAODSchema,
            chunksize = CHUNKSIZE,
            maxchunks = MAXCHUNKS
        )
        result = run(samples, 'Events', Analysis(d,pileup_data))
        result["counter"].print()
        """
        result = processor.run_uproot_job(
            samples,
            "Events",
            Analysis(d,pileup_data), ####d.run,d.isData,pileup_data,d.getPileup()),
            processor.iterative_executor,
            {"schema": NanoAODSchema},
        )
        """
        t01 = time.time()
        dt0 = t01-t00
        print("Processing time %s min %s s"%(int(dt0/60),int(dt0%60)))
        print("Writing histograms.. ",end='\r')

        with uproot.recreate(os.path.join(subdir,"results","histograms.root")) as fOUT:
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            now = datetime.datetime.now()
            m = "produced: %s %s"%(days[now.weekday()],now)
            fOUT[f"configInfo/{m}"] = ""

            gitCommit = aux.execute("git rev-parse HEAD")[0]
            gc = "git commit:"+gitCommit
            fOUT[f"configInfo/{gc}"] = gitCommit

            h_lumi = hist.Hist(
                hist.axis.StrCategory([], growth=True, name='lumi',label="label"),
                hist.axis.Regular(1,0,1, name="value", label="x value"),
            )
            h_lumi.fill('lumi',value=0.5,weight=d.lumi)
            result['lumi'] = h_lumi
            
            h_isdata = hist.Hist(
                hist.axis.StrCategory([], growth=True, name='isdata',label="label"),
                hist.axis.Regular(1,0,1, name="value", label="x value"),
            )
            h_isdata.fill('isdata',value=0.5,weight=int(d.isData))
            result['isdata'] = h_isdata
        
            for key in result.keys():
                rootdir = 'analysis'
                if 'counter' in key:
                    #fOUT.cd("configInfo")
                    h_unwCounter,h_wCounter = result[key].histo()
                    #h_unwCounter.Write()
                    #h_wCounter.Write()
                    fOUT['unweighted_counter'] = h_unwCounter
                    #fOUT['weighted_counter'] = h_wCounter
                    #fOUT.cd()
                    continue                                                                                                                                                 
                if key in ['pu_orig','pu_data','pu_corr','lumi','isdata']:
                    rootdir = 'configInfo'
                    
                for s in result[key].axes[key]:
                    fOUT[f"{rootdir}/{s}"] = result[key][{key: s}]

            
        """
        fOUT = ROOT.TFile.Open(os.path.join(subdir,"results","histograms.root"),"RECREATE")
        fOUT.cd()
        fOUT.mkdir("configInfo")
        fOUT.cd("configInfo")

        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        now = datetime.datetime.now()
        m = "produced: %s %s"%(days[now.weekday()],now)
        timestamp = ROOT.TNamed(m,"");
        timestamp.Write()

        gitCommit = aux.execute("git rev-parse HEAD")[0]
        gc = "git commit:"+gitCommit
        gitcommit = ROOT.TNamed(gc,gitCommit);
        gitcommit.Write()

        h_lumi = ROOT.TH1D("lumi","",1,0,1)
        h_lumi.SetBinContent(1,d.lumi)
        h_lumi.Write()

        h_isdata = ROOT.TH1D("isdata","",1,0,1)
        h_isdata.SetBinContent(1,int(d.isData))
        h_isdata.Write()

        fOUT.cd()
        fOUT.mkdir("analysis")

        for key in result.keys():
            #print("check keys",key)
            if 'counter' in key:
                fOUT.cd("configInfo")
                h_unwCounter,h_wCounter = result[key].histo()
                h_unwCounter.Write()
                h_wCounter.Write()
                fOUT.cd()
                continue
            if key in ['pu_orig','pu_data','pu_corr']:
                fOUT.cd("configInfo")
            else:
                fOUT.cd("analysis")
            histo = hist2root.convert(result[key]).Clone(key)
            histo.Write()
            fOUT.cd()
        fOUT.Close()
        """
        dt1 = time.time()-t01
        print("Histogram writing time %s min %s s             "%(int(dt1/60),int(dt1%60)))

    dt = time.time()-t0

    print("Total processing time %s min %s s"%(int(dt/60),int(dt%60)))
    print("output in",outputdir)

if __name__ == "__main__":
    main()
    #os.system("ls -lt")
    #os.system("pwd")
