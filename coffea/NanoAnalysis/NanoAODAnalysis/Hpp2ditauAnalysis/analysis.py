#!/usr/bin/env python
import sys
import os,re
import datetime
import numpy
import multiprocessing

import awkward as ak
from coffea import nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor, hist
from coffea import lookup_tools
from coffea import analysis_tools

from typing import Iterable, Callable, Optional, List, Generator, Dict, Union
import collections

import ROOT


basepath_re = re.compile("(?P<basepath>\S+/nanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python"))

sys.path.append('../Framework/python')

import multicrabdatasets
import hist2root
import PileupWeight
from parsePileUp import *

import Hpp2ditauSelection as selection

MAX_WORKERS = 1 #max(multiprocessing.cpu_count()-1,1)

class Analysis(processor.ProcessorABC):
    def __init__(self,run,isData, pu_data, pu_mc):
        self.run = run
        self.year = run[:4]
        self.isData = isData

        if not isData:
            self.pu_data = pu_data
            self.pu_mc   = pu_mc
            self.pu_weight = PileupWeight.PileupWeight(pu_data,pu_mc)

        self.book_histograms()
        self.first = True

    def book_histograms(self):
        self.histograms = {}
        self.histograms['counter'] = processor.defaultdict_accumulator(int)

        self.addHistogram('pT_tau', 300,0,300)
        #self.cloneHistogram('pT_tau','pT_muon')
        self.addHistogram('eta_tau',50,-4,4)
        #self.addHistogram('N_tau',100,1,100)
        if not self.isData:
            self.addHistogram('pu_orig', 100, 1, 100)
            self.cloneHistogram('pu_orig', 'pu_corr')
            self.cloneHistogram('pu_orig', 'pu_data')

        self._accumulator = processor.dict_accumulator(self.histograms)

    def addHistogram(self, name, nbins, binmin, binmax):
        self.histograms[name] = hist.Hist(
            "Events",
            hist.Bin("value", name, nbins, binmin, binmax)
        )

    def cloneHistogram(self, nameOrig, nameClone):
        self.histograms[nameClone] = self.histograms[nameOrig].copy()

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
        out = self.accumulator.identity()

        out['counter']['all events'] += len(events.event)

        events = events[selection.triggerSelection(events, self.year)]
        out['counter']['passed trigger'] += len(events.event)

        events = events[selection.METCleaning(events, self.year)]
        out['counter']['MET cleaning'] += len(events.event)

        dataset = events.metadata['dataset']
        muons = ak.zip({
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "charge": events.Muon.charge,
        }, with_name="PtEtaPhiMCandidate")

        electrons = ak.zip({
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "charge": events.Electron.charge,
        }, with_name="PtEtaPhiMCandidate")

        taus = ak.zip({
            "pt": events.Tau.pt,
            "eta": events.Tau.eta,
            "phi": events.Tau.phi,
            "mass": events.Tau.mass,
            "charge": events.Tau.charge,
        }, with_name="PtEtaPhiMCandidate")

        # numtaus = 3 # benchmark scenarios, 2003.08443                                              
        numtaus = 2 # CMS-PAS-HIG-16-036                                                
        taupt = 30
        taueta = 2.3
        #muoncut = (ak.num(events.Muon) == 4) & (ak.sum(events.Muon.pt,axis=1) > 30)
        muoncut = (ak.num(events.Muon) == 0)
        electroncut = (ak.num(events.Electron) == 0)
        
        taucut = (ak.num(events.Tau) >= (numtaus)) & (ak.sum(events.Tau.pt, axis=1) > taupt) & (ak.sum(events.Tau.eta, axis=1) < taueta)         
        cut = (muoncut & electroncut & taucut)
        events = events[cut]
        out['counter']['cuts'] += len(events.event)

        eweight = analysis_tools.Weights(len(events),storeIndividual=True)

        if not self.isData:
            out['pu_orig'].fill(value=events.Pileup.nTrueInt)
            pu = self.pu_weight.getWeight(events.Pileup.nTrueInt)
            eweight.add('pileup',pu)
            out['pu_corr'].fill(value=events.Pileup.nTrueInt,weight=eweight.weight())
            out['pT_tau'].fill(value=events.Tau.pt[:,0]) #,weight=eweight.weight())
            out['eta_tau'].fill(value=events.Tau.eta[:,0]) #,weight=eweight.weight())
            #out['N_tau'].fill(value=events.Tau[:,0])
            if self.first:
                x,y = self.getArrays(self.pu_data)
                out['pu_data'].fill(value=x,weight=y)

        self.first = False


        return out

    def postprocess(self, accumulator):
        return accumulator

def usage():
    print
    print( "### Usage:  ",os.path.basename(sys.argv[0]),"<multicrab skim>" )
    print

def printCounters(accumulator):
    print("    Counters")
    for k in accumulator['counter'].keys():
        counter = "     "+k
        while len(counter) < 25:
            counter+=" "
        counter +="%s"%accumulator['counter'][k]
        print(counter)

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

    whitelist = ["TT"]


    datasets = multicrabdatasets.getDatasets(multicrabdir,whitelist=whitelist,blacklist=blacklist)
    pileup_data = multicrabdatasets.getDataPileupMulticrab(multicrabdir)
    lumi = multicrabdatasets.loadLuminosity(multicrabdir,datasets)

    outputdir = os.path.basename(os.path.abspath(multicrabdir))+"_processed"+starttime+lepton
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    import time
    t0 = time.time()

    print("Number of cores used",MAX_WORKERS)

    for i,d in enumerate(datasets):
        print("Dataset %s/%s %s"%(i+1,len(datasets),d.name))

        subdir = os.path.join(outputdir,d.name)
        if not os.path.exists(subdir):
            os.mkdir(subdir)
            os.mkdir(os.path.join(subdir,"results"))

        samples = {d.name: d.getFileNames()}
        """
        job_executor = processor.FuturesExecutor(workers = MAX_WORKERS)
        run = processor.Runner(
                        executor = job_executor,
                        schema=nanoevents.NanoAODSchema,
                        chunksize = 1E4,
                        )
        result = run(samples, 'Events', Analysis(year,d.isData,pileup_data,d.getPileup()))
        """
        result = processor.run_uproot_job(
            samples,
            "Events",
            Analysis(d.run,d.isData,pileup_data,d.getPileup()),
            processor.iterative_executor,
            {"schema": NanoAODSchema},
        )

        fOUT = ROOT.TFile.Open(os.path.join(subdir,"results","histograms.root"),"RECREATE")
        fOUT.cd()
        for key in result.keys():
            print("check histo key",key)
            histo = hist2root.convert(result[key]).Clone(key)
            histo.Write()
        fOUT.Close()

        printCounters(result)

    dt = time.time()-t0

    print("Processing time %s min %s s"%(int(dt/60),int(dt%60)))
    print("output in",outputdir)

if __name__ == "__main__":
    main()
    #os.system("ls -lt")
    #os.system("pwd")
