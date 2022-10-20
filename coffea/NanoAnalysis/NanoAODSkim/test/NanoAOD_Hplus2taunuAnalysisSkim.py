#!/usr/bin/env python
import os
import sys
import math
import PSet
import re

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import * 

#this takes care of converting the input files from CRAB
from PhysicsTools.NanoAODTools.postprocessing.framework.crabhelper import inputFiles,runsAndLumis

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

import ROOT

from array import array

ANALYSISNAME = "NanoAOD_Hplus2taunuAnalysisSkim"

class Counter():
    def __init__(self,name):
        self.name  = name
        self.count = 0
        
    def increment(self):
        self.count += 1
        
    def Print(self):
        self.name += " "
        while len(self.name) < 39:
            self.name += "."
        print self.name,self.count

class Skim(Module):
    def __init__(self):
        self.cControl       = Counter("Skim: control")
	self.cControl.increment()
        self.cAllEvents     = Counter("Skim: All events")
        self.cTrigger       = Counter("Skim: Trigger selection")
        self.cPassedEvents  = Counter("Skim: Passed events")

	self.objs = []


    def __del__(self):
        self.cAllEvents.Print()
        self.cTrigger.Print()
        self.cPassedEvents.Print()

    def beginJob(self):

        self.h_pileup = ROOT.TH1F('pileup','',200,0,200)
        self.addObject(self.h_pileup)

	self.h_skimcounter = ROOT.TH1F("SkimCounter","",4,0,4)
        self.addObject(self.h_skimcounter)


    def endJob(self):
	pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.dir = outputFile.mkdir("configInfo")


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):

	outputFile.cd()

	self.dir.cd()
        self.h_pileup.Write()

        
        self.h_skimcounter.SetBinContent(1,self.cControl.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(1,self.cControl.name)
        self.h_skimcounter.SetBinContent(2,self.cAllEvents.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(2,self.cAllEvents.name)
	self.h_skimcounter.SetBinContent(3,self.cTrigger.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(3,self.cTrigger.name)
        self.h_skimcounter.SetBinContent(4,self.cPassedEvents.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(4,self.cPassedEvents.name)
	self.h_skimcounter.Write()


    def analyze(self, event):

        if event._tree.GetListOfBranches().FindObject("Pileup_nTrueInt"):
            self.h_pileup.Fill(event.Pileup_nTrueInt)
        
        self.cAllEvents.increment()
        
        # selection
        # 2016 trigger
        triggerDecision = False
        if hasattr(event._tree, 'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80'):
            triggerDecision = triggerDecision or event.HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80

        if hasattr(event._tree, 'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET90'):
            triggerDecision = triggerDecision or event.HLT_LooseIsoPFTau50_Trk30_eta2p1_MET90

        if hasattr(event._tree, 'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET110'):
            triggerDecision = triggerDecision or event.HLT_LooseIsoPFTau50_Trk30_eta2p1_MET110

        if hasattr(event._tree, 'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120'):
            triggerDecision = triggerDecision or event.HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120

        if hasattr(event._tree, 'HLT_VLooseIsoPFTau120_Trk50_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_VLooseIsoPFTau120_Trk50_eta2p1

        if hasattr(event._tree, 'HLT_VLooseIsoPFTau140_Trk50_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_VLooseIsoPFTau140_Trk50_eta2p1

        if hasattr(event._tree, 'HLT_MET150'):
            triggerDecision = triggerDecision or event.HLT_MET150

        if hasattr(event._tree, 'HLT_MET200'):
            triggerDecision = triggerDecision or event.HLT_MET200

        if hasattr(event._tree, 'HLT_MET250'):
            triggerDecision = triggerDecision or event.HLT_MET250


        # 2017 trigger
        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau100HighPtRelaxedIso_Trk50_eta2p1_1pr'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau100HighPtRelaxedIso_Trk50_eta2p1_1pr

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr

        if hasattr(event._tree, 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'):
            triggerDecision = triggerDecision or event.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight



        # 2018 trigger
        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90
        
        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau100HighPtRelaxedIso_Trk50_eta2p1_1pr'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau100HighPtRelaxedIso_Trk50_eta2p1_1pr

        if hasattr(event._tree, 'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr'):
            triggerDecision = triggerDecision or event.HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr

        if hasattr(event._tree, 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'):
            triggerDecision = triggerDecision or event.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight




#        if not triggerDecision:
#            return False
	self.cTrigger.increment()

        self.cPassedEvents.increment()

	return True

if __name__ == "__main__":
    SkimModule = lambda : Skim()



#files=["root://xrootd-cms.infn.it//store/data/Run2016H/Tau/NANOAOD/Nano1June2019-v1/240000/FB1AF208-9FEC-0445-AB21-772D27660951.root"]
#files=["root://xrootd-cms.infn.it//store/mc/RunIISummer20UL17NanoAODv9/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/120000/260C8C6D-BF74-B549-95FD-A732768F97BF.root"]
    files=["root://xrootd-cms.infn.it//store/data/Run2017B/Tau/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/120000/0CC24046-D478-5949-BE18-A0CA6EDAC879.root"]


    if len(sys.argv) == 1:
        p=PostProcessor(".",files,"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root",maxEntries=20000)
    else:
        p=PostProcessor(".",inputFiles(),"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root")

    p.run()

    os.system("mv *_Skim.root events.root")

    print "DONE"

