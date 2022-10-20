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

ANALYSISNAME = "JMENanoAOD_ZbAnalysisSkim.py"

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
	self.cMETCleaning   = Counter("Skim: METCleaning")
        self.cPassedEvents  = Counter("Skim: Passed events")

	self.objs = []


    def __del__(self):
        self.cAllEvents.Print()
        self.cTrigger.Print()
	self.cMETCleaning.Print()
        self.cPassedEvents.Print()

    def beginJob(self):

        self.h_pileup = ROOT.TH1F('pileup','',100,0,100)
        self.addObject(self.h_pileup)

	self.h_skimcounter = ROOT.TH1F("SkimCounter","",5,0,5)
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
        self.h_skimcounter.SetBinContent(4,self.cMETCleaning.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(4,self.cMETCleaning.name)
        self.h_skimcounter.SetBinContent(5,self.cPassedEvents.count)
        self.h_skimcounter.GetXaxis().SetBinLabel(5,self.cPassedEvents.name)
	self.h_skimcounter.Write()


    def analyze(self, event):

        if event._tree.GetListOfBranches().FindObject("Pileup_nTrueInt"):
            self.h_pileup.Fill(event.Pileup_nTrueInt)
        
        self.cAllEvents.increment()
        
        # selection
        # 2016 trigger
        triggerDecision = False
        if hasattr(event._tree, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ

        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ #Unprescaled                                                                                                  
        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL    #Prescaled                                                                                                    
        if hasattr(event._tree, 'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ #Unprescaled                                                                                                  
        if hasattr(event._tree, 'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL'):
            triggerDecision = triggerDecision or event.HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL    # Prescaled

        # 2017 trigger
        if hasattr(event._tree, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ #Prescaled                                                                                                          
        if hasattr(event._tree, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL'):
            triggerDecision = triggerDecision or event.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL #Prescaled                                                                                                             
        if hasattr(event._tree, 'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL'):
            triggerDecision = triggerDecision or event.HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL #Prescaled                                                                                                             
        if hasattr(event._tree, 'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ #Prescaled                                                                                                          
        if hasattr(event._tree, 'HLT_Mu17'):
            triggerDecision = triggerDecision or event.HLT_Mu17 #Unprescaled? or L1prescaled?                                                                                                                  
        if hasattr(event._tree, 'HLT_IsoMu24_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_IsoMu24_eta2p1 #Unprescaled                                                                                                                         

        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ #Unprescaled                                                                                                  
        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL #Unprescaled

        if hasattr(event._tree, 'HLT_HIMu17'):
            triggerDecision = True

        # 2018 trigger
        if hasattr(event._tree, 'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ #Prescaled                                                                                                          
        if hasattr(event._tree, 'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL'):
            triggerDecision = triggerDecision or event.HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL #Prescaled                                                                                                             
        if hasattr(event._tree, 'HLT_Mu19'):
            triggerDecision = triggerDecision or event.HLT_Mu19 #Unprescaled? or L1prescaled?                                                                                                                  
        if hasattr(event._tree, 'HLT_IsoMu24_eta2p1'):
            triggerDecision = triggerDecision or event.HLT_IsoMu24_eta2p1 #Unprescaled                                                                                                                         

        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL #Unprescaled                                                                                                     
        if hasattr(event._tree, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'):
            triggerDecision = triggerDecision or event.HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ #Unprescaled 


#        if not triggerDecision:
#            return False
	self.cTrigger.increment()

        # MET cleaning 
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#What_is_available_in_MiniAOD
        cleaningDecision = True
        if hasattr(event._tree, 'Flag_goodVertices'):
            cleaningDecision = cleaningDecision and event.Flag_goodVertices
        if hasattr(event._tree, 'Flag_globalSuperTightHalo2016Filter'):
            cleaningDecision = cleaningDecision and event.Flag_globalSuperTightHalo2016Filter
        if hasattr(event._tree, 'Flag_HBHENoiseFilter'):
            cleaningDecision = cleaningDecision and event.Flag_HBHENoiseFilter
        if hasattr(event._tree, 'Flag_HBHENoiseIsoFilter'):
            cleaningDecision = cleaningDecision and event.Flag_HBHENoiseIsoFilter
        if hasattr(event._tree, 'Flag_BadPFMuonFilter'):
            cleaningDecision = cleaningDecision and event.Flag_BadPFMuonFilter
        if hasattr(event._tree, 'Flag_BadPFMuonDzFilter'):
            cleaningDecision = cleaningDecision and event.Flag_BadPFMuonDzFilter

#	if not cleaningDecision:
#            return False
	self.cMETCleaning.increment()

        self.cPassedEvents.increment()

	return True

if __name__ == "__main__":
    SkimModule = lambda : Skim()



    files=["root://xrootd-cms.infn.it//store/data/Run2018A/DoubleMuon/NANOAOD/UL2018_MiniAODv2_JMENanoAODv9-v1/230000/0239BCA5-795D-8B42-AAFF-F825B4D1AFCF.root"]
#files=["root://xrootd-cms.infn.it//store/mc/RunIISummer19UL18NanoAOD/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/10000/0376ED51-0D18-8D42-8091-448B0EA2898D.root"]

    if not "BATCH_SYSTEM" in os.environ:
#        p=PostProcessor(".",files,"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root",jsonInput=runsAndLumis(),maxEntries=20000)
        p=PostProcessor(".",files,"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root",maxEntries=20000)
    else:
#        p=PostProcessor(".",inputFiles(),"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root",jsonInput=runsAndLumis())
        p=PostProcessor(".",inputFiles(),"",modules=[SkimModule()],provenance=True,fwkJobReport=True,haddFileName="events.root")

    p.run()

    os.system("mv *_Skim.root events.root")

    print "DONE"

