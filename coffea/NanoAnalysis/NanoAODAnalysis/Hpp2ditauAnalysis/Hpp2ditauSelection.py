import awkward as ak
import numpy
import sys
import os
import re

def triggerSelection(events,year):
    if '2016' in year:
        return events.HLT.LooseIsoPFTau50_Trk30_eta2p1_MET90
    if "2017" in year:
        return events.HLT.MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90
    if "2018" in year:
        return events.HLT.MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90

    print("Problem with triggerSelection")
    
def METCleaning(events,year):
    if '2016' in year:
        return (events.Flag.goodVertices &
                events.Flag.globalSuperTightHalo2016Filter &
                events.Flag.HBHENoiseFilter &
                events.Flag.HBHENoiseIsoFilter &
                events.Flag.BadPFMuonFilter &
                events.Flag.eeBadScFilter
        )

    if '2017' in year or '2018' in year:
        return (events.Flag.goodVertices &
                events.Flag.globalSuperTightHalo2016Filter &
                events.Flag.HBHENoiseFilter &
                events.Flag.HBHENoiseIsoFilter &
                events.Flag.BadPFMuonFilter &
                events.Flag.ecalBadCalibFilter
        )

    print("Problem with METCleaning")


def pileup_reweight(events,year):
    weights = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
    return weights    
