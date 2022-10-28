import awkward as ak
import numpy as np
import sys
import os
import re
import math

from METCleaning import METCleaning
from JetCorrections import JEC

import tools.Print

def triggerSelection(events,year,leptonflavor):
#def triggerSelection(events,year,leptonflavor,phtonpt):
    #if leptonflavor == 11:
    #    return (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL &
    #            events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
    #    )
    #if leptonflavor == 13:
    #    return events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ
    #return None
    # trigger thers
    return events.HLT.Photon30EB_TightID_TightIso

def lumimask(events,lmask):
    return (lmask.passed(events))

from  METCleaning import METCleaning

def leptonSelection(events, LEPTONFLAVOR):

#    if LEPTONFLAVOR == 13:
#        leptons = events.Muon
#
#        leptons = leptons[leptons.pt > 8]
#        leptons = leptons[leptons.tightId]
#        pfRelIsoMax = 0.15
#        leptons = leptons[leptons.pfRelIso04_all < pfRelIsoMax]
#        retEvents = events
#        retEvents["Muon"] = leptons
#        #retEvents = retEvents[(ak.num(leptons) >= 2)
#        #    & (ak.num(leptons) <= 3)
#        #    & (ak.sum(leptons.charge, axis=1) <= 1)
#        #    & (ak.sum(leptons.charge, axis=1) >= -1)]
#        # veto
#        retEvents = retEvents[(ak.num(leptons, axis=1) == 0)]
#        return retEvents
#
#    if LEPTONFLAVOR == 11:
#        print("leptonSelection for electrons not implemented")
#        #return (
#        #    (ak.num(events.Electron) == 2)
#        #    & (ak.num(events.Electron) <= 3)
#        #    & (ak.sum(events.Electron.charge, axis=1) <= 1)
#        #    & (ak.sum(events.Electron.charge, axis=1) >= -1)
#        #)
#        #retEvents = retEvents[(ak.num(leptons) == 0)]
#        return None
#    return None

    PT = 10
    muons = events.Muon
    electrons = events.Electron

    muons = muons[muons.pt > PT]
    electrons = electrons[electrons.pt > PT]

    lepVeto = (ak.num(muons.pt, axis=1) == 0 ) & (ak.num(electrons.pt, axis=1) == 0)

    return events[lepVeto]

def leptonPtCut(events, LEPTONFLAVOR):
    if LEPTONFLAVOR == 13:
        cut1 = 20
        cut2 = 10
        etamax = 2.3
        lepton = events.Muon
    if LEPTONFLAVOR == 11:
        cut1 = 25
        cut2 = 15
        etamax = 2.4
        lepton = events.Electron

    retEvents = events[(ak.sum((np.absolute(lepton.eta) < etamax) & (lepton.pt > cut1), axis=1) >= 1)
                     & (ak.sum((np.absolute(lepton.eta) < etamax) & (lepton.pt > cut2), axis=1) >= 2)]
    return retEvents

def Zboson(events,LEPTONFLAVOR):
    mZ = 91.1876
    if LEPTONFLAVOR == 13:
        leptons = events.Muon
    if LEPTONFLAVOR == 11:
        leptons = events.Electron

    combinations = ak.combinations(leptons, 2, fields=["first", "second"])
    combinations["Zboson"] = combinations.first + combinations.second
    combinations["diffZboson"] = np.absolute(combinations.Zboson.mass - mZ)

    combinations = combinations[(combinations.Zboson.charge == 0)]

    keep = ak.min(combinations.diffZboson,axis=1)
    combinations = combinations[(keep == combinations.diffZboson)]

    lepton1 = combinations.first
    lepton2 = combinations.second

    retEvents = events
    retEvents["Zboson"] = lepton1+lepton2
    retEvents["lepton1"] = lepton1
    retEvents["lepton2"] = lepton2

    retEvents = retEvents[ak.flatten(retEvents.Zboson.pt > 15)]
    retEvents = retEvents[ak.flatten(np.absolute(retEvents.Zboson.mass - mZ) < 20)]
    return retEvents

# Here
def photonSelection(events):
    photon = events.Photon # Access on Collection
    ptcut = 33
    etamax = 1.44
    photonId = 3 # cutBased Tight

    photon = photon[(photon.pt > ptcut) &
                    (np.abs(photon.eta) < etamax) &
                    (photon.cutBased == photonId)
                   ]

    retEvents = events
    retEvents["photon"] = photon
    retEvents = retEvents[(ak.num(retEvents.photon.pt, axis=1) == 1)]

    return retEvents

"""
def jets(events):
    jet_cands = events.Jet

    leptons = ak.with_name(ak.concatenate([events.Muon, events.Electron], axis=1), 'PtEtaPhiMCandidate')
    jet_cands = drClean(jet_cands,leptons)
#    jet_cands = drClean(jet_cands,events.Muon)
    #jet_cands = smear_jets(events,jet_cands)
#    jet_cands = JEC(events,jet_cands)
    return jet_cands
"""
def JetSelection(events):

    jet_cands = events.Jet
    jet_cands = drClean(jet_cands,events.photon)
    #jet_cands = drClean(jet_cands,events.lepton2)

    retEvents = events
    retEvents["Jet"] = jet_cands
    retEvents = retEvents[(ak.num(retEvents.Jet.jetId >= 4) > 0)]
    #TODO
    #retEvents = retEvents[(ak.num(retEvents.Jet.pt > 0) > 0)]
    return retEvents

def JetCorrections(events,jetCorrections):
    events['Jet'] = jetCorrections.apply(events)
    if hasattr(jetCorrections,'corrected_jets_l1'):
        events['Jet_L1RC'] = jetCorrections.corrected_jets_l1
    return events

def Jetvetomap(events,jetvetomap):
    return events[(jetvetomap.passmap(events.leadingJet))]

def Jetid(candidate):
    # Int_t Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto
    return (
        (candidate.jetId >= 4)
    )

# Here TODO
def PhiBB(events):
    phiBB_cut = 0.44 # 0.34 
    phiBB = abs(events.photon.delta_phi(events.leadingJet) - math.pi)
    return events[ak.flatten(
        (phiBB < phiBB_cut) | (phiBB > 2*math.pi-phiBB_cut)
    )]

def smear_jets(events,jets):
    print("check smear1",jets.pt)
    print("check smear2",jets.pt)
    return jets

def drClean(coll1,coll2,cone=0.3):
    from coffea.nanoevents.methods import vector
    j_eta = coll1.eta
    j_phi = coll1.phi
    l_eta = coll2.eta
    l_phi = coll2.phi

    j_eta, l_eta = ak.unzip(ak.cartesian([j_eta, l_eta], nested=True))
    j_phi, l_phi = ak.unzip(ak.cartesian([j_phi, l_phi], nested=True))
    delta_eta = j_eta - l_eta
    delta_phi = vector._deltaphi_kernel(j_phi,l_phi)
    dr = np.hypot(delta_eta, delta_phi)
    jets_noleptons = coll1[~ak.any(dr < cone, axis=2)]
    return jets_noleptons

def recalculateMET(jets):
    return 0.

def plotResponce(events,jets,Zboson,out,eweight):
    """
    leadingJet = jets[:,0]
    R_pT = leadingJet.pt/Zboson.pt
    
    MET = recalculateMET(jets)

    METuncl = -MET - jets_all.pt - Zboson.pt
    R_MPF = 1 + MET.pt*(cos(MET.phi)*Zboson.Px() + sin(MET.Phi())*Zboson.Py())/(Zboson.Pt()*Zboson.Pt());
    R_MPFjet1 = -leadingJet.Pt()*(cos(leadingJet.Phi())*Zboson.Px() + sin(leadingJet.Phi())*Zboson.Py())/(Zboson.Pt()*Zboson.Pt());
    R_MPFjetn = -jets_notLeadingJet.Pt()*(cos(jets_notLeadingJet.Phi())*Zboson.Px() + sin(jets_notLeadingJet.Phi())*Zboson.Py())/(Zboson.Pt()*Zboson.Pt());
    R_MPFuncl = -METuncl.Pt()*(cos(METuncl.Phi())*Zboson.Px() + sin(METuncl.Phi())*Zboson.Py())/(Zboson.Pt()*Zboson.Pt());

              hprof2D_ZpT_Mu_RpT[i]->Fill(Zboson.Pt(),mu,R_pT,eweight);
          hprof2D_ZpT_Mu_RMPF[i]->Fill(Zboson.Pt(),mu,R_MPFPF,eweight);
          hprof2D_ZpT_Mu_RMPFjet1[i]->Fill(Zboson.Pt(),mu,R_MPFjet1,eweight);
          hprof2D_ZpT_Mu_RMPFjetn[i]->Fill(Zboson.Pt(),mu,R_MPFjetn,eweight);
          hprof2D_ZpT_Mu_RMPFuncl[i]->Fill(Zboson.Pt(),mu,R_MPFuncl,eweight);
    """

def pileup_reweight(events,year):
    weights = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
    
