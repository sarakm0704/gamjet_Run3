from builtins import breakpoint
from typing import overload
import awkward as ak
import numpy as np
import sys
import os
import re
import glob
from functools import partial

from JetCorrections import JEC
from Btag import Btag


basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Hplus2taunuAnalysis/ANN"))

import datasetUtils
from disCo import InputSelector, MinMaxScaler, Sanitizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class ClassifierModel():
    def __init__(self, savedir, param_mass = False, regress_mass = False, nn_working_point = -1.0):

        self.parametrized = param_mass
        self.regress_mass = regress_mass
        folds = glob.glob(savedir + "/fold_*/model_trained.h5")
        input_vars = datasetUtils.load_input_vars(savedir)
        input_feature_keys = []
        for var in input_vars:
            if var in datasetUtils.COLUMN_KEYS.keys():
                for basename in datasetUtils.COLUMN_KEYS[var]:
                    input_feature_keys.append(basename)
            else: input_feature_keys.append(var)
        self.input_feature_keys = input_feature_keys

        try:
            self.mass_idx = input_feature_keys.index("mtrue")
        except:
            if param_mass: raise Exception("cannot find mass index in neural network inputs")
            else: pass

        try:
            nn_mt_idx = input_feature_keys.index("mt")
        except:
            raise Exception("cannot find variable of interest (transverse mass) in neural network inputs")

        with tf.keras.utils.custom_object_scope({"InputSelector": InputSelector,
                                                 "MinMaxScaler": MinMaxScaler,
                                                 "Sanitizer": Sanitizer}):
            self.models = [tf.keras.models.load_model(f, compile = False) for f in folds]
            [m.compile(run_eagerly = True) for m in self.models] # No graph tracing for now since ran into trouble with retracing
        self.k = len(folds)

        self.feature_generating_func = partial(datasetUtils.Writer.convert, input_vars, datasetUtils.MDATA, inference=True)

        self.cut = nn_working_point
        self.nn_mt_idx = nn_mt_idx

    def selection(self, events, cut = None, mass_hypot=None):

        if len(events) == 0: return events

        # set the working point
        if cut is None:
            cut = self.cut

        in_tensor = self.feature_generating_func(events)

        # replace mass parameter if using a single mass point
        if mass_hypot is not None:
            if not self.parametrized:
                msg = "Cannot specify mass hypothesis when using a model that isn't parametrized w.r.t. the H+ mass"
                raise Exception(msg)
            mass_hypot = tf.zeros_like(in_tensor[...,0]) + mass_hypot
            in_tensor = tf.concatenate([in_tensor[...,:self.mass_idx], mass_hypot, in_tensor[...,self.mass_idx+1:]], axis=-1)

        # for parametrized nn, use transverse mass as an approximation for the H+ mass
        elif self.parametrized:
            mass_approx = in_tensor[...,self.nn_mt_idx]
            in_tensor = tf.concatenate([in_tensor[...,:self.mass_idx], mass_approx, in_tensor[...,self.mass_idx+1:]], axis=-1)

        event_folds = events.event % self.k
        predfolds = np.unique(event_folds)

        # get predictions using models that haven't seen the events during training
        predslist = []
        for fold in predfolds:
            fold_x = in_tensor[(events.event % self.k == fold).to_numpy()]
            predslist.append(self.call(fold_x, fold))

        # collect predictions from different folds into single arrays 
        preds = {}
        for key in predslist[0]:

            # instantiate numpy array that will contain predictions for all events
            preds[key] = np.empty_like(events.event, dtype=float)

            # fill the array with predictions
            for n, fold in enumerate(predfolds):

                # get indices where to put preds from this fold
                idx = np.asarray(event_folds == fold).nonzero()[0]

                # place fold preds into larger array
                np.put(preds[key], idx, tf.squeeze(predslist[n][key]).numpy())

        # check which events passed the selection
        passed = preds['class_out'] >= cut

        # fill event fields for histos
        events['nn_score'] = preds['class_out']
        if self.regress_mass:
            events['nn_masspred'] = preds['mass_out']

        return events[passed]

    def call(self, x, n):
        return self.models[n](x, training=False)

def nn_server_sele(events, cut=None, mass_hypot=None):
    pass

def triggerSelection(events,year):
    if '2016' in year:
        return events.HLT.LooseIsoPFTau50_Trk30_eta2p1_MET90
    if "2017" in year:
        return events.HLT.MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90
    if "2018" in year:
        return events.HLT.MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90

    print("Problem with triggerSelection")

def lumimask(events,lmask):
    return (lmask.passed(events))

from  METCleaning import METCleaning

def tau_identification(events, pt_min = 50, eta_max = 2.1):
    # step 4: Identify tau particles.
    # Accept events 0 and 1, reject events 10 and 11

    tau = events.Tau

    #check that offline tau matches deltaR < 0.1 with HLT tau object
    HLT_tau = events.TrigObj[events.TrigObj.id==15]

    # metric_table/ak.min() returns None if the event has no HLT taus so we must also filter based on that
    delta_r = ak.min(tau.metric_table(HLT_tau), axis=2)
    tau = tau[
        (delta_r < 0.1) &
        (~ak.is_none(delta_r))
    ]

    # pt and eta cuts
    tau = tau[
        (tau.pt > pt_min)&
        (np.absolute(tau.eta) < eta_max)
    ]

    # decay mode cuts
    tau = tau[
        (tau.decayMode == 0) |
        (tau.decayMode == 1)
    ]

    # leading electrically charged particle with pt_ldgTk > 30 GeV
    pt_ldgTk = tau.pt * tau.leadTkPtOverTauPt
    tau = tau[
        (pt_ldgTk > 30)
    ]

    # AntiEle, tight
    tau = tau[
        (tau.idDeepTau2017v2p1VSe >= 32)
    ]
    # AntiMuon, loose
    tau = tau[
        (tau.idDeepTau2017v2p1VSmu >= 2)
    ]
    # loose isolation criteria
    isolation = tau.idDeepTau2017v2p1VSjet >= 8
    tau["isolated"] = isolation
    #TODO: contains one charged particle?

    events["Tau"] = tau
    events["TauIsolation"] = ak.sum(tau.isolated, axis=1) > 0
    
    keep = ak.num(tau) > 0

    events = events[
        (keep) 
    ]

    return events

def label_taus(events):
    tau = events.Tau
    gen_tau = events.GenVisTau
    genuine_matches = tau.nearest(gen_tau, threshold=0.1)
    gen_fakes = events.GenPart[
        (abs(events.GenPart.pdgId) == 11) |
        (abs(events.GenPart.pdgId) == 13) &
        (events.GenPart.pt > 10)
    ]
    lep_matches = tau.nearest(gen_fakes,threshold = 0.1)
    genuinetau = ~ak.is_none(genuine_matches,axis=1)
    tau["Genuine"] = genuinetau
    tau["Matched"] = ~ak.is_none(lep_matches,axis=1) | genuinetau
    events['Tau'] = tau
    return events


def R_tau_selection(events):
    # Selects events where R_Tau > 0.75.
    # This is done by selecting leading charged tau particle,
    # which has over 75% energy of the event. 
    tau = events.Tau
    
    tau = tau[
        (tau.leadTkPtOverTauPt > 0.75) 
    ]

    events['Tau'] = tau

    events = events[
        (ak.num(tau) > 0)
    ]

    return events 


def isolated_electron_veto(events):
    # step 5: isolated electron veto
    # _mvaFall17V2noIso_WPL
    max_pt = 15
    min_eta = 2.5 

    electron = events.Electron

    electron = electron[
        (electron.pt > max_pt) & 
        (np.absolute(electron.eta) < min_eta) &
        (electron.miniPFRelIso_all < 0.4)
    ]
 
    events['Electron'] = electron

    events = events[
        (ak.num(electron) == 0)
    ]

    return events


def isolated_muon_veto(events):
    # step 6: isolated muon veto
    # looseId
    max_pt = 10
    min_eta = 2.5

    muon = events.Muon

    muon = muon[
        (muon.pt > max_pt) & 
        (np.absolute(muon.eta) < min_eta) & 
        (muon.miniPFRelIso_all < 0.4)
    ]

    events['Muon'] = muon

    events = events[
        (ak.num(muon) == 0)
    ]

    return events


def hadronic_pf_jets(events):
    # step 7: choose hadronic pf jets with loose ID.
    # Choose jets with jetId > 4. 
    pt_min = 30
    eta_max = 4.7

    jets = events.Jet
  
    jets = jets[
        (jets.pt > pt_min) & 
        (np.absolute(jets.eta) < eta_max) &
        (ak.num(jets.jetId > 4) >= 3)
    ]    

    events["Jet"] = jets


    events = events[
        (ak.num(jets) >= 3)
    ]

    return events


def b_tagged_jets(events, wp=None):
    # step 8: B tagged jets, using CSVv2 with medium Working Point
    jets = events.Jet
    eta_max = 2.4

    btag = Btag('btagDeepB', "2017")

    if wp == 'loose':
        cut = btag.loose()
    elif wp == 'tight':
        cut = btag.tight()
    else: cut = btag.medium()
    

    jets = jets[
        (np.absolute(jets.eta) < eta_max) &
        (jets.btagCSVV2 > cut)
    ]

    events["BJet"] = jets

    events = events[
        ak.num(jets) > 0
    ]

    return events


def met_cut(events, pt_min = 90):
    # step 9: Missing transverse energy cut
    met = events.MET

    keep = met.pt > pt_min

    events = events[
        keep
    ]

    return events

def Rbb_min(events):
    # step 10: back-to-back angular cut
    # Select angular discriminant angle = 40 degrees = 0.698 radians
    tau = events.Tau
    jets = events.Jet
    met = events.MET

    # Select smallest taus and jets for each event when calculating Delta_Phi.
    # This assures that each event that passes, has minimum value for R_bb min.
    jets_delta_phi = jets.delta_phi(met)**2
    tau_delta_phi = (np.pi - np.absolute(tau.delta_phi(met)))**2
    
    jets_min = ak.min(jets_delta_phi, axis=1)
    taus_min = ak.min(tau_delta_phi, axis=1)

    events = events[
        (np.sqrt(taus_min + jets_min) > 0.689)
    ]

    return events

def fake_tau_pt_selection(events, pt_cut):
    # perform pt cut for given pt. Tau pt as variable for bins
    # pt < 60 GeV
    # 60 < pt < 80 GeV
    # 80 < pt < 100 GeV and
    # pt > 100 Gev

    tau = events.Tau

    if pt_cut == 60:
        tau = tau[
            (tau.pt < 60)
        ]

    elif pt_cut == 80:
        tau = tau[
            (tau.pt < 80) &
            (tau.pt > 60)
        ]

    elif pt_cut == 100:
        tau = tau[
            (tau.pt < 100) & 
            (tau.pt > 80)
        ]

    else:
        tau = tau[
            (tau.pt > 100)
        ]

    events = events[
        (ak.num(tau) > 0)
    ]    

    events['Tau'] = tau

    return events
    

def fake_tau_eta_selection(events, eta_cut):
    # Selection of Tau eta binning, from 
    # |eta| < 0.6
    # 0.6 < |eta| < 1.4 and
    # |eta| > 1.4

    tau = events.Tau

    if eta_cut == 0.6:
        tau = tau[
            (np.absolute(tau.eta) < 0.6)
        ]

    elif eta_cut == 1.4:
        tau = tau[
            (np.absolute(tau.eta) > 0.6) & 
            (np.absolute(tau.eta) < 1.4)
        ]

    else: 
        tau = tau[
            (np.absolute(tau.eta) > 1.4)
        ]

    events = events[
        (ak.num(tau) > 0)
    ]    

    events['Tau'] = tau

    return events    

def met_fake_tau_selection(events):
    # perform pt cut for given pt. MET pt as variable for bins
    # pt < 60 GeV
    # 60 < pt < 80 GeV
    # 80 < pt < 100 GeV and
    # pt > 100 Gev
    met = events.MET

    if pt_cut == 60:
        met = met[
            (met.pt < 60)
        ]

    elif pt_cut == 80:
        met = met[
            (met.pt < 80) &
            (met.pt > 60)
        ]

    elif pt_cut == 100:
        met = met[
            (met.pt < 100) & 
            (met.pt > 80)
        ]

    else:
        met = met[
            (met.pt > 100)
        ]

    events = events[
        (ak.num(met) > 0)
    ]    

    events.MET = met

    return events
    


def tau_selection(events):
    # Uses all criteria to identify desired tau events

    events = tau_identification(events)
    
    events = R_tau_selection(events)

    events = isolated_electron_veto(events)

    events = isolated_muon_veto(events)

    events = hadronic_pf_jets(events)
    
    events = b_tagged_jets(events)

    events = met_cut(events)

    events = Rbb_min(events)

    return events
