import awkward as ak
import numpy as np
import sys
import os
import re

from JetCorrections import JEC
from Btag import Btag
import Hplus2taunuSelection as selection

##################################
# This module imports functions from Hplus2taunuSelection.py 
# and selects all events for one given channel.
# Functions return events that pass all selection criterias
# for chosen channel
##################################

def tau_plus_jets_channel(events):
    # Selects all events for tau + jets decay channel
    events = selection.tau_selection(
            events = events,
            pt_min = 50,
            eta_max = 2.1
    )

    # max_jets=None means that there is no limit for total amount of jets. same for max_b_jets
    events = selection.jet_selection(
        events = events,
        max_jets = None,
        min_jets = 3,
        pt_min = 30,
        eta_max = 4.7,
        max_b_jets = None,
        eta_b_jet = 2.4
    )

    events = selection.pt_miss_cut(
        events = events, 
        pt_miss_cut = 90
    )

    events = selection.Rbb_min(events)

    return events



def lepton_plus_tau_channel(events):
    # Selects all events for tau + lepton (muon or electron)  decay channel
    events = selection.lepton_tau_selection(
        events = events,
        pt_min = 20,
        eta_max = 2.3
    )
    events = selection.delta_phi_leading_jet(
        events = events
    )

    events = selection.jet_selection(
        events = events,
        max_jets = 3,
        min_jets = 1,
        pt_min = 30,
        eta_max = 2.4,
        max_b_jets = 3,
        eta_b_jet = 2.4
    )

    events = selection.pt_miss_cut(
        events = events,
        pt_miss_cut = 70
    )

    return events


def lepton_no_tau_channel(events):
    # Selects all events for lepton (muon or electron) decay channel where no tau is selected
    
    events = selection.lepton_no_tau_selection(
        events = events
    )

    events = selection.delta_phi_min(
        events = events
    )


    events = selection.jet_selection(
        events = events,
        max_jets = 3,
        min_jets = 2,
        pt_min = 30,
        eta_max = 2.4,
        max_b_jets = 3,
        eta_b_jet = 2.4
    ) 

    events = selection.pt_miss_cut(
        events = events,
        pt_miss_cut = 100
    )

    # Select events with delta_phi between leading jet and lepton if it is > 0.5
    events = selection.delta_phi_leading_jet(
        events = events
    )

    events = selection.delta_phi_min(
        events = events
    )

    return events