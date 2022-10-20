import itertools
import sys
import re
import os

import awkward as ak
import numpy as np

import hist

basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python"))

import TransverseMass

def get_rBB_min(events):
    tau = events.Tau
    jets = events.Jet
    met = events.MET

    jets_delta_phi = jets.delta_phi(met)**2
    tau_delta_phi = (np.pi - np.absolute(tau.delta_phi(met)))**2
    
    jets_min = ak.min(jets_delta_phi, axis=1)
    taus_min = ak.min(tau_delta_phi, axis=1)

    return np.sqrt(taus_min + jets_min)
 
def without(d, key):
    res = d.copy()
    res.pop(key)
    return res

class AnalysisHistograms():
    def __init__(self, isData, use_nn=False, use_m=False, unblind = False, after_std=False):
        self.histograms = {}
        self.isData = isData
        self.use_nn = use_nn
        self.use_m = use_m

        #only blind histograms if nominal selection, is data and using DNN
        self.unblind = (not self.isData) or (not self.use_nn) or (self.use_nn and unblind) or after_std
        self.after_std = after_std

    def book(self):

        variation_nominal = {'Nominal': '(TauIsolation > 0)'}
        variation_tau_iso = {'Nominal': '(TauIsolation > 0)', 'Inverted': '(TauIsolation < 1)'}
        variation_Rtau = {'Rtau75': '(leadTau.leadTkPtOverTauPt < 0.75)', 'Rtau1': '(leadTau.leadTkPtOverTauPt > 0.75)'}
        ftau_variation_pt = {'lowpt': '(leadTau.pt < 60)', 'lmidpt': '(leadTau.pt > 60) & (leadTau.pt < 80)', 'hmidpt': '(leadTau.pt > 80) & (leadTau.pt < 100)', 'highpt': '(leadTau.pt > 100)', 'incl': ''}
        ftau_variation_eta = {'loweta': '(leadTau.abseta < 0.6)', 'mideta': '(leadTau.abseta > 0.6) & (leadTau.abseta < 1.4)', 'higheta': '(leadTau.abseta > 1.4)', 'incl': ''}
        # variation_nn_score = {
        #     'nn_50': '(nn_score < 0.50)',
        #     'nn_90': '(nn_score > 0.5) & (nn_score < 0.9)',
        #     'nn_1': '(nn_score > 0.9)',
        # }
        variation_nn_score = {'nn_1': '(nn_score > 0.5)',}
        variation_gen = {
            'incl': '',
            'EWKGenuineTau': '(leadTau.Genuine == True)',
            'EWKFakeTau': '(leadTau.Genuine == False)',
            'MatchedTau': '(leadTau.Matched == True)',
        }

        variationSet = {
            'Rt': variation_Rtau,
        }
        if self.use_nn:
            variationSet.update({
                'nn': variation_nn_score
            })
        variation_qcd = {
            'ftau_pt': ftau_variation_pt,
            'ftau_eta': ftau_variation_eta,
            'fake_tau': variation_tau_iso,
            'gen': variation_gen,
        }
        met_variations = dict(**variationSet, **variation_qcd)
        commonVariations = dict(**{'nominal_selection': variation_nominal}, **variationSet)

        self.variations = met_variations
        
        bins_mt     = np.concatenate([np.arange(0,1000,5), np.array([3000])], axis = 0)
        bins_teta   = np.linspace(-2.1,2.1,22) #FIXME this is not the same binning as original analysis
        bins_jeteta = np.linspace(-4.5,4.5,60) #FIXME this is not the same binning as original analysis
        bins_rBB    = np.linspace(0,260,26)
        bins_tpt    = [50,60,80,100,150,200,300,400,2000]
        bins_met    = np.linspace(0,800,801)
        bins_metvis = [0,20,40,60,80,100,120,140,160,200,250,300,1500]
        bins_bjetpt = [30,50,70,90,110,130,150,200,300,400,1500]
        bins_Rtau   = np.linspace(0.,1.,20)

        bins_nn     = np.linspace(0,1,50)
        bins_m_pred = np.geomspace(80,3000,50)

        prefix = ""
        if self.after_std:
            prefix += "AfterStdSelections_"

        self.h_counters = VariationHistograms("counters", [0,1], met_variations, self.isData, self.unblind)

        self.h_met = VariationHistograms(prefix + 'h_met', bins_met, met_variations, self.isData, self.unblind)
        self.histograms.update(self.h_met.histograms)
        self.h_mt = VariationHistograms(prefix + 'h_mt', bins_mt, met_variations, self.isData, self.unblind)
        self.histograms.update(self.h_mt.histograms)
        if not self.after_std:
            self.h_teta = VariationHistograms(prefix + 'h_taueta', bins_teta, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_teta.histograms)
            self.h_jeteta = VariationHistograms(prefix + 'h_jeteta', bins_jeteta, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_jeteta.histograms)
            self.h_rBB = VariationHistograms(prefix + 'h_rBB', bins_rBB, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_rBB.histograms)
            self.h_tpt = VariationHistograms(prefix + 'h_taupt', bins_tpt, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_tpt.histograms)
            self.h_bjetpt = VariationHistograms(prefix + 'h_bjetpt', bins_bjetpt, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_bjetpt.histograms)
            self.h_Rtau = VariationHistograms(prefix + 'h_Rtau', bins_Rtau, without(commonVariations,"Rt"), self.isData, self.unblind)
            self.histograms.update(self.h_Rtau.histograms)
            self.h_met_vis = VariationHistograms(prefix + 'h_metvis', bins_metvis, commonVariations, self.isData, self.unblind)
            self.histograms.update(self.h_met_vis.histograms)


            if self.use_nn:
                self.h_nn = VariationHistograms(prefix + 'h_nn_score', bins_nn, without(commonVariations, 'nn'), self.isData, self.unblind)
                self.histograms.update(self.h_nn.histograms)
                if self.use_m:
                    self.h_m_pred = VariationHistograms(prefix + 'h_nn_masspred', bins_m_pred, met_variations, self.isData, self.unblind)
                    self.histograms.update(self.h_m_pred.histograms)

        return self.histograms

    def addHistogram(self, name, bins, variable_binning = False):
        if variable_binning:
            self.histograms[name] = (
                hist.Hist.new
                .Var(bins, name="value", label=name)
                .Weight()
        )  
        else:
            nbins, binmin, binmax = bins
            self.histograms[name] = (
                hist.Hist.new
                .Reg(nbins, binmin, binmax, name="value", label=name)
                .Weight()
            )

    def add2DHistogram(self, name, xbins, ybins):
        self.histograms[name] = (
            hist.Hist.new
            .Var(xbins, name="x", label="x label")
            .Var(ybins, name="y", label="y label")
            .Weight()
        )

    def cloneHistogram(self, nameOrig, nameClone):
        self.histograms[nameClone] = self.histograms[nameOrig].copy()

    def fill_counters(self, name, events, out):
        self.h_counters.fill_only_counters(name, events, out, events.weight)
        return out

    def fill(self,events,out):

        if len(events) == 0: return out

        weight = events.weight
        tau = events.Tau
        isoTau = tau[tau.isolated]
        invTau = tau[~tau.isolated]
        leadTau = ak.where(events.TauIsolation, isoTau, invTau)[:,0]
        leadTau['abseta'] = np.absolute(leadTau.eta)
        events["leadTau"] = leadTau
        leadJet = events.Jet[:,0]

        met = events.MET.pt
        mt = TransverseMass.reconstruct_transverse_mass(leadTau, events.MET)

        if not self.after_std:
            teta = leadTau.eta
            jeteta = leadJet.eta
            rBB = get_rBB_min(events) * 180 / np.pi
            tpt = leadTau.pt
            bjetpt = events.BJet.pt[:,0]
            Rtau = leadTau.leadTkPtOverTauPt


        self.h_met.fill(events, out, met, weight)
        self.h_mt.fill(events, out, mt, weight)
        if not self.after_std:
            self.h_teta.fill(events, out, teta, weight)
            self.h_jeteta.fill(events, out, jeteta, weight)
            self.h_rBB.fill(events, out, rBB, weight)
            self.h_tpt.fill(events, out, tpt, weight)
            self.h_bjetpt.fill(events, out, bjetpt, weight)
            self.h_Rtau.fill(events, out, Rtau, weight)
            self.h_met_vis.fill(events, out, met, weight)

            if self.use_nn:
                score = events.nn_score
                self.h_nn.fill(events, out, score, weight)
                if self.use_m:
                    mass_pred = events.nn_masspred
                    self.h_m_pred.fill(events, out, mass_pred, weight)

        return out

class VariationHistograms():
    def __init__(self,name,bins,var,isData,is_unblinded):
        self.histograms = {}
        self.isData = isData
        self.is_unblinded = is_unblinded
        self.selection = {}
        self.weights = {}

        variations = []
        varSelections = {}

        for key in var.keys():
            variations.append(list(var[key].keys()))
            for k2 in var[key].keys():
                varSelections[k2] = var[key][k2]

        nameBase = name
        self.namebase = nameBase
        for combination in list(itertools.product(*variations)):
            hname = nameBase
            sele = None

            for comb in combination:
                if not (comb == '' or 'incl' in comb or comb == 'PSWeight'):
                    hname = hname + '_%s'%comb

                    if 'weight' in comb or 'Weight' in comb:
                        self.weights[hname] = comb
                        continue
                    if sele == None:
                        sele = varSelections[comb]
                    else:
                        sele = sele + " & " + varSelections[comb]
            self.selection[hname] = sele
            self.book(hname,bins)

        self.selection_re = re.compile("(?P<variable>[\w\.]+)\s*(?P<operator>\S+)\s*(?P<value>\S+)\)")

    def book(self,name,bins):
        self.histograms[name] = (
            hist.Hist.new
            .Var(bins, name="x", label=name)
            .Weight(label=name, name=name)
        )

    def fill_only_counters(self,name,events,out,weight):
        for key in self.histograms:
            sele, f = self.select(events,key)
            w = self.getweight(events,key)
            selected_w = weight[sele]*w[sele]
            cat_name = key.split("_")[1:]
            for i in f:
                cat_name[i] = ""
            counter_name = name + " " + " ".join((n for n in cat_name if n.strip()))
            if counter_name in out['unweighted_counter']: continue
            out['unweighted_counter'][counter_name] = len(selected_w)
            out['weighted_counter'][counter_name] = ak.sum(selected_w)
        return out

    def fill(self,events,out,x,weight):
        for key in self.histograms:
            sele, _ = self.select(events,key)
            w = self.getweight(events,key)
            selected_x = x[sele]
            selected_w = weight[sele]*w[sele]
            out[key].fill(x=selected_x,
                          weight=selected_w)
        return out

    def getweight(self,events,variable):
        w = np.array([1.0]*len(events))
        if variable in self.weights.keys():
            if 'PSWeight' in self.weights[variable] and len(self.weights[variable]) > 8:
                weightnumber = int(self.weights[variable].replace('PSWeight',''))
                w = events['PSWeight'][:,weightnumber]
        return w

    def select(self,events,variable):
        return_sele = (events["event"] > 0)
        if not self.is_unblinded:
            # NOTE: this currently automatically unblinds all events that passed the INVERTED tau isolation check,
            # because they are used for the fake tau background measurement
            return_sele = return_sele & (events["TauIsolation"] <= 0)
        if self.selection[variable] == None:
            return return_sele

        selections = self.selection[variable].split('&')

        failed = []
        keys = []
        for s in selections:
            match = self.selection_re.search(s)
            if match:
                keys.append(match.group('variable'))
                rec_variables_list = tuple(keys[-1].split('.'))
                operator = match.group('operator')
                value    = eval(match.group('value'))

                try:
                    variable = events[rec_variables_list]
                    failed.append(False)
                except ValueError:
                    failed.append(True)
                    continue

                if operator == '<':
                    return_sele = return_sele & (variable < value)
                if operator == '>':
                    return_sele = return_sele & (variable > value)
                if operator == '==':
                    return_sele = return_sele & (variable == value)

        failed_cats = []
        m = 0
        for n, (f, k) in enumerate(zip(failed, keys)):
            if k not in keys[:n]:
                if f: failed_cats.append(m)
                m += 1

        return return_sele, failed_cats