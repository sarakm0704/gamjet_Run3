import itertools
import sys
import re

import awkward as ak
import numpy as np

import hist


class AnalysisHistograms():
    def __init__(self, isData):
        self.histograms = {}
        self.isData = isData

    def book(self):
        bins_y = np.arange(-3.99,6.01,0.02)
        bins_zpt = np.array([12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 105, 130, 175, 230, 300, 400, 500, 700, 1000, 1500])
        bins_mu = np.arange(1,100,1)
        bins_01 = np.arange(0,1.01,0.01)
        self.addHistogram('Z_mass', 60, 60, 120)
#        self.add2DHistogram('h_Zpt_RpT',bins_zpt,bins_y)
#        self.add2DHistogram('h_Zpt_RMPF',bins_zpt,bins_y)

        variation_alpha = {'alpha30': '(alpha < 0.3)','alpha100': '(alpha < 1.0)'}
        variation_eta = {'eta13': '(eta < 1.3) & (eta > -1.3)','eta25': '(eta < 2.5) & (eta > -2.5)'}
        variation_btag = {'incl': '', 'btagDeepBtight': '(btag == True)', 'btagDeepCtight': '(ctag == True)', 'gluontag': '(gluontag == True)', 'quarktag': '(quarktag == True)', 'notag': '(notag == True)'}
        variation_gen = {'incl': ''}
        variation_psweight= {'PSWeight':''}

        if not self.isData:
            variation_psweight = {'PSWeight':'', 'PSWeight0': '', 'PSWeight1': '', 'PSWeight2': '', 'PSWeight3': ''}
            variation_gen = {
                'incl': '',
                'genb': '(ljetGenFlavor == 5)',
                'genc': '(ljetGenFlavor == 4)',
                'genuds': '(ljetGenFlavor > 0) & (ljetGenFlavor < 4)',
                'geng': '(ljetGenFlavor == 21)',
                'unclassified': '(ljetGenFlavor == 0)'
            }

        variationSet = {
            'alpha': variation_alpha,
            'eta': variation_eta,
            'btag': variation_btag,
            'gen': variation_gen,
            'psweight': variation_psweight
        }

        self.h_zpt_rpt = VariationHistograms("h",{"ZpT":bins_zpt,"RpT":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rpt.histograms)
        """
        self.h_zpt_rmpf = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpf.histograms)
        self.h_zpt_rmpfjet1 = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpfjet1.histograms)
        self.h_zpt_rmpfjetn = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpfjetn.histograms)
        self.h_zpt_rmpfuncl = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpfuncl.histograms)
        self.h_zpt_rmpfx = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpfx.histograms)
        self.h_zpt_rmpfjet1x = VariationHistograms("h",{"ZpT":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_zpt_rmpfjet1x.histograms)

        self.h_jpt_rpt = VariationHistograms("h",{"JetPt":bins_zpt,"RpT":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rpt.histograms)
        self.h_jpt_rmpf = VariationHistograms("h",{"JetPt":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpf.histograms)
        self.h_jpt_rmpfjet1 = VariationHistograms("h",{"JetPt":bins_zpt,"RMPFjet1":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpfjet1.histograms)
        self.h_jpt_rmpfjetn = VariationHistograms("h",{"JetPt":bins_zpt,"RMPFjet2":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpfjetn.histograms)
        self.h_jpt_rmpfuncl = VariationHistograms("h",{"JetPt":bins_zpt,"RMPFuncl":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpfuncl.histograms)
        self.h_jpt_rmpfx = VariationHistograms("h",{"JetPt":bins_zpt,"RMPFx":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpfx.histograms)
        self.h_jpt_rmpfjet1x = VariationHistograms("h",{"JetPt":bins_zpt,"RMPFjet1x":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_jpt_rmpfjet1x.histograms)

        self.h_ptave_rpt = VariationHistograms("h",{"PtAve":bins_zpt,"RpT":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rpt.histograms)
        self.h_ptave_rmpf = VariationHistograms("h",{"PtAve":bins_zpt,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpf.histograms)
        self.h_ptave_rmpfjet1 = VariationHistograms("h",{"PtAve":bins_zpt,"RMPFjet1":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpfjet1.histograms)
        self.h_ptave_rmpfjetn = VariationHistograms("h",{"PtAve":bins_zpt,"RMPFjetn":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpfjetn.histograms)
        self.h_ptave_rmpfuncl = VariationHistograms("h",{"PtAve":bins_zpt,"RMPFuncl":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpfuncl.histograms)
        self.h_ptave_rmpfx = VariationHistograms("h",{"PtAve":bins_zpt,"RMPFx":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpfx.histograms)
        self.h_ptave_rmpfjet1x = VariationHistograms("h",{"PtAve":bins_zpt,"RMPFjet1x":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_ptave_rmpfjet1x.histograms)

        self.h_mu_rpt = VariationHistograms("h",{"Mu":bins_mu,"RpT":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rpt.histograms)
        self.h_mu_rmpf = VariationHistograms("h",{"Mu":bins_mu,"RMPF":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpf.histograms)
        self.h_mu_rmpfjet1 = VariationHistograms("h",{"Mu":bins_mu,"RMPFjet1":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpfjet1.histograms)
        self.h_mu_rmpfjetn = VariationHistograms("h",{"Mu":bins_mu,"RMPFjetn":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpfjetn.histograms)
        self.h_mu_rmpfuncl = VariationHistograms("h",{"Mu":bins_mu,"RMPFuncl":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpfuncl.histograms)
        self.h_mu_rmpfx = VariationHistograms("h",{"Mu":bins_mu,"RMPFx":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpfx.histograms)
        self.h_mu_rmpfjet1x = VariationHistograms("h",{"Mu":bins_mu,"RMPFjet1x":bins_y},variationSet, self.isData)
        self.histograms.update(self.h_mu_rmpfjet1x.histograms)

        #
        self.h_zpt_QGL = VariationHistograms("h",{"ZpT":bins_zpt,"QGL":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_QGL.histograms)
        self.h_zpt_muEF = VariationHistograms("h",{"ZpT":bins_zpt,"muEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_muEF.histograms)
        self.h_zpt_chEmEF = VariationHistograms("h",{"ZpT":bins_zpt,"chEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_chEmEF.histograms)
        self.h_zpt_chHEF = VariationHistograms("h",{"ZpT":bins_zpt,"chHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_chHEF.histograms)
        self.h_zpt_neEmEF = VariationHistograms("h",{"ZpT":bins_zpt,"neEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_neEmEF.histograms)
        self.h_zpt_neHEF = VariationHistograms("h",{"ZpT":bins_zpt,"neHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_zpt_neHEF.histograms)

        self.h_jpt_QGL = VariationHistograms("h",{"JetPt":bins_zpt,"QGL":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_QGL.histograms)
        self.h_jpt_muEF = VariationHistograms("h",{"JetPt":bins_zpt,"muEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_muEF.histograms)
        self.h_jpt_chEmEF = VariationHistograms("h",{"JetPt":bins_zpt,"chEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_chEmEF.histograms)
        self.h_jpt_chHEF = VariationHistograms("h",{"JetPt":bins_zpt,"chHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_chHEF.histograms)
        self.h_jpt_neEmEF = VariationHistograms("h",{"JetPt":bins_zpt,"neEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_neEmEF.histograms)
        self.h_jpt_neHEF = VariationHistograms("h",{"JetPt":bins_zpt,"neHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_jpt_neHEF.histograms)

        self.h_ptave_QGL = VariationHistograms("h",{"PtAve":bins_zpt,"QGL":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_QGL.histograms)
        self.h_ptave_muEF = VariationHistograms("h",{"PtAve":bins_zpt,"muEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_muEF.histograms)
        self.h_ptave_chEmEF = VariationHistograms("h",{"PtAve":bins_zpt,"chEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_chEmEF.histograms)
        self.h_ptave_chHEF = VariationHistograms("h",{"PtAve":bins_zpt,"chHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_chHEF.histograms)
        self.h_ptave_neEmEF = VariationHistograms("h",{"PtAve":bins_zpt,"neEmEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_neEmEF.histograms)
        self.h_ptave_neHEF = VariationHistograms("h",{"PtAve":bins_zpt,"neHEF":bins_01},variationSet, self.isData)
        self.histograms.update(self.h_ptave_neHEF.histograms)
        """
        return self.histograms

    def addHistogram(self, name, nbins, binmin, binmax):
        self.histograms[name] = hist.Hist(
            hist.axis.StrCategory([], growth=True, name=name, label="label"),
            hist.axis.Regular(nbins, binmin, binmax, name="value", label="x value"),
            storage="weight",
            name=name
        )

    def add2DHistogram(self, name, xbins, ybins):
        self.histograms[name] = hist.Hist(
            hist.axis.StrCategory([], growth=True, name=name, label="label"),
            hist.axis.Variable(xbins, name="x", label="x label"),
            hist.axis.Variable(ybins, name="y", label="y label"),
            storage="weight",
            name=name
        )

    def cloneHistogram(self, nameOrig, nameClone):
        self.histograms[nameClone] = self.histograms[nameOrig].copy()

    def fill(self,events,out):

        # variables
        weight     = events.weight
        mu         = events.mu
        MET        = events.METtype1
        leadingJet = events.leadingJet
        Zboson     = events.Zboson


        # Adding a pt-0 jet in cases there are no subleading jets
        events1j = events[(ak.num(events.Jet) == 1)]
        zerojets = events1j.Jet
        zerojets['pt'] = 0
        zerojets['px'] = 0
        zerojets['py'] = 0
        zerojets['phi'] = 0
        jets = ak.concatenate([events1j.Jet,zerojets], axis=1)
        events1j['Jet'] = jets
        eventsNj = events[(ak.num(events.Jet) > 1)]
        events = ak.concatenate([eventsNj,events1j])

        subLeadingJet = events.Jet[:, 1]

        x = ak.sum(events.Jet.pt*np.cos(events.Jet.phi), axis=1)
        y = ak.sum(events.Jet.pt*np.sin(events.Jet.phi), axis=1)
        jets_all = ak.zip({"pt": np.hypot(x, y), "phi": np.arctan2(y, x), "px": x, "py": y})

        x2 = x - leadingJet.px
        y2 = y - leadingJet.py
        jets_notLeadingJet = ak.zip({"pt": np.hypot(x2, y2), "phi": np.arctan2(y2, x2), "px": x2, "py": y2})


        mx = -MET.px -jets_all.px -Zboson.px
        my = -MET.py -jets_all.py -Zboson.py
        METuncl = ak.zip({"pt": np.hypot(mx, my), "phi": np.arctan2(my, mx), "px": mx, "py": my})

        # alpha = 0 if subLeadingJet.pt < 15
        subLeadingJetpt = np.array(subLeadingJet.pt)
        subLeadingJetpt[(subLeadingJetpt < 15)] = 0
        events["alpha"] = ak.flatten(subLeadingJetpt/Zboson.pt)
        events["eta"] = events.leadingJet.eta

        R_pT = ak.flatten(leadingJet.pt/Zboson.pt)
        R_MPF = ak.flatten(1 + MET.pt*(np.cos(MET.phi)*Zboson.px + np.sin(MET.phi)*Zboson.py)/(Zboson.pt*Zboson.pt))
        R_MPFjet1 = ak.flatten(-leadingJet.pt*(np.cos(leadingJet.phi)*Zboson.px + np.sin(leadingJet.phi)*Zboson.py)/(Zboson.pt*Zboson.pt))
        R_MPFjetn = ak.flatten(-jets_notLeadingJet.pt*(np.cos(jets_notLeadingJet.phi)*Zboson.px + np.sin(jets_notLeadingJet.phi)*Zboson.py)/(Zboson.pt*Zboson.pt))
        R_MPFuncl = ak.flatten(-METuncl.pt*(np.cos(METuncl.phi)*Zboson.px + np.sin(METuncl.phi)*Zboson.py)/(Zboson.pt*Zboson.pt))

        pi2 = np.pi/2
        R_MPFx = ak.flatten(1 + MET.pt*(np.cos(MET.phi+pi2)*Zboson.px + np.sin(MET.phi+pi2)*Zboson.py)/(Zboson.pt*Zboson.pt))
        R_MPFjet1x = ak.flatten(-leadingJet.pt*(np.cos(leadingJet.phi+pi2)*Zboson.px + np.sin(leadingJet.phi+pi2)*Zboson.py)/(Zboson.pt*Zboson.pt))


        jpt = np.array(leadingJet.pt)
        zpt = np.array(ak.flatten(Zboson.pt))
        jpt[(jpt < 12) | (zpt < 12)] = 0
        zpt[(jpt < 12) | (zpt < 12)] = 0
        ptave = 0.5*(jpt + zpt)

        #filling
        self.h_zpt_rpt.fill(events,zpt,R_pT,weight)
        """
        self.h_zpt_rmpf.fill(events,out,zpt,R_MPF,weight)
        self.h_zpt_rmpfjet1.fill(events,out,zpt,R_MPFjet1,weight)
        self.h_zpt_rmpfjetn.fill(events,out,zpt,R_MPFjetn,weight)
        self.h_zpt_rmpfuncl.fill(events,out,zpt,R_MPFuncl,weight)
        self.h_zpt_rmpfx.fill(events,out,zpt,R_MPFx,weight)
        self.h_zpt_rmpfjet1x.fill(events,out,zpt,R_MPFjet1x,weight)

        self.h_jpt_rpt.fill(events,out,jpt,R_pT,weight)
        self.h_jpt_rmpf.fill(events,out,jpt,R_MPF,weight)
        self.h_jpt_rmpfjet1.fill(events,out,jpt,R_MPFjet1,weight)
        self.h_jpt_rmpfjetn.fill(events,out,jpt,R_MPFjetn,weight)
        self.h_jpt_rmpfuncl.fill(events,out,jpt,R_MPFuncl,weight)
        self.h_jpt_rmpfx.fill(events,out,jpt,R_MPFx,weight)
        self.h_jpt_rmpfjet1x.fill(events,out,jpt,R_MPFjet1x,weight)

        self.h_ptave_rpt.fill(events,out,ptave,R_pT,weight)
        self.h_ptave_rmpf.fill(events,out,ptave,R_MPF,weight)
        self.h_ptave_rmpfjet1.fill(events,out,ptave,R_MPFjet1,weight)
        self.h_ptave_rmpfjetn.fill(events,out,ptave,R_MPFjetn,weight)
        self.h_ptave_rmpfuncl.fill(events,out,ptave,R_MPFuncl,weight)
        self.h_ptave_rmpfx.fill(events,out,ptave,R_MPFx,weight)
        self.h_ptave_rmpfjet1x.fill(events,out,ptave,R_MPFjet1x,weight)

        self.h_mu_rpt.fill(events,out,mu,R_pT,weight)
        self.h_mu_rmpf.fill(events,out,mu,R_MPF,weight)
        self.h_mu_rmpfjet1.fill(events,out,mu,R_MPFjet1,weight)
        self.h_mu_rmpfjetn.fill(events,out,mu,R_MPFjetn,weight)
        self.h_mu_rmpfuncl.fill(events,out,mu,R_MPFuncl,weight)
        self.h_mu_rmpfx.fill(events,out,mu,R_MPFx,weight)
        self.h_mu_rmpfjet1x.fill(events,out,mu,R_MPFjet1x,weight)

        self.h_zpt_QGL.fill(events,out,zpt,leadingJet.qgl,weight)
        self.h_zpt_muEF.fill(events,out,zpt,leadingJet.muEF,weight)
        self.h_zpt_chEmEF.fill(events,out,zpt,leadingJet.chEmEF,weight)
        self.h_zpt_chHEF.fill(events,out,zpt,leadingJet.chHEF,weight)
        self.h_zpt_neEmEF.fill(events,out,zpt,leadingJet.neEmEF,weight)
        self.h_zpt_neHEF.fill(events,out,zpt,leadingJet.neHEF,weight)

        self.h_jpt_QGL.fill(events,out,jpt,leadingJet.qgl,weight)
        self.h_jpt_muEF.fill(events,out,jpt,leadingJet.muEF,weight)
        self.h_jpt_chEmEF.fill(events,out,jpt,leadingJet.chEmEF,weight)
        self.h_jpt_chHEF.fill(events,out,jpt,leadingJet.chHEF,weight)
        self.h_jpt_neEmEF.fill(events,out,jpt,leadingJet.neEmEF,weight)
        self.h_jpt_neHEF.fill(events,out,jpt,leadingJet.neHEF,weight)

        self.h_ptave_QGL.fill(events,out,ptave,leadingJet.qgl,weight)
        self.h_ptave_muEF.fill(events,out,ptave,leadingJet.muEF,weight)
        self.h_ptave_chEmEF.fill(events,out,ptave,leadingJet.chEmEF,weight)
        self.h_ptave_chHEF.fill(events,out,ptave,leadingJet.chHEF,weight)
        self.h_ptave_neEmEF.fill(events,out,ptave,leadingJet.neEmEF,weight)
        self.h_ptave_neHEF.fill(events,out,ptave,leadingJet.neHEF,weight)

        """
        self.histograms['Z_mass'].fill('Z_mass',value=ak.flatten(Zboson.mass),weight=weight)
        return self.histograms

class VariationHistograms():
    def __init__(self,name,xybins,var,isData):
        self.histograms = {}
        self.isData = isData
        self.selection = {}
        self.weights = {}

        variations = []
        varSelections = {}

        for key in var.keys():
            variations.append(list(var[key].keys()))
            for k2 in var[key].keys():
                varSelections[k2] = var[key][k2]

        nameBase = name
        xKey = ''
        yKey = ''
        for key in xybins.keys():
            nameBase = nameBase + '_' + key
            if xKey == '':
                xKey = key
            else:
                yKey = key
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
            self.book(hname,xKey,xybins[xKey],yKey, xybins[yKey])

        self.selection_re = re.compile("(?P<variable>\w+)\s*(?P<operator>\S+)\s*(?P<value>\S+)\)")

    def book(self,name,xname,xbins,yname,ybins):
        self.histograms[name] = hist.Hist(
            hist.axis.StrCategory([], growth=True, name=name, label="label"),
            hist.axis.Variable(xbins, name="x", label=xname),
            hist.axis.Variable(ybins, name="y", label=yname),
            storage="weight",
            name="Counts"
        )

    def fill(self,events,x,y,weight):
        for key in self.histograms:
            sele = self.select(events,key)
            w = self.getweight(events,key)
            selected_x = x[sele]
            selected_y = y[sele]
            selected_w = weight[sele]*w[sele]
            self.histograms[key].fill(key,
                                      x=selected_x,
                                      y=selected_y,
                                      weight=selected_w)

        return self.histograms

    def getweight(self,events,variable):
        w = np.array([1.0]*len(events))
        if variable in self.weights.keys():
            if 'PSWeight' in self.weights[variable] and len(self.weights[variable]) > 8:
                weightnumber = int(self.weights[variable].replace('PSWeight',''))
                w = events['PSWeight'][:,weightnumber]
        return w

    def select(self,events,variable):
        return_sele = (events["event"] > 0)
        if self.selection[variable] == None:
            return return_sele

        selections = self.selection[variable].split('&')

        for s in selections:
            match = self.selection_re.search(s)
            if match:
                variable = match.group('variable')
                operator = match.group('operator')
                value    = eval(match.group('value'))

                if operator == '<':
                    return_sele = return_sele & (events[variable] < value)
                if operator == '>':
                    return_sele = return_sele & (events[variable] > value)
                if operator == '==':
                    return_sele = return_sele & (events[variable] == value)
        return return_sele

