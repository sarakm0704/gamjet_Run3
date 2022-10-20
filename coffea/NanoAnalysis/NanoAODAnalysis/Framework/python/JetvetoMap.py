import os

import awkward as ak
from coffea import lookup_tools

from DataPath import getDataPath

class JetvetoMap():
    def __init__(self,year,isData):
        filename = ""
        histo = ""
        if '2016' in year:
            filename = "Summer19UL16_V0/hotjets-UL16.root"
            if isData:
                histo = "h2hot_ul16_plus_hbm2_hbp12_qie11"
            else:
                histo = "h2hot_mc"
        if '2017' in year:
            filename = "Summer19UL17_V2/hotjets-UL17_v2.root"
            if isData:
                histo = "h2hot_ul17_plus_hep17_plus_hbpw89"
            #else:
            #    histo = "h2hot_mc"
        if '2018' in year:
            filename = "Summer19UL18_V1/hotjets-UL18.root"
            if isData:
                histo = "h2hot_ul18_plus_hem1516_and_hbp2m1"
            #else:
            #    histo = "h2hot_mc"

        self.useMap = False
        if len(histo) > 0:
            self.useMap = True

        if self.useMap:
            datapath = getDataPath()
            ext = lookup_tools.extractor()
            ext.add_weight_sets(["vetomap %s %s"%(histo,os.path.join(datapath,'JECDatabase','jet_veto_maps',filename))])
            ext.finalize()

            self.evaluator = ext.make_evaluator()
        
    def passmap(self,jet):
        if not self.useMap:
            return [True]*len(jet)

        values = self.evaluator["vetomap"](jet.eta, jet.phi)
        return (values <= 0)
