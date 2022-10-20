from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor
import cachetools
import awkward as ak
import numpy as np
import os

from DataPath import getDataPath


class JEC():
    def getCorrectionVersion(self,run,isData):
        self.isData = isData
        corr = None
        if '2016' in run:
            if isData:
                runletter = run[4]
                if runletter in 'BCD':
                    corr = 'Summer19UL16APV_RunBCD_V7_DATA'
                if runletter in 'EF' and 'APV' in run:
                    corr = 'Summer19UL16APV_RunEF_V7_DATA/'
                if runletter in 'FGH' and not 'APV' in run:
                    corr = 'Summer19UL16_RunFGH_V7_DATA'
            else:
                corr = 'Summer19UL16_V7_MC'
        if '2017' in run:
            if isData:
                runletter = run[4:]
                corr = 'Summer19UL17_Run%s_V6_DATA'%runletter
            else:
                corr = 'Summer19UL17_V6_MC'
        if '2018' in run:
            if isData:
                runletter = run[4:]
                corr = 'Summer19UL18_Run%s_V5_DATA'%runletter
            else:
                corr = 'Summer19UL18_V5_MC'

        return corr

    def getJERVersion(self,run,isData):
        self.isData = isData
        corr = None
        if '2016' in run and 'APV' in run:
            if isData:
                corr = 'Summer20UL16APV_JRV3_DATA'
            else:
                corr = 'Summer20UL16APV_JRV3_MC'
        if '2016' in run and not 'APV' in run:
            if isData:
                corr = 'Summer20UL16_JRV3_DATA'
            else:
                corr = 'Summer20UL16_JRV3_MC'
        if '2017' in run:
            if isData:
                corr = 'Summer19UL17_JRV3_DATA'
            else:
                corr = 'Summer19UL17_JRV3_MC'
        if '2018' in run:
            if isData:
                corr = 'Summer19UL18_JRV2_DATA'
            else:
                corr = 'Summer19UL18_JRV2_MC'
        return corr

    def __init__(self,run,isData):
        # JEC
        CORRECTION_VERSION = self.getCorrectionVersion(run,isData)
        if CORRECTION_VERSION == None:
            print("Not using JEC")
            self.noCorrections = True
        else:
            print("Using JEC",CORRECTION_VERSION)
            self.noCorrections = False

            CORRECTION_SET = ['L1FastJet_AK4PFchs','L2Relative_AK4PFchs','L3Absolute_AK4PFchs','L2L3Residual_AK4PFchs']

            datapath = getDataPath()

            weight_sets = ['* * %s'%(os.path.join(datapath,'JECDatabase/textFiles',CORRECTION_VERSION,CORRECTION_VERSION+'_'+correctionset+'.txt')) for correctionset in CORRECTION_SET]

            jec_stack_names = [CORRECTION_VERSION+'_'+correctionset for correctionset in CORRECTION_SET]

            ext = extractor()
            ext.add_weight_sets(weight_sets)
            ext.finalize()

            evaluator = ext.make_evaluator()

            jec_inputs = {name: evaluator[name] for name in jec_stack_names}
            jec_stack = JECStack(jec_inputs)

            name_map = jec_stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'rho'
            name_map['ptGenJet'] = 'pt_genjet'

            self.jet_factory = CorrectedJetsFactory(name_map, jec_stack)

            # JEC L1RC only
            CORRECTION_SET_L1 = ['L1RC_AK4PFchs']

            weight_sets_l1 = ['* * %s'%(os.path.join(datapath,'JECDatabase/textFiles',CORRECTION_VERSION,CORRECTION_VERSION+'_'+correctionset+'.txt')) for correctionset in CORRECTION_SET_L1]
            jec_stack_names_l1 = [CORRECTION_VERSION+'_'+correctionset for correctionset in CORRECTION_SET_L1]

            ext_l1 = extractor()
            ext_l1.add_weight_sets(weight_sets_l1)
            ext_l1.finalize()

            evaluator_l1 = ext_l1.make_evaluator()

            jec_inputs_l1 = {name: evaluator_l1[name] for name in jec_stack_names_l1}
            jec_stack_l1 = JECStack(jec_inputs_l1)

            name_map_l1 = jec_stack_l1.blank_name_map
            name_map_l1['JetPt'] = 'pt'
            name_map_l1['JetMass'] = 'mass'
            name_map_l1['JetEta'] = 'eta'
            name_map_l1['JetA'] = 'area'
            name_map_l1['ptRaw'] = 'pt_raw'
            name_map_l1['massRaw'] = 'mass_raw'
            name_map_l1['Rho'] = 'rho'
            name_map_l1['ptGenJet'] = 'pt_genjet'

            self.jet_factory_l1 = CorrectedJetsFactory(name_map_l1, jec_stack_l1)

            # MET for new JEC
            name_map["METpt"] = "pt"
            name_map["METphi"] = "phi"
            name_map["JetPhi"] = "phi"
            name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
            name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

            self.met_factory = CorrectedMETFactory(name_map)

            # JER, JER Scale Factor
            CORRECTION_SET_JER = ['PtResolution_AK4PFchs','SF_AK4PFchs']
            JER_VERSION = self.getJERVersion(run,isData)
            JERweight_sets = ['* * %s'%(os.path.join(datapath,'JRDatabase/textFiles',JER_VERSION,JER_VERSION+'_'+correctionset+'.txt')) for correctionset in CORRECTION_SET_JER]
            jer_stack_names = [JER_VERSION+'_'+correctionset for correctionset in CORRECTION_SET_JER]

            jerext = extractor()
            jerext.add_weight_sets(JERweight_sets)
            jerext.finalize()

            jerevaluator = jerext.make_evaluator()

            self.reso = JetResolution(**{name: jerevaluator[name] for name in jer_stack_names[0:1]})
            self.resosf = JetResolutionScaleFactor(**{name: jerevaluator[name] for name in jer_stack_names[1:2]})

    def apply(self,events):
        if self.noCorrections:
            return events.Jet

        jets = events.Jet
        jets['pt_orig'] = jets['pt']
        jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
        jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
        jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]

        if not self.isData:
            jets['pt_genjet'] = events.GenJet[jets.genJetIdx].pt

        events_cache = events.caches[0]

        corrected_jets_array = self.jet_factory.build(jets, lazy_cache=events_cache)
        corrected_jets = jets
        corrected_jets['pt'] = corrected_jets_array['pt']
        corrected_jets['mass'] = corrected_jets_array['mass']
        # pt-order the corrected jets
        corrected_jets = corrected_jets[ak.argsort(corrected_jets.pt, axis=1, ascending=False)]

#        if not self.isData:
#            smear = self.smearing(events)
#            corrected_jets["pt"] = corrected_jets.pt*smear
#            corrected_jets["mass"] = corrected_jets.mass*smear

        corrected_jets_l1_array = self.jet_factory_l1.build(jets, lazy_cache=events_cache)
        self.corrected_jets_l1 = jets
        self.corrected_jets_l1['pt'] = corrected_jets_l1_array['pt']
        self.corrected_jets_l1['mass'] = corrected_jets_l1_array['mass']
        self.corrected_jets_l1 = self.corrected_jets_l1[ak.argsort(self.corrected_jets_l1.pt, axis=1, ascending=False)] # pt-order

        return corrected_jets

    def recalculateMET(self,events):
        if self.noCorrections:
            return events.MET
        met = events.MET
        corrected_jets = events.Jet
        jec_cache = cachetools.Cache(np.inf)
        corrected_met = self.met_factory.build(met, corrected_jets, lazy_cache=jec_cache)
        return corrected_met

    def recalculateMET2(self,events):
        if self.noCorrections:
            return events.MET
        met             = events.MET
        jet_pt_corr     = events.Jet.pt
        jet_phi         = events.Jet.phi
        jet_pt_orig     = events.Jet.pt_orig
        jet_pt_raw      = events.Jet.pt_raw
        jet_pt_corrL1rc = events.Jet_L1RC.pt

        sj, cj = np.sin(jet_phi), np.cos(jet_phi)
#        x = met_pt * np.cos(met_phi) + ak.sum(jet_pt * cj - jet_pt_orig * cj, axis=1)
#        y = met_pt * np.sin(met_phi) + ak.sum(jet_pt * sj - jet_pt_orig * sj, axis=1)
        x = met.pt * np.cos(met.phi) + ak.sum(jet_pt_orig * cj - jet_pt_corr * cj - jet_pt_raw * cj + jet_pt_corrL1rc * cj, axis=1)
        y = met.pt * np.sin(met.phi) + ak.sum(jet_pt_orig * sj - jet_pt_corr * sj - jet_pt_raw * sj + jet_pt_corrL1rc * sj, axis=1)
        return ak.zip({"pt": np.hypot(x, y), "phi": np.arctan2(y, x), "px": x, "py": y})

    # MET = −sum(jets) -sum(uncl)
    # MET_type1 = −sum(jets_corrected) -sum(uncl) = −sum(jets_corrected) + MET +sum(jets)
    # met += jet_orig - jet_corr - (jet_raw - jet_corrL1rc);

    def smearing(self,events): # not needed, smearing done already by the POG code
        jets = events.Jet
        jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
        jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]

        scalar_form = ak.without_parameters(jets["pt_raw"]).layout.form
        jec_cache = cachetools.Cache(np.inf)
        jetResolution = ak.flatten(self.reso.getResolution(
            JetEta=jets["eta"],
            Rho=jets["rho"],
            JetPt=jets["pt_raw"],
            form=scalar_form,
            lazy_cache=jec_cache,
        ))
        eta = ak.flatten(jets["eta"])

        jetResolutionSF = self.resosf.getScaleFactor(JetEta=eta)
        # getScaleFactor output format central-up-down, taking the central value
        jetResolutionSF = jetResolutionSF[:, 0]

        #KIT: jecSmearFactor = 1 + std::normal_distribution<>(0, jetResolution)(m_randomNumberGenerator) * std::sqrt(std::max(jetResolutionSF * jetResolutionSF - 1, 0.0));
        #KIT: // apply factor (prevent negative values)
        #KIT: recoJets[iJet]->p4 *= (jecSmearFactor < 0) ? 0.0 : jecSmearFactor;

        jecSmearFactor = np.random.normal(1.0,jetResolution)*np.sqrt(np.maximum(jetResolutionSF*jetResolutionSF-1,0))
        jecSmearFactor = ak.unflatten(jecSmearFactor,ak.num(events.Jet.eta))
        return jecSmearFactor

