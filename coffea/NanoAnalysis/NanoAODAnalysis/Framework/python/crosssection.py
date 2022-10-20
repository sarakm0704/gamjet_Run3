
class CrossSection:
    '''
    Cross section of a single process (physical dataset)
    
    Constructor
    
    \parma name              Name of the process
    \param energyDictionary  Dictionary of energy -> cross section (energy as string in TeV, cross section as float in pb)
    '''
    def __init__(self, name, energyDictionary):
        self.name = name
        for key, value in energyDictionary.items():
            setattr(self, key, value)

    def get(self, energy):
        '''
        Get cross section
        
        \param energy  Energy as string in TeV
        '''
        try:
            return getattr(self, energy)
        except AttributeError:
            raise Exception("No cross section set for process %s for energy %s" % (self.name, energy))



class CrossSectionList:
    '''
    List of CrossSection objects
    '''
    def __init__(self, *args):
        self.crossSections = args[:]

    def crossSection(self, name, energy):
        for obj in self.crossSections:
            #if name[:len(obj.name)] == obj.name:
            if name == obj.name:
                return obj.get(energy)
        return None

backgroundCrossSections = CrossSectionList(
    CrossSection("QCD_Pt_20toInf_MuEnrichedPt15", {
            "13": 720648000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_15to20_EMEnriched", {
            "13": 1279000000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_20to30_EMEnriched", {
            "13": 557600000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_30to50_EMEnriched", {
            "13": 136000000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_50to80_EMEnriched", {
            "13": 19800000.0, # [27] LO
            }),
    CrossSection("QCD_Pt_80to120_EMEnriched", {
            "13": 2800000.0, # [27] LO
            }),
    CrossSection("QCD_Pt_120to170_EMEnriched", {
            "13": 477000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_170to300_EMEnriched", {
            "13": 114000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_300toInf_EMEnriched", {
            "13": 9000.0, # [27] LO 
            }),
    CrossSection("QCD_Pt_15to30", {
            "13": 2237000000., # [12]
            }),
    CrossSection("QCD_Pt_30to50", {
            "7": 5.312e+07, # [2]
            "8": 6.6285328e7, # [1]
            "13": 161500000., # [12]
            }),
    CrossSection("QCD_Pt_50to80", {
            "7": 6.359e+06, # [2]
            "8": 8148778.0, # [1]
            "13": 22110000., # [12]
            }),
    CrossSection("QCD_Pt_80to120", {
            "7": 7.843e+05, # [2]
            "8": 1033680.0, # [1]
            "13": 3000114.3, # [12]
            }),
    CrossSection("QCD_Pt_120to170", {
            "7": 1.151e+05, # [2]
            "8": 156293.3, # [1]
            "13": 493200., # [12] # McM: 471100
            }),
    CrossSection("QCD_Pt_170to300", {
            "7": 2.426e+04, # [2]
            "8": 34138.15, # [1]
            "13": 120300., # [12]
            }),
    CrossSection("QCD_Pt_300to470", {
            "7": 1.168e+03, # [2]
            "8": 1759.549, # [1]
            "13": 7475., # [12]
            }),
    CrossSection("QCD_Pt_470to600", {
            "13": 587.1, # [12]
            }),
    CrossSection("QCD_Pt_600to800", {
            "13": 167., # [12]
            }),
    CrossSection("QCD_Pt_800to1000", {
            "13": 28.25, # [12]
            }),
    CrossSection("QCD_Pt_1000to1400", {
            "13": 8.195, # [12]
            }),
    CrossSection("QCD_Pt_1400to1800", {
            "13": 0.7346, # [12] # McM: 0.84265
            }),
    CrossSection("QCD_Pt_1800to2400", {
            "13": 0.1091, # [12] # McM: 0.114943
            }),
    CrossSection("QCD_Pt_2400to3200", {
            "13": 0.00682981, # [15]
            }),
    CrossSection("QCD_Pt_3200toInf", {
            "13": 0.000165445 , # [15]
            }),
    CrossSection("QCD_Pt20_MuEnriched", {
            "7": 296600000.*0.0002855, # [2]
            "8": 3.64e8*3.7e-4, # [1]
            }),
    CrossSection("QCD_Pt_15to20_MuEnrichedPt5", {
            "13": 3.625e+06, # 3.625e+06 +- 1.780e+03 [14]
    }),
    CrossSection("QCD_Pt_20to30_MuEnrichedPt5", {
            "13": 3.153e+06, # 3.153e+06 +- 5.608e+02 [14]
    }),
    CrossSection("QCD_Pt_30to50_MuEnrichedPt5", {
            "13": 1.652e+06, # 1.652e+06 +- 3.005e+02 [14]
    }),
    CrossSection("QCD_Pt_50to80_MuEnrichedPt5", {
            "13": 4.488e+05, # 4.488e+05 +- 9.995e+01 [14]
    }),
    CrossSection("QCD_Pt_80to120_MuEnrichedPt5", {
            "13": 1.052e+05, # 1.052e+05 +- 2.136e+01 [14]
    }),
    CrossSection("QCD_Pt_120to170_MuEnrichedPt5", {
            "13": 2.549e+04, # 2.549e+04 +- 8.800e+00 [14]
    }),
    CrossSection("QCD_Pt_170to300_MuEnrichedPt5", {
            "13": 8.639e+03, # 8.639e+03 +- 2.015e+00 [14]
    }),
    CrossSection("QCD_Pt_300to470_MuEnrichedPt5", {
            "13": 7.961e+02, # 7.961e+02 +- 1.092e-01 [14]
    }),
    CrossSection("QCD_Pt_470to600_MuEnrichedPt5", {
            "13": 7.920e+01, # 7.920e+01 +- 1.712e-02 [14]
    }),
    CrossSection("QCD_Pt_600to800_MuEnrichedPt5", {
            "13": 2.525e+01, # 2.525e+01 +- 7.557e-03 [14]
    }),
    CrossSection("QCD_Pt_800to1000_MuEnrichedPt5", {
            "13": 4.724e+00, # 4.724e+00 +- 1.000e-03 [14]
    }),
    CrossSection("QCD_Pt_1000toInf_MuEnrichedPt5", {
            "13": 1.619e+00, # 1.619e+00 +- 7.617e-04
    }),
    CrossSection("SingleNeutrino", {
            "13": 1.0, # Unknown. Dummy value.                                                                                                                                           
    }),
    CrossSection("VVTo2L2Nu", {
            "13": 11.95, # [25]
            }),
    CrossSection("WW", {
            "7" :  43.0, # [3]
            "8" :  54.838, # [9], took value for CTEQ PDF since CTEQ6L1 was used in pythia simulation
            "13": 118.7, # [13] from Andrea: WW -> lnqq : 52pb + WW -> lnln : 12.46pb
            }),
    CrossSection("WWToLNuQQ", {
            "13": 49.997, #[17] 
            }),
    CrossSection("WWTo2L2Nu", {
            "13": 12.178, #[17]
            }),
    CrossSection("WWTo4Q", {
            "13": 51.723 , #[17]
            }),
    CrossSection("WWTo4Q_NNPDF31", {
            "13": 51.723, # [17]
            }),
    CrossSection("WWToLNuQQ", {
            "13": 49.997, #[17] 
            }),
    CrossSection("WWTo2L2Nu", {
            "13": 12.178, #[17]
            }),
    CrossSection("WZTo1L1Nu2Q", {
            "13": 10.71, # [25] 
            }),
    CrossSection("WZTo1L3Nu", {
            "13": 3.033e+00, # [25] 
            }),
    CrossSection("WZTo2L2Q", {
            "13": 5.595, # [25] 
            }),
    CrossSection("WZ", {
            "7": 18.2, # [3]
            "8": 33.21, # [9], took value for CTEQ PDF since CTEQ6L1 was used in pythia simulation
            #"13": 29.8 + 18.6, # [13] W+ Z/a* + W- Z/a*, MCFM 6.6 m(l+l-) > 40 GeV
            #"13": 28.55 + 18.19, # [17]
            "13": 47.13, # [17] 
            }),
    CrossSection("ZGToLLG_01J_5f_lowMLL_lowGPt", {
            "13": 172.8, #pb [25] 
            }),
    CrossSection("ZZTo2L2Q_pythia8", {
            "13": 3.22, # [25] 
            }),
    CrossSection("ZZTo4L_pythia8", {
            "13": 1.256, # [26] 
            }),
    CrossSection("ZZ", {
            "7": 5.9, # [3]
            "8": 17.654, # [9], took value for CTEQ PDF since CTEQ6L1 was used in pythia simulation, this is slightly questionmark, since the computed value is for m(ll) > 12
            #"13": 15.4, # [13]
            "13": 16.523, # [17] 
            }),
    CrossSection("TTJets_FullLept", {
            "8": 245.8* 26.1975/249.50, # [10], BR from [11]
            }),
    CrossSection("TTJets_SemiLept", {
            "8": 245.8* 109.281/249.50, # [10], BR from [11]
            }),
    CrossSection("TTJets_Hadronic", {
            "8": 245.8* 114.0215/249.50, # [10], BR from [11]
            }),
    CrossSection("TTJets", {            
            "7": 172.0, # [10]
            "8": 245.8, # [10]
            "13": 831.76, # [18], same as TT because apparently TTJets is also an inclusive sample         
#            "13": 6.639e+02, #6.639e+02 +- 8.237e+00 pb [16] (inputFiles="001AFDCE-C33B-E611-B032-0025905D1C54.root")            
            }),
    CrossSection("TT", {
            "7": 172.0, # [10]
            "8": 245.8, # [10]
            "13": 831.76, # [18]
            }),
    CrossSection("TTTo2L2Nu", {
           "13": 831.76*0.3259*0.3259, # 88.34 pb [29] (XSection with GenXsecAnalyzer must be multiplied with BR since POWHEG doesn't include decay branch ratios!)
            }),
    CrossSection("TTToHadronic", {
            "13": 831.76*0.6741*0.6741, # 377.96 pb [29] (XSection with GenXsecAnalyzer must be multiplied with BR since POWHEG doesn't include decay branch ratios!)
            }),
    CrossSection("TTToSemiLeptonic", {
            "13": 831.76*2*0.6741*0.3259, # 365.45 pb [29] (XSection with GenXsecAnalyzer must be multiplied with BR since POWHEG doesn't include decay branch ratios!)
            }),
    CrossSection("TT_Mtt_0to700", {
            "13": 831.76, # [18], same sas TT inclusive (because this sample is in fact the inclusive sample skimmed to Mtt<700)
            }),
    CrossSection("TT_Mtt_700to1000", {
            "13": 76.60518, # [18],[20], calculated as 75.15 (from generation) /815.96 (from generation) * 831.76 (NNLO)
            }),
    CrossSection("TT_Mtt_1000toInf", {
            "13": 20.57789, # [18],[20], calculated as 20.187 (from generation) /815.96 (from generation) * 831.76 (NNLO)
            }),
    CrossSection("TTGJets", {            
            "13": 1.0, # FIXME (Marina)
            }),
    CrossSection("TT_fsrdown", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_fsrup", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_isrdown", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_isrup", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_hdampDOWN", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_hdampUP", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1665", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1695", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1715", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1735", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1755", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_mtop1785", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx0p2", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx0p5", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx0p8", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx2", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx4", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_widthx8", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_evtgen", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_erdON", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_TuneEE5C", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_TuneCUETP8M2T4up", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_TuneCUETP8M2T4down", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_GluonMoveCRTune", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TT_QCDbasedCRTune_erdON", {
            "13": 831.76, # [18] 
            }),
    CrossSection("TTJets_HT600to800", {
            "13": 0.0, 
            }),
    CrossSection("TTJets_HT800to1200", {
            "13": 0.0, 
            }),
    CrossSection("TTJets_HT1200to2500", {
            "13": 0.0, 
            }),
    CrossSection("TTJets_HT2500toInf", {
            "13": 0.0, 
            }),
    #CrossSection("WJets", {
            #"7": 31314.0, # [2], NNLO
            #"8": 36703.2, # [9], NNLO
            #}),
    CrossSection("WJetsToQQ_HT400to600_qc19_3j", {
            "13": 313.1, # [24]
            }),
    CrossSection("WJetsToQQ_HT600to800_qc19_3j", {
            "13": 68.54, # [24]
            }),
    CrossSection("WJetsToQQ_HT_800toInf_qc19_3j", {
            "13": 34.0,  # [24]
            }),
    CrossSection("WJetsToLNu", {
            "13": 20508.9*3, # [13,17] 20508.9*3, McM for the MLM dataset: 5.069e4
	    }),
    CrossSection("WJetsToLNu_HT_0To70", {
            "13": 20508.9*3, # set to inclusive xsect as HT_0To70 is skimmed from the inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_70To100", {
            "13": 1.353e+03*1.2138, # [14] times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_100To200", {
            "13": 1.293e+03*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_200To400", {
            "13": 3.86e+02*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_400To600", {
            "13": 47.9*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_600ToInf", {
            "13": 0.0, # Forcing to zero to avoid overlap
            }),
    CrossSection("WJetsToLNu_HT_600To800", {
            "13": 12.8*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_800To1200", {
            "13": 5.26*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_1200To2500", {
            "13": 1.33*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("WJetsToLNu_HT_2500ToInf", {
            "13": 3.089e-02*1.2138, # McM times NNLO/LO ratio of inclusive sample
            }),
    # PREP (LO) cross sections, for W+NJets weighting
    CrossSection("PREP_WJets", {
            "7": 27770.0,
            "8": 30400.0,
            }),
    CrossSection("PREP_W1Jets", {
            "7": 4480.0,
            "8": 5400.0,
            }),
    CrossSection("PREP_W2Jets", {
            "7": 1435.0,
            "8": 1750.0,
            }),
    CrossSection("PREP_W3Jets", {
            "7": 304.2,
            "8": 519.0,
            }),
    CrossSection("PREP_W4Jets", {
            "7": 172.6,
            "8": 214.0,
            }),
    # end W+Njets 
    CrossSection("DYJetsToLL_M_50", {
            "7": 3048.0, # [4], NNLO
            "8": 3531.9, # [9], NNLO
            "13": 1921.8*3.0 # [14], NNLO
            }),
    CrossSection("DYJetsToLL_M_50_HERWIGPP", {
            "7": 3048.0, # [4], NNLO
            "8": 3531.9, # [9], NNLO
            "13": 1921.8*3.0 # [14]
            }),
    CrossSection("DYJetsToLL_M_50_TauHLT", {
            "7": 3048.0, # [4], NNLO
            "8": 3531.9, # [9], NNLO
            "13": 1921.8*3.0  # [14]
            }),
    CrossSection("DYJetsToLL_M_50_amcatnloFXFX", {
            "7": 3048.0, # [4], NNLO
            "8": 3531.9, # [9], NNLO
            "13": 1921.8*3.0 # [14], NNLO
            }),
    CrossSection("DYJetsToLL_M_50_madgraphMLM", {
            "7": 3048.0, # [4], NNLO
            "8": 3531.9, # [9], NNLO
            "13": 1921.8*3.0 # [14], NNLO
            }),
    CrossSection("DYJetsToLL_M_10to50", {
            "7": 9611.0, # [1]
            "8": 11050.0, # [1]
            "13" :3205.6*3.0, # [14]
            }),
    CrossSection("DYJetsToLL_M_50_HT_70to100", {
            "13": 209.592, # [21]
            }),
    CrossSection("DYJetsToLL_M_50_HT_100to200", {
            "13": 181.302, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("DYJetsToLL_M_50_HT_200to400", {
            "13": 50.4177, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("DYJetsToLL_M_50_HT_400to600", {
            "13": 6.98314, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("DYJetsToLL_M_50_HT_600toInf", {
            "13": 2.21*1.231, # McM times NNLO/LO ratio of inclusive sample
            }),
    CrossSection("DYJetsToLL_M_50_HT_600to800", {
            "13": 1.6841, # [21]
            }),
    CrossSection("DYJetsToLL_M_50_HT_800to1200", {
            "13": 0.775392, # [21]
            }),
    CrossSection("DYJetsToLL_M_50_HT_1200to2500", {
            "13": 0.18622, # [21]
            }),
    CrossSection("DYJetsToLL_M_50_HT_2500toInf", {
            "13": 0.004384, # [21]
            }),
    CrossSection("DYJetsToLL_M_100to200", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_200to400", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_400to500", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_500to700", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_700to800", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_800to1000", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_1000to1500", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_1500to2000", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DYJetsToLL_M_2000to3000", {
            "13": 0.0, # FIXME
            }),
    CrossSection("DY2JetsToLL_M_50", {
            "13": 3.345e+02, # [14]
            }),
    CrossSection("DY3JetsToLL_M_50", {
            "13": 1.022e+02, # [14]
            }),
    CrossSection("DY4JetsToLL_M_50", {
            "13": 5.446e+01, # [14]
            }),
    CrossSection("DYToTauTau_M_20_", {
            "7": 4998, # [4], NNLO
            "8": 5745.25, # [9], NNLO
            }),
    CrossSection("DYToTauTau_M_100to200", {
            "7": 0, # []
            "8": 34.92, # [1]
	    "13": 2.307e+02, # [14]
            }),
    CrossSection("DYToTauTau_M_200to400", {
            "7": 0, # []      
            "8": 1.181, # [1]
            "13": 7.839e+00, # [14]
            }),
   CrossSection("DYToTauTau_M_400to500", {
            "13": 3.957e-01, # [14]
            }),
   CrossSection("DYToTauTau_M_500to700", {
            "13": 2.352e-01, # [14]
            }),
   CrossSection("DYToTauTau_M_700to800", {
            "13": 3.957e-02, # [14]
            }),
    CrossSection("DYToTauTau_M_400to800", {
            "7": 0, # []      
            "8": 0.08699, # [1]
            }),
    CrossSection("DYToTauTau_M_800", {
            "7": 0, # []      
            "8": 0.004527, # [1]
            }),
    CrossSection("DYJetsToQQ_HT180", {
            "13": 1.209e+03, # 1.209e+03 +- 1.302e+00 pb [16]
            }),
    CrossSection("ZprimeToTauTau_M_500", {
            "13": 5.739,
            }),
    CrossSection("ZprimeToTauTau_M_1000", {
            "13": 3.862e-01,
            }),
    CrossSection("ZprimeToTauTau_M_3000", {
            "13": 1.585e-03,
            }),
    CrossSection("GluGluHToTauTau_M125", {
            "13": 1, # dummy value, not really needed as this sample is not merged with anything else
            }),
    CrossSection("GluGluHToTauTau_M125_TauHLT", {
            "13": 1, # dummy value, not really needed as this sample is not merged with anything else
            }),
    CrossSection("T_t-channel", {
            "7": 41.92, # [5,6]
            "8": 56.4, # [8]
            }),
    CrossSection("Tbar_t-channel", {
            "7": 22.65, # [5,6]
            "8": 30.7, # [8]
            }),
    CrossSection("T_tW-channel", {
            "7": 7.87, # [5,6]
            "8": 11.1, # [8]
            }),
    CrossSection("Tbar_tW-channel", {
            "7": 7.87, # [5,6]
            "8": 11.1, # [8]
            }),
    CrossSection("T_s-channel", {
            "7": 3.19, # [5,6]
            "8": 3.79, # [8]
            }),
    CrossSection("Tbar_s-channel", {
            "7": 1.44, # [5,6]
            "8": 1.76, # [8]
            }),
    CrossSection("ST_tW_antitop_5f_inclusiveDecays", {
            "13": 35.85, # [19]
            }),
    CrossSection("ST_tW_antitop_5f_inclusiveDecays_TuneCUETP8M1", {
            "13": 35.85, # [19]
            }),
    CrossSection("ST_tW_antitop_5f_inclusiveDecays_TuneCUETP8M2T4", {
            "13": 35.85, # [19]
            }),
    CrossSection("ST_tW_top_5f_inclusiveDecays", {
            "13": 35.85, # [19]
            }),
    CrossSection("ST_t_channel_antitop_4f_inclusiveDecays", {
            "13": 80.95, # [19]
            "14": 93.28, # [19]
            }),
    CrossSection("ST_t_channel_antitop_4f_InclusiveDecays", {
            "13": 80.95, # [19]
            "14": 93.28, # [19]
            }),
    CrossSection("ST_t_channel_antitop_5f", {
            "13": 80.95, # [19, 23]
            "14": 93.28, # [19, 23]
            }),
    CrossSection("ST_t_channel_antitop_5f_InclusiveDecays", {
            "13": 80.95, # [19, 23]
            "14": 93.28, # [19, 23]
            }),
    CrossSection("ST_t_channel_top_4f_inclusiveDecays", {
            "13": 136.02, # [19, 23]
            "14": 154.76, # [19, 23]
            }),
    CrossSection("ST_t_channel_top_4f_InclusiveDecays", {
            "13": 136.02, # [19, 23]
            "14": 154.76, # [19, 23]
            }),
    CrossSection("ST_t_channel_top_5f", {
            "13": 136.02, # [19, 23]
            "14": 154.76, # [19, 23]
            }),
    CrossSection("ST_t_channel_top_5f_InclusiveDecays", {
            "13": 136.02, # [19, 23]
            "14": 154.76, # [19, 23]
            }),
    CrossSection("ST_s_channel_4f_inclusiveDecays", {
            "13": 11.36, # [19] (ref. [23] slightly different 10.32)
            "14": 11.86, # [19] (ref. [23] slightly different 11.39)
            }),
    CrossSection("ST_s_channel_4f_InclusiveDecays", {
            "13": 11.36, # [19] (ref. [23] slightly different 10.32)
            "14": 11.86, # [19] (ref. [23] slightly different 11.39)
            }),
    CrossSection("ST_s_channel_4f_hadronicDecays", {
            "13": 11.36*0.676, # [19]
            }),
    CrossSection("ST_s_channel_4f_leptonDecays", {
            "13": 3.36, # [28]
            }),
    CrossSection("QCD_bEnriched_HT100to200", {
            "13": 1.318e+06, # 1.318e+06 +- 6.249e+03 pb [16] (only 1 input file used)
            }),
    CrossSection("QCD_bEnriched_HT200to300", {
            "13": 8.823e+04, # 8.823e+04 +- 3.818e+01 pb [16]
            }),
    CrossSection("QCD_bEnriched_HT300to500", {
            "13": 8.764e+04, # 8.764e+04 +- 2.824e+02 pb [16] (only 1 input file used)
            }),
    CrossSection("QCD_bEnriched_HT500to700", {
            "13": 1.596e+03, # 1.596e+03 +- 9.784e-01 pb [16]
            }),
    CrossSection("QCD_bEnriched_HT700to1000", {
            "13": 3.213e+02, # 3.213e+02 +- 3.283e-01 pb [16]
            }),
    CrossSection("QCD_bEnriched_HT1000to1500", {
            "13": 5.093e+01, # 5.093e+01 +- 3.080e-01 pb [16]
            }),
    CrossSection("QCD_bEnriched_HT1500to2000", {
            "13": 4.445e+00, # 4.445e+00 +- 1.886e-02 pb [16]
            }),
    CrossSection("QCD_bEnriched_HT2000toInf", {
            "13": 7.847e-01, # 7.847e-01 +- 4.879e-03 pb [16]
            }),
    CrossSection("QCD_HT1500to2000_GenJets5", {
            "13": 6.718e+01, # 6.718e+01 +- 4.535e-02 pb [16]
            }),
    CrossSection("QCD_HT2000toInf_GenJets5", {
            "13": 1.446e+01, # 1.446e+01 +- 1.846e-02 pb [16]
            }),
    CrossSection("QCD_HT1000to1500_BGenFilter", {
            "13": 1.894e+02, # 1.894e+02 +- 1.660e-01 pb [16]
            }),
    CrossSection("QCD_HT1500to2000_BGenFilter", {
            "13": 2.035e+01, # 2.035e+01 +- 3.256e-02 pb [16]
            }),
    CrossSection("QCD_HT50to100", {
            "13": 2.464e+08, # 2.464e+08 +- 1.081e+05 pb [16]
            }),
    CrossSection("QCD_HT100to200", {
            "13": 2.803e+07, # 2.803e+07 +- 1.747e+04 pb [16]
            }),
    CrossSection("QCD_HT200to300", {
            "13": 1.713e+06, # 1.713e+06 +- 8.202e+02 pb [16]
            }),
    CrossSection("QCD_HT300to500", {
            "13": 3.475e+05, # 3.475e+05 +- 1.464e+02 pb [16]
            #"13": 351300, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("QCD_HT500to700", {
            "13": 3.208e+04, # 3.208e+04 +- 1.447e+01 pb [16]
            #"13": 31630, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("QCD_HT700to1000", {
            "13": 6.833e+03, # 6.833e+03 +- 1.668e+00 pb [16]
            #"13": 6802, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("QCD_HT1000to1500", {
            "13": 1.208e+03, # 1.208e+03 +- 5.346e-01 pb [16]
            #"13": 1206, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("QCD_HT1500to2000", {
            "13": 1.201e+02, # 1.201e+02 +- 5.823e-02 pb [16]
            #"13": 120.4, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("QCD_HT2000toInf", {
            "13": 2.526e+01, # 2.526e+01 +- 1.728e-02 pb [16]
            #"13": 25.25, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("TTTT", {
            "13": 9.103e-03, #9.103e-03 +- 1.401e-05 pb [16]
            }),
    CrossSection("TTTT_TuneCUETP8M1", {
            "13": 9.103e-03, #9.103e-03 +- 1.401e-05 pb [16] # NB: Assumed identical to TTTT_TuneCUETP8M1
            }),
    CrossSection("TTTT_TuneCUETP8M2T4", {
            "13": 9.103e-03, #9.103e-03 +- 1.401e-05 pb [16]
            }),
    CrossSection("TTWH", {
            #"13": 0.001344, #LO [30]
            "13": 1.349e-03, #1.349e-03 +- 9.745e-06 pb by running https://twiki.cern.ch/twiki/bin/view/CMS/HowToGenXSecAnalyzer#Running_the_GenXSecAnalyzer_on_a
            }),
    CrossSection("TTZH", {
            #"13": 0.001244,  #LO [31]
            "13": 1.243e-03,  #1.243e-03 +- 7.297e-06 pb by running https://twiki.cern.ch/twiki/bin/view/CMS/HowToGenXSecAnalyzer#Running_the_GenXSecAnalyzer_on_a
            }),
    CrossSection("THW_ctcvcp_HIncl_M125", {
            "13": 1.464e-01, #1.464e-01 +- 1.040e-04 pb by running https://twiki.cern.ch/twiki/bin/view/CMS/HowToGenXSecAnalyzer#Running_the_GenXSecAnalyzer_on_a
            }),
    CrossSection("TTWJetsToQQ", {
            "13": 4.034e-01, #4.034e-01 +- 2.493e-03 pb [16] (inputFiles="2211E19A-CC1E-E611-97CC-44A84225C911.root")
            }),
    CrossSection("TTWJetsToLNu", {
            "13": 0.2043, #+/- 0.0020 [17]]
            }),
    CrossSection("TTZToQQ", {
            "13": 5.297e-01, #5.297e-01 +- 7.941e-04 pb [16] (inputFiles="204FB864-5D1A-E611-9BA7-001E67A3F49D.root")
            }),
    CrossSection("TTZToLLNuNu_M_10", {
            "13": 0.2529, # [22]
            }),
    CrossSection("ttHToNonbb_M125", {
            "13": 0.5071*0.4176, # [24]
            }),
    CrossSection("ttHTobb_M125", {
            "13": 0.5071*0.5824, # [24]
            }),
    CrossSection("ttHJetTobb_M125", {
            "13": 0.2934, # [22] (NNLO)
            }),
    CrossSection("ttHJetToNonbb_M125", {
            "13": 0.2151, # [22] (NNLO QCD and NLO EW)
            }),
    CrossSection("ttHJetToTT_M125", {
            "13": 0.0321, # [22]
            }),
    CrossSection("ttHJetToGG_M125", {
            "13": 0.0012, # [22]
            }),
    CrossSection("WJetsToQQ_HT_600ToInf", {
            "13": 9.936e+01, #9.936e+01 +- 4.407e-01 pb [16] (inputFiles="0EA1D6CA-931A-E611-BFCD-BCEE7B2FE01D.root")
            }),
    CrossSection("ZJetsToQQ_HT600toInf", {
            #"13": 5.822e+02, #5.822e+02 +- 7.971e-02 pb [16] (inputFiles="0E546A76-E03A-E611-9259-0CC47A4DEDEE.root")
            "13": 5.67, # CMS AN-16-411 (approved for publication as HIG-17-022)
            }),
    CrossSection("ZJetsToQQ_HT400to600_qc19_4j", {
            "13": 146.0, # [24]
            }),
    CrossSection("ZJetsToQQ_HT600to800_qc19_4j", {
            "13": 34.1,  # [24]
            }),
    CrossSection("ZJetsToQQ_HT_800toInf_qc19_4j", {
            "13": 18.69, # [24]
            }),
    CrossSection("ZJetsToQQ_HT_800toInf", {
            "13": 18.69, # [24]
            }),
    CrossSection("ZZTo4Q", {
            "13": 6.883e+00, #6.883e+00 +- 3.718e-02 pb [16] (inputFiles="024C4223-171B-E611-81E5-0025904E4064.root")
            }),
    CrossSection("ttbb_4FS_ckm_amcatnlo_madspin_pythia8", {
            "13": 1.393e+01, #1.393e+01 +- 3.629e-02 pb [16] (inputFiles="0641890F-F72C-E611-9EA8-02163E014B5F.root")
            }),   
    )

