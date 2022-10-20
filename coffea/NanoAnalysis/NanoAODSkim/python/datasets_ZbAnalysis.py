import sys,os
JSONPATH = os.path.realpath(os.path.join(os.getcwd(),"../data"))

# The name of this file MUST contain the analysis name: datasets_<name>Analysis.py.
# E.g. NanoAOD_Hplus2taunuAnalysisSkim.py -> datasets_Hplus2taunuAnalysis.py

class Dataset:
    def __init__(self, url, dbs="global", dataVersion="94Xmc", lumiMask="", name=""):
        self.URL = url
        self.DBS = dbs
        self.dataVersion = dataVersion
        self.lumiMask = lumiMask
        self.name = name

    def isData(self):
        if "Run20" in self.URL:
            return True
        return False

    def getName(self):
        return self.name

    def getYear(self):
        year_re = re.compile("/Run(?P<year>20\d\d)\S")
        match = year_re.search(self.URL)
        if match:
            return match.group("year")
        else:
            return None

datasets = {}
datasets["2018UL"] = []
datasets["2018UL"].append(Dataset('/DoubleMuon/Run2018A-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_315257-316995_13TeV_Legacy2018_Collisions18_JSON_Run2018A.txt")))
datasets["2018UL"].append(Dataset('/DoubleMuon/Run2018B-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_317080-319310_13TeV_Legacy2018_Collisions18_JSON_Run2018B.txt")))
datasets["2018UL"].append(Dataset('/DoubleMuon/Run2018C-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_319337-320065_13TeV_Legacy2018_Collisions18_JSON_Run2018C.txt")))
datasets["2018UL"].append(Dataset('/DoubleMuon/Run2018D-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_320413-325172_13TeV_Legacy2018_Collisions18_JSON_Run2018D.txt")))

datasets["2018UL"].append(Dataset('/EGamma/Run2018A-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_315257-316995_13TeV_Legacy2018_Collisions18_JSON_Run2018A.txt")))
datasets["2018UL"].append(Dataset('/EGamma/Run2018B-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_317080-319310_13TeV_Legacy2018_Collisions18_JSON_Run2018B.txt")))
datasets["2018UL"].append(Dataset('/EGamma/Run2018C-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_319337-320065_13TeV_Legacy2018_Collisions18_JSON_Run2018C.txt")))
datasets["2018UL"].append(Dataset('/EGamma/Run2018D-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_320413-325172_13TeV_Legacy2018_Collisions18_JSON_Run2018D.txt")))

#datasets["2018UL"].append(Dataset('/SingleMuon/Run2018A-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_315257-316995_13TeV_Legacy2018_Collisions18_JSON_Run2018A.txt")))
#datasets["2018UL"].append(Dataset('/SingleMuon/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_317080-319310_13TeV_Legacy2018_Collisions18_JSON_Run2018B.txt")))
#datasets["2018UL"].append(Dataset('/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_319337-320065_13TeV_Legacy2018_Collisions18_JSON_Run2018C.txt")))
#datasets["2018UL"].append(Dataset('/SingleMuon/Run2018D-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_320413-325172_13TeV_Legacy2018_Collisions18_JSON_Run2018D.txt")))


#datasets["2018UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-20UL18JMENano_Pilot_106X_upgrade2018_realistic_v16', dataVersion="mc"))
datasets["2018UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM', dataVersion="mc"))
datasets["2018UL"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM', dataVersion="mc"))

#2017

datasets["2017UL"] = []
datasets["2017UL"].append(Dataset('/DoubleMuon/Run2017B-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_297050-299329_13TeV_UL2017_Collisions17_GoldenJSON_Run2017B.txt")))
datasets["2017UL"].append(Dataset('/DoubleMuon/Run2017C-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_299368-302029_13TeV_UL2017_Collisions17_GoldenJSON_Run2017C.txt")))
datasets["2017UL"].append(Dataset('/DoubleMuon/Run2017D-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_302031-302663_13TeV_UL2017_Collisions17_GoldenJSON_Run2017D.txt")))
datasets["2017UL"].append(Dataset('/DoubleMuon/Run2017E-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_303824-304797_13TeV_UL2017_Collisions17_GoldenJSON_Run2017E.txt")))
datasets["2017UL"].append(Dataset('/DoubleMuon/Run2017F-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_305040-306460_13TeV_UL2017_Collisions17_GoldenJSON_Run2017F.txt")))

datasets["2017UL"].append(Dataset('/DoubleEG/Run2017B-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_297050-299329_13TeV_UL2017_Collisions17_GoldenJSON_Run2017B.txt")))
datasets["2017UL"].append(Dataset('/DoubleEG/Run2017C-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_299368-302029_13TeV_UL2017_Collisions17_GoldenJSON_Run2017C.txt")))
datasets["2017UL"].append(Dataset('/DoubleEG/Run2017D-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_302031-302663_13TeV_UL2017_Collisions17_GoldenJSON_Run2017D.txt")))
datasets["2017UL"].append(Dataset('/DoubleEG/Run2017E-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_303824-304797_13TeV_UL2017_Collisions17_GoldenJSON_Run2017E.txt")))
datasets["2017UL"].append(Dataset('/DoubleEG/Run2017F-UL2017_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_305040-306460_13TeV_UL2017_Collisions17_GoldenJSON_Run2017F.txt")))

#datasets["2017UL"].append(Dataset('/SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_297050-299329_13TeV_UL2017_Collisions17_GoldenJSON_Run2017B.txt")))
#datasets["2017UL"].append(Dataset('/SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_299368-302029_13TeV_UL2017_Collisions17_GoldenJSON_Run2017C.txt")))
#datasets["2017UL"].append(Dataset('/SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_302031-302663_13TeV_UL2017_Collisions17_GoldenJSON_Run2017D.txt")))
#datasets["2017UL"].append(Dataset('/SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_303824-304797_13TeV_UL2017_Collisions17_GoldenJSON_Run2017E.txt")))
#datasets["2017UL"].append(Dataset('/SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_305040-306460_13TeV_UL2017_Collisions17_GoldenJSON_Run2017F.txt")))

#datasets["2017UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-Pilot_106X_mc2017_realistic_v9-v1/NANOAODSIM', dataVersion="mc"))                                                                 
datasets["2017UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-20UL17JMENano_106X_mc2017_realistic_v9-v1/NANOAODSIM', dataVersion="mc"))
#datasets["2017UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer19UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM', dataVersion="mc"))                                                                       
datasets["2017UL"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-20UL17JMENano_106X_mc2017_realistic_v9-v1/NANOAODSIM', dataVersion="mc"))
#datasets["2017UL"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer19UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM', dataVersion="mc"))

datasets["2017ULlowPU"] = []
#datasets["2017ULlowPU"].append(Dataset('/SingleMuon/Run2017H-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Custommix.txt")))
#datasets["2017ULlowPU"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv2-FlatPU0to75_106X_mc2017_realistic_v8-v2/NANOAODSIM', dataVersion="mc"))                                                       
#datasets["2017ULlowPU"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM', dataVersion="mc"))
#datasets["2017ULlowPU"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v1/NANOAODSIM', dataVersion="mc"))
#datasets["2017ULlowPU"].append(Dataset('/DYJetsToMuMu_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-PUForMUOVal_106X_mc2017_realistic_v9-v1/NANOAODSIM', dataVersion="mc"))

# 2016                                                                                                                                                                                                                                           
datasets["2016ULAPV"] = []
#datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016B-ver1_HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016C-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_275656-276283_13TeV_Legacy2016_Collisions16_JSON_Run2016C.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016D-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276315-276811_13TeV_Legacy2016_Collisions16_JSON_Run2016D.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016E-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276831-277420_13TeV_Legacy2016_Collisions16_JSON_Run2016E.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleMuon/Run2016F-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_277932-278800_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIP.txt")))

#datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016B-ver1_HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016B-ver2_HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016C-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_275656-276283_13TeV_Legacy2016_Collisions16_JSON_Run2016C.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016D-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276315-276811_13TeV_Legacy2016_Collisions16_JSON_Run2016D.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016E-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276831-277420_13TeV_Legacy2016_Collisions16_JSON_Run2016E.txt")))
datasets["2016ULAPV"].append(Dataset('/DoubleEG/Run2016F-HIPM_UL2016_MiniAODv2_JMENanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_277932-278800_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIP.txt")))

#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016B-ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_275656-276283_13TeV_Legacy2016_Collisions16_JSON_Run2016C.txt")))
#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276315-276811_13TeV_Legacy2016_Collisions16_JSON_Run2016D.txt")))
#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276831-277420_13TeV_Legacy2016_Collisions16_JSON_Run2016E.txt")))
#datasets["2016ULAPV"].append(Dataset('/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_277932-278800_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIP.txt")))

datasets["2016ULAPV"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-20UL16APVJMENano_106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM', dataVersion="mc"))
datasets["2016ULAPV"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-20UL16APVJMENano_106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM', dataVersion="mc"))                                                                  

datasets["2016UL"] = []
datasets["2016UL"].append(Dataset('/DoubleMuon/Run2016F-UL2016_MiniAODv2_JMENanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278801-278808_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIPfixed.txt")))
datasets["2016UL"].append(Dataset('/DoubleMuon/Run2016G-UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278820-280385_13TeV_Legacy2016_Collisions16_JSON_Run2016G.txt")))
datasets["2016UL"].append(Dataset('/DoubleMuon/Run2016H-UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_281613-284044_13TeV_Legacy2016_Collisions16_JSON_Run2016H.txt")))

datasets["2016UL"].append(Dataset('/DoubleEG/Run2016F-UL2016_MiniAODv2_JMENanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278801-278808_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIPfixed.txt")))
datasets["2016UL"].append(Dataset('/DoubleEG/Run2016G-UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278820-280385_13TeV_Legacy2016_Collisions16_JSON_Run2016G.txt")))
datasets["2016UL"].append(Dataset('/DoubleEG/Run2016H-UL2016_MiniAODv2_JMENanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_281613-284044_13TeV_Legacy2016_Collisions16_JSON_Run2016H.txt")))

#datasets["2016UL"].append(Dataset('/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278801-278808_13TeV_Legacy2016_Collisions16_JSON_Run2016F.txt")))
#datasets["2016UL"].append(Dataset('/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278820-280385_13TeV_Legacy2016_Collisions16_JSON_Run2016G.txt")))
#datasets["2016UL"].append(Dataset('/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_281613-284044_13TeV_Legacy2016_Collisions16_JSON_Run2016H.txt")))

datasets["2016UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-20UL16JMENano_106X_mcRun2_asymptotic_v17-v1/NANOAODSIM', dataVersion="mc"))
datasets["2016UL"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-20UL16JMENano_106X_mcRun2_asymptotic_v17-v1/NANOAODSIM', dataVersion="mc"))

def getDatasets(dataVersion):
    if not dataVersion in datasets.keys():
        print("Unknown dataVersion",dataVersion,", dataVersions available",datasets.keys())
        sys.exit()
    return datasets[dataVersion]
