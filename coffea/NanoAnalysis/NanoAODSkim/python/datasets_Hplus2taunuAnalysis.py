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
        year_re = re.compile("/Run(?P<year>201\d)\S")
        match = year_re.search(self.URL)
        if match:
            return match.group("year")
        else:
            return None

datasets = {}
datasets['2016ULAPV'] = []
datasets["2016ULAPV"].append(Dataset('/Tau/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_273150-275376_13TeV_Legacy2016_Collisions16_JSON_Run2016B.txt")))
datasets["2016ULAPV"].append(Dataset('/Tau/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_275656-276283_13TeV_Legacy2016_Collisions16_JSON_Run2016C.txt")))
datasets["2016ULAPV"].append(Dataset('/Tau/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276315-276811_13TeV_Legacy2016_Collisions16_JSON_Run2016D.txt")))
datasets["2016ULAPV"].append(Dataset('/Tau/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_276831-277420_13TeV_Legacy2016_Collisions16_JSON_Run2016E.txt")))
datasets["2016ULAPV"].append(Dataset('/Tau/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_277932-278800_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIP.txt")))

datasets["2016ULAPV"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M100_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M120_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M140_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M145_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M150_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M155_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M160_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M165_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M170_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M175_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M180_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M190_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M200_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M80_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M90_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))

datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))

datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M180_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M190_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M200_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M220_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M250_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M300_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M400_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M600_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M700_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M800_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))
datasets["2016ULAPV"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M3000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM'))

datasets['2016UL'] = []
datasets["2016UL"].append(Dataset('/Tau/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278801-278808_13TeV_Legacy2016_Collisions16_JSON_Run2016F_HIPfixed.txt")))
datasets["2016UL"].append(Dataset('/Tau/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_278820-280385_13TeV_Legacy2016_Collisions16_JSON_Run2016G.txt")))
datasets["2016UL"].append(Dataset('/Tau/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_281613-284044_13TeV_Legacy2016_Collisions16_JSON_Run2016H.txt")))

datasets["2016UL"].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))

datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M80_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M90_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M100_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M120_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M140_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M145_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M150_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M155_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M160_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M165_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M170_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M175_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M180_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M190_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_IntermediateNoNeutral_M200_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))

datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))

datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M180_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M190_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M200_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M220_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M300_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M400_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M600_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M700_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M800_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets["2016UL"].append(Dataset('/ChargedHiggsToTauNu_Heavy_M3000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))


datasets['2017UL'] = []
datasets["2017UL"].append(Dataset('/Tau/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_297050-299329_13TeV_UL2017_Collisions17_GoldenJSON_Run2017B.txt")))
datasets["2017UL"].append(Dataset('/Tau/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_299368-302029_13TeV_UL2017_Collisions17_GoldenJSON_Run2017C.txt")))
datasets["2017UL"].append(Dataset('/Tau/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_302031-302663_13TeV_UL2017_Collisions17_GoldenJSON_Run2017D.txt")))
datasets["2017UL"].append(Dataset('/Tau/Run2017E-UL2017_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_303824-304797_13TeV_UL2017_Collisions17_GoldenJSON_Run2017E.txt")))
datasets["2017UL"].append(Dataset('/Tau/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_305040-306460_13TeV_UL2017_Collisions17_GoldenJSON_Run2017F.txt")))

datasets['2017UL'].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM'))

datasets['2017UL'].append(Dataset('/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM'))

datasets['2017UL'].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM'))

datasets['2017UL'].append(Dataset('/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))

datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M200_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M220_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M250_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M400_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M700_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M800_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))
datasets['2017UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M3000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'))

datasets["2018UL"] = []
datasets["2018UL"].append(Dataset('/Tau/Run2018A-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_315257-316995_13TeV_Legacy2018_Collisions18_JSON_Run2018A.txt")))
datasets["2018UL"].append(Dataset('/Tau/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_317080-319310_13TeV_Legacy2018_Collisions18_JSON_Run2018B.txt")))
datasets["2018UL"].append(Dataset('/Tau/Run2018C-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_319337-320065_13TeV_Legacy2018_Collisions18_JSON_Run2018C.txt")))
datasets["2018UL"].append(Dataset('/Tau/Run2018D-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD', dataVersion="data",lumiMask=os.path.join(JSONPATH,"Cert_320413-325172_13TeV_Legacy2018_Collisions18_JSON_Run2018D.txt")))

datasets['2018UL'].append(Dataset('/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))

datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))

datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M170_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M175_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M180_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M190_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M200_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M220_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M250_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M300_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M400_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M600_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M700_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M800_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))  
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M1500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M2500_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))
datasets['2018UL'].append(Dataset('/ChargedHiggsToTauNu_Heavy_M3000_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM'))


def getDatasets(dataVersion):
    if not dataVersion in datasets.keys():
        print("Unknown dataVersion",dataVersion,", dataVersions available",datasets.keys())
        sys.exit()
    return datasets[dataVersion]
