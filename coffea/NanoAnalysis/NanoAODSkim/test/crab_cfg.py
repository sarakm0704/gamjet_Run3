from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config, getUsernameFromCRIC

config = Configuration()

config.section_("General")
config.General.requestName = 'NanoAnalysis'
config.General.workArea = dirName
config.General.transferOutputs = True
config.General.transferLogs=True

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'PSet.py'
config.JobType.scriptExe = 'NanoAOD_crab_script.sh'
config.JobType.inputFiles = ['NanoAOD_*AnalysisSkim.py','PSet.py']
config.JobType.maxJobRuntimeMin = 2*1315
config.JobType.sendPythonFolder	 = True
config.JobType.outputFiles = ['events.root']

config.section_("Data")
config.Data.inputDataset = 'FromMulticrab'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outLFNDirBase = '/store/user/%s/NanoPost' % (getUsernameFromCRIC())
config.Data.publication = False

config.section_("Site")
config.Site.storageSite = "T2_FI_HIP"


