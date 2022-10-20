################################################
# NanoAOD skimming with trigger bit only
# 10.12.2019/SLehti
################################################

## CMSSW working area

scram p CMSSW CMSSW_10_6_31_patch1
cd CMSSW_10_6_31_patch1
git clone ssh://git@gitlab.cern.ch:7999/${USER}/NanoAnalysis.git
cd NanoAnalysis/NanoAODSkim/test

## How to make your own skim

Copy test/NanoAOD_*AnalysisSkim.py for your analysis name
and change the triggers in the code.

Copy python/datasets_*Analysis.py for your analysis name
and change the datasets in the file.

## How to submit

../scripts/multicrab.py NanoAOD_*AnalysisSkim.py 2017UL

After DATA is 100% complete:
lumicalc.py <multicrabdir>
pileup.py <multicrabdir>
pileup_mc.py <multicrabdir>
