# NanoAnalysis

CMS Physics analysis using nanoAOD's and columnar processing


## Skimming

https://gitlab.cern.ch/HPlus/nanoAnalysis/-/blob/master/NanoAODSkim/test/README.md

## Virtual environment

### First time
```
python -m venv VirtualAnalysisEnvironment
source VirtualAnalysisEnvironment/bin/activate
pip install wheel
pip install xrootd
pip install numpy==1.20
pip install coffea
```
### After the virtual environment already exists
```
source VirtualAnalysisEnvironment/bin/activate
```

### Exit virtual env
```
deactivate
```


## lxplus
```
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
```
