import sys
import os
import json
import itertools
import importlib

from DataPath import getSkimDataPath,getSkimPythonPath
from aux import execute

SKIMPYTHONPATH = getSkimPythonPath()
if not SKIMPYTHONPATH in sys.path:
    sys.path.append(SKIMPYTHONPATH)
DATAPATH = getSkimDataPath()

def FindLumiJSON(dataset):
    if not dataset.isData:
        return None

    runrange = dataset.runRange.replace('_','-')
    jsonpath = os.path.join(dataset.path,'inputs')
    jsonfile = execute("ls %s/Cert*.*"%(jsonpath))[0]
    if os.path.exists(jsonfile):
        return jsonfile
    print("No json file for lumimask found for run range %s. File Framework/python/LumiMask.py"%runrange)
    return None

class LumiMask():
    def __init__(self,dataset,lumijson = ""):
        self.isdata = dataset.isData
        if self.isdata:
            if lumijson == "":
                lumijson = FindLumiJSON(dataset)
            self.LoadJSON(lumijson)

    def passed(self,event):
        if not self.isdata:
            return [True]*len(event)

        run = event.run
        lumi = event.luminosityBlock

        value = [str(i) in self.lumidata.keys() and (j in itertools.chain.from_iterable(range(self.lumidata[str(i)][k][0],self.lumidata[str(i)][k][1]+1) for k in range(len(self.lumidata[str(i)])))) for i,j in zip(run,lumi)]

        return value

    def LoadJSON(self,fname):
        f = open(fname)
        self.lumidata = json.load(f)
        f.close()
