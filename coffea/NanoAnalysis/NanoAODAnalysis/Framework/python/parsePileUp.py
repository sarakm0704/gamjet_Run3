#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea.lookup_tools import extractor

from DataPath import getDataPath

# Get pileup for data as a function of run and lumiSection.
# It needs the pileup json file for lookup, value from the
# file multiplied with the minbias cross section
#
# Usage: call parsePileUpJSON2 in the Analysis.__init__, which
# makes the lookup table, and call getAvgPU in the Analysis.process
#
# pileup jsons
# scp lxplus.cern.ch:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/pileup_latest.txt pileup_2016.txt
# scp lxplus.cern.ch:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt pileup_2017.txt
# scp lxplus.cern.ch:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/pileup_latest.txt pileup_2018.txt


#MINBIAS_XS = 69200
ext = None

def getAvgPU(run,luminosityBlock):
    global ext
    evaluator = ext.make_evaluator()
    return evaluator["pileup"](run,luminosityBlock)
    
def getAvgPUNoGlobal(ext,run,luminosityBlock):
    evaluator = ext.make_evaluator()
    return evaluator["pileup"](run,luminosityBlock)

def parsePileUpJSON2(year):
    datapath = getDataPath()
    filename = ""
    MINBIAS_XS = 69200
    if "2016" in year:
        filename = "pileup_2016.txt"
    if "2017" in year:
        filename = "pileup_2017.txt"
    if "2018" in year:
        filename = "pileup_2018.txt"
    if "2022" in year:
        filename = "pileup_latest_2022.txt"
        MINBIAS_XS = 80000

    filename = os.path.join(datapath,'pileup',filename)
    print("Data pileup from",os.path.basename(filename),", Minimum Bias Cross Section: ",MINBIAS_XS)

    global ext
    ext = extractor()
    ext.add_weight_sets(["pileup pileup %s"%filename])
    multiplyWeight(ext._weights,MINBIAS_XS)
    ext.finalize()

def parsePileUpJSONNoGlobal(year):
    datapath = getDataPath()
    filename = ""
    MINBIAS_XS = 69200
    if "2016" in year:
        filename = "pileup_2016.txt"
    if "2017" in year:
        filename = "pileup_2017.txt"
    if "2018" in year:
        filename = "pileup_2018.txt"
    if "2022" in year:
        filename = "pileup_latest_2022.txt"
        MINBIAS_XS = 80000

    filename = os.path.join(datapath,'pileup',filename)
    print("Data pileup from",os.path.basename(filename),", Minimum Bias Cross Section: ",MINBIAS_XS)

    ext = extractor()
    ext.add_weight_sets(["pileup pileup %s"%filename])
    multiplyWeight(ext._weights,MINBIAS_XS)
    ext.finalize()
    return ext

def multiplyWeight(weightset,value):
    for i in range(len(weightset)):
        w2 = weightset[i]
        for j in range(len(w2)):
            w3 = w2[j]
            for k in w3.keys():
                w4 = w3[k]
                for l in w4.keys():
                    w4[l] = w4[l]*value


