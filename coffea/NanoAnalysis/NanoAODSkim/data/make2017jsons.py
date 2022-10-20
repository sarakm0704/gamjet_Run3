#!/usr/bin/env python

import os
import re

RunRanges = {}
RunRanges["Run2017B"] = "297050-299329"
RunRanges["Run2017C"] = "299368-302029"
RunRanges["Run2017D"] = "302031-302663"
RunRanges["Run2017E"] = "303824-304797"
RunRanges["Run2017F"] = "305040-306460"

origJson = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"

templateJson = os.path.basename(origJson)
templateJson = templateJson.replace(".txt","_%s.txt")
templateJson = re.sub(r'_\d+-\d+_','_%s-%s_',templateJson)
#print templateJson

for rr in RunRanges.keys():
    runmin = RunRanges[rr][:6]
    runmax = RunRanges[rr][7:]
    newJson = templateJson%(runmin,runmax,rr)
    command = "jsonrunsel.py %s %s %s %s"%(runmin,runmax,origJson,newJson)
    os.system(command)
    print command
#jsonrunsel.py 297050 299329 Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt Cert_297031-297723_13TeV_PromptReco_Collisions17_JSON_Run2017B.txt
