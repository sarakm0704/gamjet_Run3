#!/usr/bin/env python

import os
import re

RunRanges = {}
RunRanges["Run2018A"] = "315257-316995"
RunRanges["Run2018B"] = "317080-319310"
RunRanges["Run2018C"] = "319337-320065"
RunRanges["Run2018D"] = "320413-325172"
RunRanges["Run2018E"] = "325343-325520"

origJson = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

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
