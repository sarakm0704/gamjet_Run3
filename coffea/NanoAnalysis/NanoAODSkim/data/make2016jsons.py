#!/usr/bin/env python

import os
import re

RunRanges = {}
RunRanges["Run2016B"] = "273150-275376"
RunRanges["Run2016C"] = "275656-276283"
RunRanges["Run2016D"] = "276315-276811"
RunRanges["Run2016E"] = "276831-277420"
RunRanges["Run2016F"] = "277932-278808"
RunRanges["Run2016G"] = "278820-280385"
RunRanges["Run2016H"] = "281613-284044"

origJson = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

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
