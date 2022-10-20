#!/usr/bin/env python2

# Needs the original multicrab dir as an argument. Before copying the multicrab dir, one should run the pileup and lumi scripts.
# Uses ARC middleware, need to have ARC client installed, and a valid proxy (arcproxy)
# Srm-part not tested
# ### Usage:   multicrabGetFromMadhatter.py <multicrab dir|crab dir>
# S.Lehti 18.9.2020

import os
import sys
import re
import subprocess
import json

from optparse import OptionParser

#STORAGE_ELEMENT_PATH = "gsiftp://madhatter.csc.fi/pnfs/csc.fi/data/cms/store/group/local/Higgs/CRAB3_TransferData/"
#MADHATTER_PATH = "gsiftp://madhatter.csc.fi/pnfs/csc.fi/data/cms/store/group/local/HiggsChToTauNuFullyHadronic/CRAB3_TransferData"
STORAGE_ELEMENT_PATH = "gsiftp://madhatter.csc.fi/pnfs/csc.fi/data/cms//store/user/slehti/CRAB3_TransferData/"

USEARC = True

if USEARC:
    GRIDCOPY  = "arccp"
    GRIDLS    = "arcls"
    GRIDPROXY = "arcproxy --info"
else:
    GRIDCOPY  = "srmcp"
    GRIDLS    = "srmls"
    GRIDPROXY = "grid-proxy-info"
    
def usage():
    print
    print("### Usage:   ",os.path.basename(sys.argv[0])," <multicrab dir|crab dir>")
    print
    sys.exit()

def execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    f = p.stdout
    ret=[]
    for line in f:
        line = line.decode('utf-8')
        #sys.stdout.write(line)
        ret.append(line.replace("\n", ""))
    f.close()
    return ret

def getMulticrabDirName(path):
    if not os.path.exists(path):
        print("Path",path,"does not exist")
        sys.exit()
    multicrab_re = re.compile("(?P<basedir>.+)(?P<mg>multicrab(_|-)\S+?)(/|$)")
    match = multicrab_re.search(os.path.abspath(path))
    #match = multicrab_re.search(path)
    if match:
        if ' ' in match.group("basedir"):
            return "./",match.group("mg")
        return match.group("basedir"),match.group("mg")
    else:
        usage()

def getCrabdirs(multicrabdir,paths):
    if os.path.abspath(paths[0]) == os.path.abspath(multicrabdir):
        paths = execute("ls %s"%multicrabdir)

    cleanedPaths = []
    for p in paths:
        if p[len(p)-1:] == "/":
            cleanedPaths.append(p[:len(p)-1])
        else:
            cleanedPaths.append(p) 

    crabdirs = []
    cands = execute("ls %s"%multicrabdir)
    for c in cands:
        if c not in cleanedPaths:
            continue
        cp = os.path.join(multicrabdir,c)
        if os.path.isdir(cp) and os.path.exists(os.path.join(cp,"results")):
            crabdirs.append(c)
    return crabdirs

def getSEPath(path):
    return STORAGE_ELEMENT_PATH
    cands = execute("ls %s"%path)
    cfg_re = re.compile("crabConfig_\S+.py")
    crabCfgs = []
    for c in cands:
        match = cfg_re.search(c)
        if os.path.isfile(os.path.join(path,c)) and match:
            crabCfgs.append(c)
    sepath = execute("grep outLFNDirBase %s"%os.path.join(path,crabCfgs[0]))
    sepath_re = re.compile("(?P<path>/store\S+)multicrab")
    match = sepath_re.search(sepath[0])
    if match:
        retpath = STORAGE_ELEMENT_PATH
        if not retpath.endswith('/'):
            retpath += '/'
        retpath += match.group("path")
        return retpath
    else:
        print("Could not determine SE path")
        sys.exit()
    return None
    
def findRootFiles(path):
    root_re = re.compile("(?P<rootfile>(\S+\.root))")
    files = []
    subpaths = execute("%s %s"%(GRIDLS,path))
    #print("gridls %s"%path)
    for sp in subpaths:
        if sp == "log":
            continue
        #print sp
        match = root_re.search(sp)
        if match:
            files.append(os.path.join(path,sp))
        else:
            files.extend(findRootFiles(os.path.join(path,sp)))
    return files

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def retrieve(path,savedir,opts):
    cdir = os.path.basename(os.path.dirname(savedir))
    #print "check retrieve os.path.basename(savedir)",os.path.basename(savedir)
    if not os.path.exists(savedir):
        return

    rootfiles = findRootFiles(path)

    if opts.list:
        fOUT = os.path.join(savedir,"files_%s.json"%cdir)
        if not os.path.exists(fOUT):
            dict = {}
            dict['files'] = list(map(lambda x : x.replace("gsiftp://","root://"), rootfiles))
            f = open(fOUT, "wb")
            json.dump(dict, f, sort_keys=True, indent=2)
            f.close()
            sys.stdout.write('    Listed files in %s\n'%os.path.basename(fOUT))
        else:
            sys.stdout.write('    List file already found: %s\n'%os.path.basename(fOUT))
        return

    length = 0
    for i,rf in enumerate(rootfiles):
        filename = os.path.basename(rf)
        str_out = "%s, retrieved %i/%i"%(cdir,i+1,len(rootfiles))
        while len(str_out) < length:
            str_out+=" "
        length = len(str_out)
        sys.stdout.write(str_out)
        sys.stdout.flush()
        restart_line()
        if not os.path.exists(os.path.join(savedir,filename)):
            cp_cmd = GRIDCOPY+" %s file:///%s"%(rf,os.path.join(savedir,filename))
            print(cp_cmd)
            os.system(cp_cmd)
            chmod_cmd = "chmod 644 %s"%os.path.join(savedir,filename)
            os.system(chmod_cmd)
    sys.stdout.write("\n")

def proxy():
    proxyResult = execute(GRIDPROXY)
    timeleft_re = re.compile("Time left for \S+: (?P<time>.*)")
    for line in proxyResult:
        match = timeleft_re.search(line)
        if match:
            print("Time left for proxy:",match.group("time"))
            if "expired" in match.group("time"):
                sys.exit()

def main(opts, args):

    proxy()

    pIN = []
    if len(sys.argv) == 1:
        pIN.append(os.getcwd())
    else:
        pIN.extend(args)

    basedir,multicrab = getMulticrabDirName(pIN[0])
    crabdirs  = getCrabdirs(os.path.join(basedir,multicrab),pIN)

    storagePath = getSEPath(os.path.join(basedir,multicrab))

    print(multicrab)

    retrievePaths = {}
    for cdir in crabdirs:
        sys.stdout.write("    Scanning files at madhatter: %s\n"%cdir)
        gsipath = os.path.join(storagePath,multicrab)
        #print(gsipath)
        cands = execute("%s %s"%(GRIDLS,gsipath))
        #print "check cands",cands
        for ddir in cands:
            cands2 = execute("%s %s"%(GRIDLS,os.path.join(gsipath,ddir)))
            for edir in cands2:
                if "crab_"+cdir == edir:
                    retrievePaths[cdir] = os.path.join(gsipath,ddir,edir)
                    #print "check retrievePaths",cdir,os.path.join(gsipath,ddir,edir)
    #print(crabdirs)
    for cdir in crabdirs:
        #print("check cdir",cdir)
        if cdir in retrievePaths.keys():
            #print "check retrieve",retrievePaths[cdir],os.path.join(basedir,multicrab,cdir,"results")
            retrieve(retrievePaths[cdir],os.path.join(basedir,multicrab,cdir,"results"),opts)
        else:
            print("\033[93m%s not retrieved\033[0m"%cdir)

if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog [options]")

    parser.add_option("--list", dest="list", default=False, action="store_true",
                      help="Write a list of root files in text files [defaut: False]")
    (opts, args) = parser.parse_args()

    main(opts, args)
