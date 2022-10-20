#!/usr/bin/env python                                                                                                                                                                                          
#
# Read Pileup from NanoAOD MC
# 1231.8.2022/S.Lehti
#
import sys,os,re
import subprocess
from optparse import OptionParser

import ROOT


def usage():
    print
    print( "### Usage:  ",os.path.basename(sys.argv[0]),"<multicrab skim>" )
    print

root_re = re.compile("(?P<rootfile>([^/]*events_\d+\.root))")

class Dataset:
    def __init__(self,path,run):
        self.path = path
        self.name = os.path.basename(path)
        self.isData = False
        if "Run20" in self.name:
            self.isData = True

        self.files = []
        cands = execute("ls %s"%os.path.join(path,"results"))
        for c in cands:
            match = root_re.search(c)
            if match:
                self.files.append(os.path.join(path,"results",match.group("rootfile")))

        if len(self.files) == 0:
            print("Dataset contains no root files")
            return

    def getFileNames(self):
        return self.files

def getDatasets(multicrabdir,whitelist=[],blacklist=[]):
    datasets = []
    cands = execute("ls %s"%multicrabdir)

    run = getRun(cands)

    for c in cands:
        resultdir = os.path.join(multicrabdir,c,"results")
        if os.path.exists(resultdir):
            datasets.append(Dataset(os.path.join(multicrabdir,c),run))

    return datasets

def getRun(datasetnames):
    run = ""
    run_re = re.compile("(?P<run>Run201\d)")
    for n in datasetnames:
        match = run_re.search(n)
        if match:
            run = match.group("run")
    return run

def execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    (s_in, s_out) = (p.stdin, p.stdout)

    f = s_out
    ret=[]
    for line in f:
        line = str(line,'utf-8')
        ret.append(line.replace("\n", ""))
    f.close()
    return ret

def main():
    
    if len(sys.argv) == 1:
        usage()
        sys.exit()

    multicrabdir = os.path.abspath(sys.argv[1])
    if not os.path.exists(multicrabdir) or not os.path.isdir(multicrabdir):
        usage()
        sys.exit()

    parser = OptionParser(usage="Usage: %prog [options]")
    parser.add_option("--overwrite", dest="overwrite", default=False, action="store_true",
                      help="Overwrite the pileup distributions [default: False")
    (opts, args) = parser.parse_args()

    blacklist = []
    whitelist = []
    datasets = getDatasets(multicrabdir,whitelist=whitelist,blacklist=blacklist)

    pileup_template = ROOT.TH1F("pileup","",100,0,100)

    for dataset in datasets:
        if dataset.isData:
            continue

        filepath = os.path.join(dataset.path,"results","PileUp.root")
        if os.path.exists(filepath) and not opts.overwrite:
            print("%s: Pileup file already exists"%dataset.name)
            continue

        tchain = ROOT.TChain("Events")
        tchain.SetCacheSize(10000000)

        for f in dataset.getFileNames():
            tchain.Add(f)

        if tchain.GetEntries() == 0:
            return

        pileup_template.Reset()
        tchain.Draw("Pileup_nTrueInt>>pileup", ROOT.TCut(""), "goff")
        print("Pileup distribution with",pileup_template.GetEntries(),"entries, integral",pileup_template.Integral())

        fOUT = ROOT.TFile.Open(filepath,"RECREATE")
        fOUT.cd()
        pileup_template.Write()
        fOUT.Close()
        print(filepath)


if __name__=="__main__":
    main()
