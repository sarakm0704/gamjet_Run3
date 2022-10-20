#!/usr/bin/env python

import os
import sys
import re
import subprocess

import ROOT

def usage():
    print
    print("### Usage:  ",os.path.basename(sys.argv[0]),"<multicrab processed|histograms.root>")
    print

def printFiles(histofiles):
    counters = {}
    for f in histofiles:
        if os.path.isfile(f):
            fIN = ROOT.TFile.Open(f)
            cHisto = fIN.Get(os.path.join("configInfo","unweighted counters")).Clone("unweighted")
            cHisto.SetDirectory(0)
            counters[os.path.basename(f)] = cHisto
            fIN.Close()

    printcounters(counters)

def printAll(datasets):
    counters = {}
    for d in datasets:
        print("check",d,d.counters)
        cHisto = d.counters #fRF.Get("configInfo/unweighted counters").Clone("unweighted")
        counters[d.name] = cHisto

    printcounters(counters)

def printcounters(counters):
    # sort
    counternames = []
    if "Data" in counters.keys():
        counternames.append("Data")
    for k in counters.keys():
        if k not in counternames:
            counternames.append(k)
    print("check counternames",counters)
    print
    line = " "*29
    run_re = re.compile("(?P<run>\S+_Run201\S+?)_")
    for k in counternames:
        match = run_re.search(k)
        if match:
            line += "{:>20.19}".format(match.group("run"))
        else:
            line += "{:>20.19}".format(k)
    line += "\n"

    sys.stdout.write(line)
        
    for i in range(1,counters[counternames[0]].GetNbinsX()+1):
        line = "    "
        if len(counters[counternames[0]].GetXaxis().GetBinLabel(i)) > 0:
            line += "{:25.24}".format(counters[counternames[0]].GetXaxis().GetBinLabel(i))
            for n in counternames:
                line += str("{:20.1f}".format(counters[n].GetBinContent(i)))
        line += "\n"
        sys.stdout.write(line)
    
def main():

    if len(sys.argv) == 1:
        usage()
        sys.exit()

    if not os.path.isdir(sys.argv[1]):
        histofiles = sys.argv[1:]
        printFiles(histofiles)
        sys.exit()

    multicrabdir = sys.argv[1]
    if not os.path.exists(multicrabdir):
        usage()
        sys.exit()

    from plot import Dataset,getDatasets,read,mergeExtDatasets,mergeDatasets,reorderDatasets

    whitelist = []
#    whitelist = ["DoubleMuon_Run2016G","DoubleMuon_Run2016H"]
#    whitelist = ["DoubleMuon_Run2017"]
#    whitelist = ["DoubleEG_Run2017"]
#    whitelist = ["DYJets"]

    if len(sys.argv) > 2:
        whitelist = sys.argv[2:]

    blacklist = []
    datasets = getDatasets(multicrabdir,whitelist=whitelist,blacklist=blacklist)
    datasets = read(datasets)
    datasets = mergeExtDatasets(datasets)
#    datasets = mergeDatasets("Data","_Run201\d\S_",datasets)
#    datasets = mergeDatasets("DYJetsToLL_M_50","DY\S+",datasets)
    datasets = reorderDatasets(datasets)
    
    printAll(datasets)


if __name__ == "__main__":
    main()
