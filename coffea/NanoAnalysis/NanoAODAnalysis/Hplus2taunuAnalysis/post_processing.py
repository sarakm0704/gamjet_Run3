import ROOT
import ROOT.gDirectory as gdir
import sys
from pathlib import Path
from argparse import ArgumentParser
import copy
import re, os

basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    BASEPATH = os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python")
    sys.path.append(BASEPATH)

from hist2root import convertCounter
'''
This script adds the dataVersion field to pseudomulticrabs produced in analysis.py
Additionally it can rename the analysis folder in the result rootfiles

USAGE: python add_dataver2results.py <version_name> <resultdir1> [<resultdir2> ...]

type the version names without the postfix specifying if the rootfile contains data or mc,
it will automatically be inferred from the configInfo/isData field.

Warning: any previous dataVersion objects in configInfos traversed by this script will be
overwritten! (with the one exception explained below)

Example: python add_dataver2results.py 2017_106XUL .

The above example will look for all rootfiles in the current working directory, check if
they contain a configInfo and write the dataVersion field into files that do. An exception
is with QCD measurement pseudomulticrabs, which already should have the dataVersion set as
"pseudo". These are left untouched in this script.
'''
def parse_args():
    parser = ArgumentParser(description="")
    parser.add_argument("pseudo_locations", nargs='+', help='')
    parser.add_argument("-v", "--data_version", help = "")
    parser.add_argument("-n", "--newname", help='')
    parser.add_argument("-c", "--clean", action = "store_true", help='')
    parser.add_argument("-s", "--splitted_bins", help='comma separated bin counts: control,taupt,eta')
    parser.add_argument("-E", "--energy", type=int, default=13, help='')
    parser.add_argument("-p", "--isPUReweighted", action="store_true",help='')
    parser.add_argument("-t", "--isTopPtReweighted", action="store_true",help='')
    return parser.parse_args()

def copy_dir(src, newname=None):
    src_keys = src.GetListOfKeys()
    savdir = ROOT.gDirectory.GetDirectory('')
    name = src.GetName() if newname is None else newname
    new = savdir.mkdir(name)
    new.cd()
    for key in src_keys:
        classname = key.GetClassName()
        cls = ROOT.gROOT.GetClass(classname)
        if not cls: continue
        if cls.InheritsFrom("TDirectory"):
            src.cd(key.GetName())
            subdir = ROOT.gDirectory.GetDirectory('')
            new.cd()
            copy_dir(subdir)
            new.cd()
        elif cls.InheritsFrom("TTree"):
            t = src.Get(key.GetName())
            new.cd()
            newt = t.CloneTree(-1,"fast")
            newt.Write()
        else:
            src.cd()
            obj = key.ReadObj()
            new.cd()
            obj.Write()
            del obj
    new.SaveSelf(ROOT.kTRUE)
    savdir.cd()

def write_confinfo_h(args):
    gdir.Delete("configinfo;*")
    control=1
    energy = control * args.energy
    isdata = control * gdir.Get("isdata").GetBinContent(1)
    ptreweight = control * (not isdata and args.isPUReweighted)
    topreweight = control * (not isdata and args.isTopPtReweighted)
    d = {
        'control': control,
        'energy': energy,
        'isData': isdata,
        'isPileupReweighted': ptreweight,
        'isTopPtReweighted': topreweight
    }
    return convertCounter(d, "configinfo").Write()


def main(args):
    pseudos = args.pseudo_locations
    assert len(pseudos) > 0
    rfiles = [Path(d).rglob("*.root") for d in pseudos]
    rfiles = (str(filepath) for pseudofiles in rfiles for filepath in pseudofiles)
    for fname in rfiles:
        print("opening "+fname)
        f = ROOT.TFile.Open(fname,"update")
        keys = f.GetListOfKeys()
        if "configInfo" not in keys: f.Close(); continue
        f.cd("configInfo")
        write_confinfo_h(args)
        f.cd()

        for key in keys:
            if key.GetName() != "configInfo":
                classname = key.GetClassName()
                cls = ROOT.gROOT.GetClass(classname)
                if not cls: continue
                if cls.InheritsFrom("TDirectory"):
                    f.cd(key.GetName())
                    if "counters" in gdir.GetListOfKeys():
                        gdir.rmdir("counters")
                    gdir.mkdir("counters")
                    gdir.cd("counters")
                    unw_counters = f.Get("configInfo/unweighted_counter")
                    unw_counters.SetName("counter")
                    unw_counters.Write()
                    gdir.mkdir("weighted")
                    gdir.cd("weighted")
                    w_counters = f.Get("configInfo/weighted_counter")
                    w_counters.SetName("counter")
                    w_counters.Write()

                    f.cd(key.GetName())
                    if args.splitted_bins and "QCDMeasurement" in f.GetName():
                        gdir.Delete("SplittedBinInfo;*")
                        gdir.Delete("splittedBinInfo;*")
                        bin_h = ROOT.TH1D("SplittedBinInfo","",3,0,3)
                        ctrl_bins, pt_bins, eta_bins = (int(n) for n in args.splitted_bins.split(","))
                        bin_h.SetBinContent(1,ctrl_bins)
                        bin_h.GetXaxis().SetBinLabel(1,"Control")
                        bin_h.SetBinContent(2,pt_bins)
                        bin_h.GetXaxis().SetBinLabel(2,"tauPt")
                        bin_h.SetBinContent(3,eta_bins)
                        bin_h.GetXaxis().SetBinLabel(3,"tauEta")
                        bin_h.Write()

        f.cd()
        if args.newname:
            if "analysis" in keys:
                [f.rmdir(key.GetName()) for key in keys if any(i in key.GetName() for i in ["QCDMeasurement_", "SignalAnalysis"])]
                is_qcd = "QCDMeasurement" in f.GetName().split("/")[-4]
                prefix = "QCDMeasurement_" if is_qcd else "SignalAnalysis_"
                newname = prefix + args.newname
                copy_dir(f.Get("analysis"), newname)
                if args.clean:
                    f.rmdir("analysis")

        if args.data_version:
            ver = args.data_version
            if "QCDMeasurementMT" in f.GetName(): f.Close(); continue
            f.cd("configInfo")
            ROOT.gDirectory.Delete("dataVersion;*")
            isdata = bool(f.Get("configInfo/isdata").GetBinContent(1))
            postfix = "data" if isdata else "mc"
            dataver = ROOT.TNamed("dataVersion", ver + postfix)
            dataver.Write()
            f.cd()
            f.Close()

if __name__ == "__main__":
    args = parse_args()
    main(args)