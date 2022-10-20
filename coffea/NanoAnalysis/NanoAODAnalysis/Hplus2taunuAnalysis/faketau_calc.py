# THIS SCRIPT WILL ONLY WORK IF THE CHANNELS HAVE BEEN NORMALIZED

import ROOT
import numpy as np
import os, sys
import glob
from operator import add, sub, mul, truediv
import datetime

# construct categorized histograms for the baseline selections and the inverted complete selection
# summed over the processes specified in ids. If inverted, instead leave out processes in ids.
def get_histo(f, ids, inverse=False):
    procs = os.listdir(f)
    def cond(name):
        return (not inverse and any([i in name for i in ids])) or (inverse and all([i not in name for i in ids]))
    procs = [os.path.join(f, p) for p in procs if cond(p)]
    files = [ROOT.TFile.Open(os.path.join(p, "results/histograms.root")) for p in procs]

    def get_mt_histos(tfile):
        histos = tfile.Get("analysis").GetListOfKeys()
#        print([h.GetName() for h in histos])
        histos = [h.ReadObj() for h in histos if "h_mt" in h.GetName()]
        grouped_histos = []
        for n, identifiers in enumerate([["baseline","Inverted"], ["baseline"], ["Inverted"]]):
            grouped_histos.append([])
            i = 0
            while i < len(histos):
                if all([idfr in histos[i].GetName() for idfr in identifiers]):
                    grouped_histos[n].append(histos.pop(i))
                else: i += 1

        return grouped_histos

    histos = [get_mt_histos(tfile) for tfile in files]

    # make the group (A, B, C) the outermost dimension
    histos = list(zip(*histos))

    def binned_sum(histos):
        # transpose so that the outest dimension contains categories and the inner processes
        histos = list(zip(*histos))

        def sum_histos(histos):
            res = histos[0].Clone()
            res.SetDirectory(0)
            for h in histos[1:]:
                res.Add(h)
            return res

        histos = [sum_histos(cat_hs) for cat_hs in histos]
        return histos

    histos = [binned_sum(group_histos) for group_histos in histos]
    return histos


def get_obs(f):
    ids = ["Tau_Run"]
    return get_histo(f, ids)

def getMC(f):
    excl_ids = ["Tau_Run", "ChargedHiggs", "QCDandFakeTau"]
    return get_histo(f, excl_ids, inverse=True)

def write_results(histos, f):
    out_path = os.path.join(f, "QCDandFakeTau/results")
    os.makedirs(out_path, exist_ok=True)
    fout = ROOT.TFile.Open(os.path.join(out_path, "histograms.root"), "RECREATE")
    fout.cd()
    fout.mkdir("configInfo")
    fout.cd("configInfo")

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    now = datetime.datetime.now()
    m = "produced: %s %s"%(days[now.weekday()],now)
    timestamp = ROOT.TNamed(m,"")
    timestamp.Write()

    fout.cd()
    fout.mkdir("analysis")
    fout.cd("analysis")
    for h in histos:
        h.Write()
    fout.cd()
    fout.Close()
    return


def nested_op(op, a, b):
    if type(a) == list and type(b) == list:
        return [nested_op(op, aa, bb) for aa, bb in zip(a, b)]
    else:
        return op(a, b)

def update_names(histos):
    def remove_subs(text, subs):
        for item in subs:
            text = text.replace(item, "")
        return text
    unique_names = []
    targets = []
    for h in histos:
        name = h.GetName()
        newname = remove_subs(name, ("Inverted_", "lowpt_", "midpt_", "highpt_", "loweta_", "mideta_", "higheta_"))
        if newname not in unique_names:
            targets.append(len(unique_names))
            unique_names.append(newname)
        else:
            targets.append(unique_names.index(newname))
        h.SetName(newname)
    return targets

def combine_bins(histos, targets):
    n_ret = np.max(targets)
    ret = []
    for i in range(n_ret + 1):
        first_loc = targets.index(i)
        ret_h = histos[first_loc].Clone()
        for j, h in enumerate(histos[first_loc+1:]):
            if targets[j] == i:
                ret_h.Add(h)
        ret.append(ret_h)
    return ret

def main(f):

    # get binned histograms of observed events
    obs = get_obs(f)

    # get binned background sample histograms
    mc = getMC(f)

    # subtract simulated histograms from observations to get approximation of QCD/fake tau backgrounds
    A, B, C = nested_op(sub, obs, mc)

    # derive transfer factors
    transfer_factors = nested_op(truediv, B, A)

    # apply transfer factors
    D_components = nested_op(mul, C, transfer_factors)

    # remove things relating to fake tau process from histogram names
    targets = update_names(D_components)

    # merge fake tau bins into total distributions
    D_categorized = combine_bins(D_components, targets)

    # write to a QCDandFakeTau folder, formatted the same way as original result directories
    write_results(D_categorized, f)

if __name__ == "__main__":
    resultdir = sys.argv[1]
    main(resultdir)