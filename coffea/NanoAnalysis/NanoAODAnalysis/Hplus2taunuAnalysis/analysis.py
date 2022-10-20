#!/usr/bin/env python

# grey out deprecation warnings and tensorflow information
print("\033[2;1m")
import gc
import ctypes
import sys
import os,re
import shutil
import getpass
import subprocess
import datetime
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt
import matplotlib.scale as scl
import shutil
import getpass
import os.path
from functools import reduce
import operator
import json
from contextlib import nullcontext
import awkward as ak

from coffea import nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from coffea import lookup_tools
from coffea import analysis_tools

from typing import Iterable, Callable, Optional, List, Generator, Dict, Union
import collections

import ROOT

try:
    import tensorflow as tf
except:
    print("failed importing tensorflow, make sure you have it installed if you are running with neural networks!")

basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    BASEPATH = os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python")
    sys.path.append(BASEPATH)
    ANALYSISPATH = os.path.join(match.group("basepath"),"NanoAODAnalysis/Hplus2taunuAnalysis")
    sys.path.append(ANALYSISPATH)
    ANNPATH = os.path.join(match.group("basepath"),"NanoAODAnalysis/Hplus2taunuAnalysis/ANN")
    sys.path.append(ANNPATH)


import multicrabdatasets
import hist2root
import Counter
import PileupWeight
from parsePileUp import getAvgPUNoGlobal, parsePileUpJSONNoGlobal, parsePileUpJSON2, getAvgPU
import LumiMask
import JetvetoMap
import Btag
import aux
import crosssection
import TransverseMass

import Hplus2taunuSelection as selection
import Hplus2taunuHistograms as Histograms
import Hplus2taunuChannels as channels
from datasetUtils import ss, ns, ts, hs, ls, es, cs
from NNServer import init_server, kill_server
print(ns)

try:
    import dask
    from distributed import Client
    from distributed.diagnostics import MemorySampler
    from dask_jobqueue import HTCondorCluster
    import socket
    JOBS = 16
    try:
        usr = getpass.getuser()
        initial = usr[0]
        bash_command = "voms-proxy-info -path"
        proxy_stored_path = subprocess.run(bash_command.split(), check=True, text=True, capture_output=True).stdout.strip()
        proxy_name = proxy_stored_path.split("/")[-1]
        PROXYPATH = f"/afs/cern.ch/user/{initial}/{usr}/private/{proxy_name}"
        shutil.copyfile(proxy_stored_path, PROXYPATH)
        DISTRIBUTED=True
    except Exception as e:
        DISTRIBUTED = False
        print(e)
        print(ls + "Failed transferring grid proxy to available location, will run analysis iteratively" + ns)
except:
    DISTRIBUTED = False
    print(ss + "Failed importing daskExecutor dependencies, will run analysis iteratively" + ns)

# defining magic variables, should some day be moved to some kind of config (file, script arguments?)
MEASUREMENT_BINS = [["Rtau75", "Rtau1"],]
QCDSEP = ["SignalAnalysis", "QCDMeasurement"]
SEPARATE = ak.Array(MEASUREMENT_BINS + [QCDSEP,])#, ["nn_50", "nn_90", "nn_1"]])
QCD_DIRS = { # for compatibility with old HiggsAnalysis code
    'ForQCDNormalization': [
        "NormalizationMETBaselineTauAfterStdSelections",
        "NormalizationMtBaselineTauAfterStdSelections",
        "NormalizationMETInvertedTauAfterStdSelections",
        "NormalizationMtInvertedTauAfterStdSelections",
    ], 'ForQCDNormalizationEWKFakeTaus': [
        "NormalizationMETBaselineTauAfterStdSelections",
        "NormalizationMtBaselineTauAfterStdSelections",
        "NormalizationMETInvertedTauAfterStdSelections",
        "NormalizationMtInvertedTauAfterStdSelections",
    ], 'ForQCDNormalizationEWKGenuineTaus': [
        "NormalizationMETBaselineTauAfterStdSelections",
        "NormalizationMtBaselineTauAfterStdSelections",
        "NormalizationMETInvertedTauAfterStdSelections",
        "NormalizationMtInvertedTauAfterStdSelections",
    ], 'ForQCDMeasurement': [
        "BaselineTauShapeTransverseMass",
    ], 'ForQCDMeasurementEWKFakeTaus': [
        "BaselineTauShapeTransverseMass",
    ], 'ForQCDMeasurementEWKGenuineTaus': [
        "BaselineTauShapeTransverseMass",
    ],
}
TAU_BINS = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 190, 220, 250, 300, 400, 500]

PRINTED_COUNTERS = ["Inclusive"]

# DISTRIBUTED = False # uncomment if you want to force local execution, for example debugging purposes

MAX_WORKERS = max(multiprocessing.cpu_count() - 2,1)

CHUNKSIZE = 100000
MAXCHUNKS = None

MAXEVENTS = -1
#MAXEVENTS = 1000
if MAXEVENTS > 0:
    CHUNKSIZE = MAXEVENTS
    MAXCHUNKS = 1

ENERGY = "13"

class Analysis(processor.ProcessorABC):
    def __init__(self,dataset, pu_data, luminosity, nn_path = None, mass_point = None, **analysis_kwargs): #### run, isData, pu_data, pu_mc):
    # def __init__(self,dataset, pu_data, nn = None, mass_point = None): #### run, isData, pu_data, pu_mc):
        self.run = dataset.run
        self.year = dataset.run[:4]
        self.isData = dataset.isData
        self.luminosity = luminosity
        self.nn_path = nn_path
        self.nn_kwargs = analysis_kwargs
        # nn_sele = None
        self.use_m = "regress_mass" in analysis_kwargs
        # if nn_path is not None:
            # nn = selection.ClassifierModel(nn_path, **analysis_kwargs)
            # self.nn_sele = nn.selection
            # use_m = nn.regress_mass

        self.mass_point = mass_point

        if not self.isData:
            self.pu_data = pu_data
            self.pu_mc   = dataset.getPileup()
            self.pu_weight = PileupWeight.PileupWeight(self.pu_data,self.pu_mc)
        else:
            # self.ext = parsePileUpJSONNoGlobal(self.year)
            parsePileUpJSON2(self.year)

        self.lumimask = LumiMask.LumiMask(dataset)
        self.btag = Btag.Btag('btagDeepB',self.year)
        skim_h = dataset.histograms["skimCounter"]
        self.skimCounter = Counter.ModularCounters().set_skim(skim_h)
#        self.stdsele_histo = Histograms.AnalysisHistograms(self.isData, after_std=True)
#        self.histo = Histograms.AnalysisHistograms(self.isData, use_nn=nn_path is not None, use_m=self.use_m)

#        self.counter = Counter.Counters()
#        self.counter.book(dataset.histograms["skimCounter"])

#        self.book_histograms()

#        self.jetCorrections = JetCorrections.JEC(self.run,self.isData)
#        self.jetvetoMap = JetvetoMap.JetvetoMap(self.year,self.isData)


    def book_histograms(self):
        self.histograms = {}
#        self.histograms.update(self.counter.get())
        self.histograms.update(self.histo.book())
        self.histograms.update(self.stdsele_histo.book())
        
        if not self.isData:
            self.histograms.update(self.make_histogram('pu_orig', [100, 1, 100]))
            self.histograms.update(self.clone_histogram(self.histograms['pu_orig'], 'pu_corr'))
            self.histograms.update(self.clone_histogram(self.histograms['pu_orig'], 'pu_data'))

        print("Booked",len(self.histograms),"histograms")
        self.histograms.update(self.make_histogram('tau', TAU_BINS, variable_binning=True))

    def make_histogram(self, name, bins, variable_binning = False):
        if variable_binning:
            h = (
                hist.Hist.new
                .Var(bins, name="value", label=name)
                .Weight(label=name, name=name)
            )  
        else:
            nbins, binmin, binmax = bins
            h = (
                hist.Hist.new
                .Reg(nbins, binmin, binmax, name="value", label=name,)
                .Weight(label=name, name=name)
            )
        return {name: h}

    def clone_histogram(self, orig, nameClone):
        h = orig.copy(deep=True)
        h.name = nameClone
        h.label = nameClone
        return {nameClone: h}

    def getArrays(self, histo):
        x = []
        axis  = histo.axis()
        edges = axis.edges()
        for i in range(0,len(edges)-1):
            bincenter = edges[i] + 0.5*(edges[i+1]-edges[i])
            x.append(bincenter)
        y = histo.values()
        return ak.from_numpy(np.array(x)),y

    def isolation_count(self, events):
        return events[events.TauIsolation > 0]

    def nn_sele_functional(self, events, mass_hypot=None):
        model = selection.ClassifierModel(self.nn_path, **self.nn_kwargs)
        return model.selection(events, mass_hypot=mass_hypot)

    def process(self, events):

        out = {'unweighted_counter': {}, 'weighted_counter': {}}

        stdsele_histo =  Histograms.AnalysisHistograms(self.isData, after_std=True)
        out.update(stdsele_histo.book())
        fullsele_histo = Histograms.AnalysisHistograms(self.isData, use_nn=self.nn_path is not None, use_m=self.use_m)
        out.update(fullsele_histo.book())
        if not self.isData:
            out.update(self.make_histogram('pu_orig', [100, 1, 100]))
            out.update(self.clone_histogram(out['pu_orig'], 'pu_corr'))
            out.update(self.clone_histogram(out['pu_orig'], 'pu_data'))

        # Weights
        eweight = analysis_tools.Weights(len(events),storeIndividual=True)
        if not self.isData:
            # take into account only the sign of the generator weight
            genw = np.sign(events['genWeight'])
            eweight.add('gen',genw)
            pu = self.pu_weight.getWeight(events.Pileup.nTrueInt)
            eweight.add('pileup',pu)
        events["weight"] = eweight.weight()
        
        # selection cuts
        if self.isData:
            events = events[selection.lumimask(events,self.lumimask)]
        out = stdsele_histo.fill_counters('JSON filter', events, out)

        events = events[selection.triggerSelection(events, self.year)]
        out = stdsele_histo.fill_counters('passed trigger', events, out)

        events = events[selection.METCleaning(events, self.year)]
        out = stdsele_histo.fill_counters('MET cleaning', events, out)

        events = selection.tau_identification(events)
        out = stdsele_histo.fill_counters('Tau event selection', events, out)

        if not self.isData:
            events = selection.label_taus(events)
        else:
            tau = events.Tau
            tau["Genuine"] = False
            tau["Matched"] = False
            events["Tau"] = tau

        events = selection.isolated_electron_veto(events)
        out = stdsele_histo.fill_counters('electron veto', events, out)

        events = selection.isolated_muon_veto(events)
        out = stdsele_histo.fill_counters('muon veto', events, out)

        events = selection.hadronic_pf_jets(events)

        out = stdsele_histo.fill(events, out)
        out = stdsele_histo.fill_counters('after standard selections', events, out)

        events = selection.b_tagged_jets(events)
        out = fullsele_histo.fill_counters('b-jet selection',events, out)

        events = selection.met_cut(events)
        out = fullsele_histo.fill_counters('met selection',events, out)

        events = selection.Rbb_min(events)
        out = fullsele_histo.fill_counters('RBB-min selection',events, out)

        if self.nn_path is not None:
            events = self.nn_sele_functional(events, mass_hypot=self.mass_point)
            if not self.isData:
                out = fullsele_histo.fill_counters('DNN event selection',events, out)

        # Constructing variables
        mu = 0
        if self.isData:
            # mu = getAvgPUNoGlobal(self.ext, events.run, events.luminosityBlock)
            mu = getAvgPU(events.run, events.luminosityBlock)
        else:
            mu = events.Pileup.nTrueInt
        events["mu"] = mu

        # final histograms
        out = fullsele_histo.fill(events,out)

        if not self.isData:
            out['pu_orig'].fill(value=mu)
            out['pu_corr'].fill(value=mu,weight=events["weight"])

        return out

    def postprocess(self, accumulator):
        for key, d in self.skimCounter.get().items():
            accumulator[key].update(d)
        if not self.isData:
            x,y = self.getArrays(self.pu_data)
            accumulator['pu_data'].fill(value=x,weight=y)
        Counter.ModularCounters.print(accumulator, PRINTED_COUNTERS)
        return accumulator

def usage():

    print
    print( "### Usage:  ",os.path.basename(sys.argv[0]),"<multicrab skim>" )
    print

def trim_memory() -> int:
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def name_matches_cat(name, cat, alts):
    return (cat in name) or all(not altcat in name for altcat in alts)

def belongs_to(category, name): # split results into QCDMeasurement and SignalAnalysis, for backwards compatibility
    all_cats = SEPARATE
    qcd_identifiers = ["Inverted", "StdSelections"]
    name_inQCD = any((i in name) for i in qcd_identifiers)
    QCD_matched = not (("QCDMeasurement" in category) ^ name_inQCD) # XNOR of cat matches qcd and name matches qcd
    cond = QCD_matched or (not name_inQCD and any((i in name) for i in ["h_mt_", "h_met"]))
    matched = cond & all([name_matches_cat(name, cat, alternatives) or (cat in ["QCDMeasurement", "SignalAnalysis"]) for cat, alternatives in zip(category, all_cats)])
    return matched

def extend_QCDhist_name(name):
    vars_search = ["pt", "eta"] #NOTE: magic strings
    categories = re.findall("|".join([f"[^_]+{var}" for var in vars_search]), name)
    post = "_".join(categories)
    if len(post) > 0: post = "_"+post
    return post

def navigate_subdirs(key, is_qcd_dir): # for backwards compatibility
    if "vis_" in key: return
    is_mt = "h_mt" in key
    is_met = "h_met" in key
    is_inverted = "Inverted" in key
    if not (is_mt or is_met):
        return
    if is_qcd_dir:
        dirname = "ForQCD"
        sub = "/Normalization"
        if "AfterStdSelections" in key:
            dirname += "Normalization"
            sub += "Mt" if is_mt else "MET"
            sub += "Inverted" if is_inverted else "Baseline"
            sub += "TauAfterStdSelections"
        elif not is_inverted and is_mt:
            dirname += "Measurement"
            sub = "/BaselineTauShapeTransverseMass"
        else:
            dirname = "ForDataDrivenCtrlPlots"
            sub = "/shapeTransverseMass" if is_mt else ""
        if "MatchedTau" in key:
            dirname = "ForDataDrivenCtrlPlots"
            sub = ""
        if "EWKFakeTau" in key:
            dirname += "EWKFakeTaus"
        elif "EWKGenuineTau" in key:
            dirname += "EWKGenuineTaus"
    else:
        if (is_mt or is_met) and any((i in key) for i in ["pt", "eta"]):
            dirname = "qcd_stuff"; sub = ""
        else:
            dirname = "ForDataDrivenCtrlPlots"
            sub = ""
            if "EWKFakeTau" in key:
                dirname += "EWKFakeTaus"
            elif "EWKGenuineTau" in key:
                dirname += "EWKGenuineTaus"

    ROOT.gDirectory.cd(dirname + sub)

    # return the current location
    return dirname + sub

def getFromDict(d, keys):
    return reduce(operator.getitem, keys, d)

def get_legacy_name(dname, hname, counts, is_bg, is_qcd): # for backwards compatibility
    if dname is None:
        return hname
    path = dname.split("/")
    rename = ["h_met_", "h_mt_"]
    if (
        all(var not in hname for var in rename)
        or (
            any(i in dname for i in ["ForDataDrivenCtrlPlots", "qcd_stuff"])
            and "shapeTransverseMass" not in dname
        )
    ):
        if is_bg and not is_qcd:
            if all(i in hname for i in ["h_mt_", "Nominal", "MatchedTau"]): return "shapeTransverseMass"
            else: return hname
        else:
            if "MatchedTau" in hname: return hname
            if all(i in hname for i in ["h_mt_", "Nominal"]): return "shapeTransverseMass"
            else: return hname
    else:
        newname = path[-1]
    count_idx = 0
    if "pt" not in hname and "eta" not in hname: newname += "Inclusive"; count_idx = 3
    elif "pt"  not in hname: newname += "ptInclusive"; count_idx = 1
    elif "eta" not in hname: newname += "etaInclusive"; count_idx = 2
    count = getFromDict(counts, path)
    try:
        count[count_idx] += 1
        getFromDict(counts, path[:-1])[path[-1]] = count
        return newname + str(count[count_idx]-1)
    except TypeError:
        return newname

def write_multicrab_cfg(pseudo_dir, datasets):
    d_names = [d.name for d in datasets]
    def is_dataset(folder):
        return folder in d_names

    def get_dset_block(name):
        return f"[{name}]\n\n"

    subfolders = next(os.walk(pseudo_dir))[1]
    names = filter(is_dataset, subfolders)

    with open(os.path.join(pseudo_dir, "multicrab.cfg"), "w") as f:
        [f.write(get_dset_block(name)) for name in names]
    return

def write_lumi_json(pseudo_dir, datasets):
    def is_data(dataset):
        return dataset.isData
    datasets = filter(is_data, datasets)
    lumi = {d.name: d.lumi for d in datasets}

    names_in_pseudo = next(os.walk(pseudo_dir))[1]
    lumi_in_pseudo = {k: v for k, v in lumi.items() if k in names_in_pseudo}
    lumi_json = json.dumps(lumi_in_pseudo, indent = 4)
    with open(os.path.join(pseudo_dir, "lumi.json"), "w") as f:
        return f.write(lumi_json)

def filter_counters(category, counters):
    no_qcd_cat = [x for x in category if all(x != s for s in QCDSEP)]
    no_qcd_alts = MEASUREMENT_BINS

    common = ['Skim', 'JSON', 'trigger', 'MET cleaning']
    def isin(cat, countername):
        return any(
            i in countername for i in common
        ) or (
            all(
                name_matches_cat(countername, c, alts) for c, alts in zip(cat, no_qcd_alts)
            )
        )

    return dict((key, val) for key, val in counters.items() if isin(no_qcd_cat, key))

def main():
    if DISTRIBUTED:
        print("successfully imported " + ls + "dask" + ns + " and transferred proxy certificate to home directory, will distribute analysis to an HTCondor cluster with " + ss + f"{JOBS}" + ns + " workers")

    # configuration of the neural network
    analysis_kwargs = {}
    nn_path = None
    if len(sys.argv) > 2:
        nn_path = os.path.abspath(sys.argv[2])
        if 'regress' in nn_path.lower():
            analysis_kwargs["regress_mass"] = True
        elif 'param' in nn_path.lower():
            analysis_kwargs["param_mass"] = True
    if len(sys.argv) > 3:
        analysis_kwargs['nn_working_point'] = float(sys.argv[3])

    multicrabdir = os.path.abspath(sys.argv[1])
    if not os.path.exists(multicrabdir) or not os.path.isdir(multicrabdir):
        usage()
        sys.exit()

    year = multicrabdatasets.getYear(multicrabdir)

    starttime = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    # blacklist = ["^ST","^ChargedHiggs","^DYJets"]
    blacklist = []
    whitelist = []

    # whitelist = ["^TT"]

    if DISTRIBUTED:
        n_port = 8786
        print(ss + "initializing condor cluster" + ns)

        # The following config is to ensure workers don't get killed while the local process is
        # stuck in a long function call
        dask.config.set({"timeouts.connect": "700s"})
        clustermanager = HTCondorCluster
        cluster_kwargs = {
            "cores": 1,
            "processes": 1,
            "memory": '4000MB',
            "disk": '1000MB',
            "death_timeout": '300',
            "scheduler_options": {
                'port': n_port,
                'host': socket.gethostname()
                },
            "job_extra": {
                'log': 'dask_job_output.log',
                'output': 'dask_job_output.out',
                'error': 'dask_job_output.err',
                'should_transfer_files': 'YES',
                'when_to_transfer_output': 'ON_EXIT',
                '+JobFlavour': '"longlunch"',
                'environment': f'X509_USER_PROXY={PROXYPATH} PATH=$(ENV(PATH))'
            },
            "extra": ['--worker-port 10000:10100',
                f"--preload \"import os; os.chdir('{os.getcwd()}')\""
            ],
            "env_extra": [f'PYTHONPATH={BASEPATH}:{ANALYSISPATH}:{ANNPATH}',
                            f'PATH={BASEPATH}:{ANALYSISPATH}:{ANNPATH}',
                            f'X509_USER_PROXY={PROXYPATH}',
                            'MALLOC_TRIM_THRESHOLD_=16384',
                            'MALLOC_MMAP_THRESHOLD_=16384'],
        }
        clientmanager = Client

    else:
        clustermanager = nullcontext
        cluster_kwargs = {}
        clientmanager = nullcontext

    with clustermanager(**cluster_kwargs) as cluster:
        
        with clientmanager(cluster) as client:
            if cluster is not None and client is not None:
                print(cluster.job_script())
                client.amm.start()

                print(hs + f"scaling cluster to {JOBS} jobs..." + ns)
                cluster.scale(JOBS)

            datasets = multicrabdatasets.getDatasets(multicrabdir,whitelist=whitelist,blacklist=blacklist)
            pileup_data = multicrabdatasets.getDataPileupMulticrab(multicrabdir)
            lumi = multicrabdatasets.loadLuminosity(multicrabdir,datasets)

            nntext = ""
            if nn_path is not None:
                nntext = "_" + nn_path.split("/")[-1]
            outputdir = os.path.basename(os.path.abspath(multicrabdir))+"_processed"+nntext+starttime

            t0 = time.time()

            # print("Number of cores used",MAX_WORKERS)
            if len(datasets) == 0:
                print("No datasets to be processed")
                print("  whitelist:",whitelist)
                print("  blacklist:",blacklist)
                sys.exit()


            if client is not None:
                print(hs + "Fetched datasets, waiting for workers..." + ns)
                client.wait_for_workers(JOBS, 1800) # timeout if htcondor hasn't matched machines within 30 minutes, something is probably wrong
                print(ss + "All workers assigned!" + ns)
                d_names = [d.name for d in datasets]
                process_first = "TTJets"
                first_dataset = datasets.pop(d_names.index(process_first))
                datasets.insert(0,first_dataset)
                sampler = MemorySampler()
                ms_manager = sampler.sample
            else:
                sampler = None
                ms_manager = nullcontext

            if client is not None:
                job_executor = processor.DaskExecutor(client = client,
                                                      treereduction = 20)
                                                    #   worker_affinity = True)
            else:
                # job_executor = processor.IterativeExecutor() # kept here commented for processor debugging purposes
                job_executor = processor.FuturesExecutor(workers=MAX_WORKERS)

            run = processor.Runner(
                executor = job_executor,
                schema = NanoAODSchema,
                chunksize = CHUNKSIZE,
                maxchunks = MAXCHUNKS,
                processor_compression = None,
                mmap=True
            )

            qcdstr = "QCDMeasurement"

            written_to = []
            with ms_manager("run"):
                for i, d in enumerate(datasets):
                    print("Dataset %s/%s %s"%(i+1,len(datasets),d.name))

                    is_sig = "ChargedHiggs" in d.name
                    is_bg = not is_sig and (not d.isData)
                    
                    t00 = time.time()

                    samples = {d.name: d.getFileNames()}

                    processor_instance = Analysis(d,pileup_data,lumi, nn_path = nn_path, **analysis_kwargs)

                    result = run(
                        samples,
                        "Events",
                        processor_instance = processor_instance,
                    )

                    # result = processor.run_uproot_job(
                    #     samples,
                    #     "Events",
                    #     Analysis(d,pileup_data,lumi,nn = None),
                    #     job_executor,
                    #     executor_kwargs,
                    #     chunksize = CHUNKSIZE,
                    #     maxchunks = MAXCHUNKS
                    # )

                    t01 = time.time()
                    dt0 = t01-t00
                    print("Processing time %s min %s s"%(int(dt0/60),int(dt0%60)))

                    sep_listed = [arr for arr in SEPARATE]
                    # if nn_path is None: # ignore the nn categorizations
                    #     sep_listed.pop(-1)
                    categories = ak.cartesian(sep_listed, axis=0)
                    for cat in categories:
                        cat = cat.tolist()
                        cat_str = "_".join(cat)
                        # print(cat_str)
                        is_qcd = qcdstr in cat_str
                        if is_qcd:
                            if "ChargedHiggs" in d.name: continue
                            cat_outputdir = qcdstr + "_" + outputdir + cat_str.replace(qcdstr, "")
                        else:
                            cat_outputdir = outputdir + cat_str

                        if not os.path.exists(cat_outputdir):
                            os.mkdir(cat_outputdir)
                            written_to.append(cat_outputdir)

                        subdir = os.path.join(cat_outputdir,d.name)

                        if not os.path.exists(subdir):
                            os.mkdir(subdir)
                            os.mkdir(os.path.join(subdir,"res"))

                        fOUT = ROOT.TFile.Open(os.path.join(subdir,"res",f"histograms-{d.name}.root"),"RECREATE")
                        fOUT.cd()
                        fOUT.mkdir("configInfo")
                        fOUT.cd("configInfo")

                        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                        now = datetime.datetime.now()
                        m = "produced: %s %s"%(days[now.weekday()],now)
                        timestamp = ROOT.TNamed(m,"")
                        timestamp.Write()

                        gitCommit = aux.execute("git rev-parse HEAD")[0]
                        gc = "git commit:"+gitCommit
                        gitcommit = ROOT.TNamed(gc,gitCommit)
                        gitcommit.Write()

                        h_lumi = ROOT.TH1D("lumi","",1,0,1)
                        h_lumi.SetBinContent(1,d.lumi)
                        h_lumi.Write()

                        h_isdata = ROOT.TH1D("isdata","",1,0,1)
                        h_isdata.SetBinContent(1,int(d.isData))
                        h_isdata.Write()

                        fOUT.cd()
                        fOUT.mkdir("analysis")

                        h_counters = {}
                        fOUT.cd("analysis")
                        fake_cats = ["", "EWKGenuineTaus", "EWKFakeTaus"]
                        for postfix in fake_cats:
                            fOUT.mkdir("analysis/ForDataDrivenCtrlPlots" + postfix)
                            h_counters["ForDataDrivenCtrlPlots" + postfix] = {"shapeTransverseMass": [0, 0, 0, None]}
                        if is_qcd:
                            for postfix in fake_cats:
                                fOUT.mkdir("analysis/ForDataDrivenCtrlPlots" + postfix + "/shapeTransverseMass")
                            dirs = QCD_DIRS
                            for dir in dirs:
                                h_counters[dir] = {}
                                ROOT.gDirectory.mkdir(dir)
                                ROOT.gDirectory.cd(dir)
                                for subd in dirs[dir]:
                                    ROOT.gDirectory.mkdir(subd)
                                    h_counters[dir][subd] = [0, 0, 0, None] # binned_both, binned_eta, binned_pt, inclusive
                                fOUT.cd("analysis")
                        else: ROOT.gDirectory.mkdir("qcd_stuff")
                        fOUT.cd()


                        for key in result.keys():

                            iscounter = 'counter' in key
                            in_configs = iscounter or key in ['pu_orig','pu_data','pu_corr']

                            # only write the histo if it matches the category or it is part of the config
                            if not belongs_to(cat, key) and not in_configs:
                                continue

                            dname = None
                            if in_configs:
                                fOUT.cd("configInfo")
                            else:
                                fOUT.cd("analysis")
                                dname = navigate_subdirs(key, is_qcd)
                            # print(key)
                            # print(dname)

                            # newkey = key
                            # if "h_mt" in key:
                            #     newkey = "TransverseMass_ttRegion" + extend_QCDhist_name(key)
                            newkey = get_legacy_name(dname, key, h_counters, is_bg, is_qcd)

                            if iscounter:
                                h = filter_counters(cat, result[key])
                            else: h = result[key]
                            histo = hist2root.convert(h)
                            histo.SetName(newkey)
                            histo.Write()
                            fOUT.cd()
                        fOUT.Close()
                        dt1 = time.time()-t01
                    print("Converting hist2root time %s min %s s"%(int(dt1/60),int(dt1%60)))

            if sampler is not None:
                p = sampler.plot(figsize=(15,10), grid=True)
                plt.savefig("dask_memorysample")

    # write multicrab.cfg files to each pseudo-multicrab folder
    for folder in written_to:
        write_multicrab_cfg(folder, datasets)
        write_lumi_json(folder, datasets)
    dt = time.time()-t0

    # for boson in diboson:
    #     try:
    #         hist.plot1d(boson, stack= True, fill_opts= {"color": "blue"})
    #     except:
    #         pass    
    # for t in tt:
    #     hist.plot1d(t, stack= True, fill_opts={"color": "purple"})
    # for wjet in wjets:
    #     hist.plot1d(wjet, stack= True, fill_opts={"color": "red"})
    # for single in single_t:
    #     hist.plot1d(wjet, stack= True, fill_opts={"color": "green"})
    # for jet in dyjets:
    #     hist.plot1d(jet, stack= True, fill_opts={"color": "cyan"})
    # for tau in tau_run:
    #     hist.plot1d(tau, stack= True, fill_opts={"color": "orange"})
    

    # plt.ylim(10**(-3), 10**6)
    # plt.yscale("log")
    # plt.title("Tau pt for Rtau > 0.75 and jet selection. Diboson")
    # plt.savefig("/home/joonapankkonen/plots/Tau_pt_plot_Dibosons")
    # plt.show()
    print("Total processing time %s min %s s"%(int(dt/60),int(dt%60)))
    print("output in",outputdir)

    return

if __name__ == "__main__":
    main()
    #os.system("ls -lt")
    #os.system("pwd")
