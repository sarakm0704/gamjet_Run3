# from builtins import breakpoint
# from this import d
import numpy as np
import awkward as ak
import os, re
import sys
import multiprocessing
import pathlib
import time
import pickle

# add framework path and analysis path to $PATH in order to access utilities
basepath_re = re.compile("(?P<basepath>\S+/NanoAnalysis)/")
match = basepath_re.search(os.getcwd())
if match:
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Framework/python"))
    sys.path.append(os.path.join(match.group("basepath"),"NanoAODAnalysis/Hplus2taunuAnalysis"))

import crosssection
import coffea
import uproot3
import multicrabdatasets
from coffea import processor
import coffea.hist as hist
from coffea.nanoevents import NanoAODSchema
import time
import Hplus2taunuSelection
import Btag
import Counter
import TransverseMass
import Hplus2taunuHistograms as Histograms
import glob

#import PyROOT before tensorflow!
import ROOT
import tensorflow as tf
import tensorflow.io as tfio
import tensorflow_probability as tfp

# by default this gets overwritten by disCo, where the mass parameter is propagated from the previous signal sample to the background
# If no_copy_mass is specified (running disCo.py), then the mass parameter will be sampled uniformly from MDATA
MDATA = np.array([170,175,200,220,250,400,500,700,800,2500,3000])

# transverse mass bins for planing
MT_BINS = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,280,300,320,340,360,380,400,450,500,550,600,700,800,1000,1500,2000,3000],
                    dtype=float) #TODO: enable user input for bin edge specification or otherwise do it smarter
EPS = 1e-3
tf.random.set_seed(1234)
TF_OP_SEED = 8765
DEFAULTVARS = "tau_pt,pt_miss,R_tau,bjet_pt,btag,mt,dPhiTauNu,dPhitaub,dPhibNu,mtrue"
FULLVARS = "tau_pt,pt_miss,R_tau,btag_all,dPhiTauNu,dPhiTauJets,dPhiNuJets,dPhiAllJets,Jets_pt,mt,mtrue"
COLUMN_KEYS = {
    "btag_all": ['btag'] * 4,
    "dPhiAllJets": ['dPhiJets'] * 6,
    "Jets_pt": ['jet_pt'] * 4,
    "dPhiTauJets": ['dPhiTauJet'] * 4,
    "dPhiNuJets": ['dPhiNuJet'] * 4,
}
# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
ss = "\033[92m"
ns = "\033[0;0m"
ts = "\033[1;34m"
hs = "\033[0;35m"   
ls = "\033[0;33m"
es = "\033[1;31m"
cs = "\033[0;44m\033[1;37m"


def load_input_vars(savedir):
    try:
        with open(savedir + "/input_vars.txt", 'r') as f:
            input_vars = [l.rstrip() for l in f]
    except FileNotFoundError:
        try:
            with open(savedir + "/input_variables.txt", 'r') as f:
                input_vars = [l.rstrip() for l in f]
        except FileNotFoundError:
            raise Exception(f"no input variable file found at {savedir}/input_vars.txt or input_variables.txt")
    return input_vars

def load_true_mass_samples(datadir, mass_idx, k):
    datasets = []
    for fold in range(k):
        dir = os.path.join(datadir, f"/*ChargedHiggs*/data/{fold}/final_*/")
        sig_dirs = glob.glob(dir)

        spec_dir = glob.glob(datadir + f"/*ChargedHiggs*/data/{fold}/elementSpec")[0]
        with open(spec_dir, 'rb') as in_:
            element_spec = pickle.load(in_)

        def map_fn(x, y):
            return x[...,mass_idx]

        choice = tf.data.Dataset.range(len(sig_dirs)).repeat()
        data = [tf.data.Dataset.load(sig, element_spec=element_spec) for sig in sig_dirs]
        datasets.append(tf.data.Dataset.choose_from_datasets(data, choice, stop_on_empty_dataset=False).map(map_fn))

    return datasets

def get_ds_len(dataset): # TODO: recursive tuples and dictionaries as elements are currently unsupported
    if type(dataset.element_spec) is tuple:
        def get_batch_len(elem):
            return elem[0].shape[0]
    elif type(dataset.element_spec) is dict:
        key = dataset.element_spec.keys()[0]
        def get_batch_len(elem):
            return elem[key].shape[0]
    else:
        def get_batch_len(elem):
            return elem.shape[0]
    n = 0
    dataset = dataset.batch(2**15)
    for elem in dataset:
        n += get_batch_len(elem)
    return n

# function to load k-folded datasets, returns a list of tf.data.Datasets
def load(datadir,
    k,
    bg_dataset_sizes,
    testset = False,
    sig_hists = None, 
    bkg_hists = None,
    mt_idx=None,
    marginalized=False,
    energy="13",
    exclude = None
    ):

    def interleave_signals(sig_datasets):
        n = len(sig_datasets)
        choice_ds = tf.data.Dataset.range(n).repeat()
        return tf.data.Dataset.choose_from_datasets(sig_datasets, choice_ds)

    def shuffle_backgrounds(bg_datasets):
        n = len(bg_datasets)
        ds_lens = [get_ds_len(ds) for ds in bg_datasets]
        sampling_arr = tf.random.shuffle(tf.repeat(tf.range(n, dtype=tf.int64), ds_lens), seed = TF_OP_SEED)
        choice_ds = tf.data.Dataset.from_tensor_slices(sampling_arr)
        return tf.data.Dataset.choose_from_datasets(bg_datasets, choice_ds), ds_lens

    planing = sig_hists != None
    if planing:
        edges = bkg_hists[0]["bins"].astype(np.float32)
        total_sig_counts = [sum([sum(hist["values"]) for hist in hists.values()]) for hists in sig_hists]
        total_sig_dist = [np.sum(np.stack([hist["values"] for hist in hists.values()], axis=0), axis=0) for hists in sig_hists]

        def match_with_weights(path, idx, fold):

            bg_dist = np.array(bkg_hists[fold]["values"])
            bg_count = np.sum(bg_dist)
            normalized_bg_dist = bg_dist / bg_count

            dataset_name = path.split("/")[idx]
            
            if dataset_name not in sig_hists[fold].keys():
                sigma = crosssection.backgroundCrossSections.crossSection(dataset_name, energy)
                N0 = bg_dataset_sizes[fold][dataset_name]
                if sigma == None:
                    msg = f"No cross-section data found for dataset named {dataset_name}!"
                    raise Exception(msg)
                return tf.zeros_like(edges) + sigma / N0
            
            if not marginalized: # planing for each individual signal dataset
                dataset_sig_count = sum(sig_hists[fold][dataset_name]["values"])
                sig_dist = np.array(sig_hists[fold][dataset_name]["values"]) + EPS
            else: # planing for the combined signal dataset of samples of different masses
                dataset_sig_count = total_sig_counts[fold]
                sig_dist = total_sig_dist[fold] + EPS
            
            normalized_sig_dist = sig_dist / dataset_sig_count
            return tf.constant(normalized_bg_dist / normalized_sig_dist, dtype = tf.float32)

    else: # no planing, only need to weight different backgrounds according to cross-sections
        edges = [0.,1.,] #find_bins requires atleast two edges, the weights are all set to the same value.
        def match_with_weights(path, idx, fold):
            dataset_name = path.split("/")[idx]
            if "ChargedHiggs" not in dataset_name: # get background class weights according to interaction cross-sections
                sigma = crosssection.backgroundCrossSections.crossSection(dataset_name, energy)
                N0 = bg_dataset_sizes[fold][dataset_name]
                assert N0 > 0
                if sigma == None:
                    msg = f"No cross-section data found for dataset named {dataset_name}!"
                    raise Exception(msg)
                return tf.zeros_like(edges) + sigma / N0

            # for the signal samples with no planing, we want to have equal weights for all the samples
            return tf.ones_like(edges)

    targets = datadir + "/*/data"
    datasets = []

    edges = tf.constant(edges)

    def add_weights(ds, binned_weights): # function to map bin weights onto the event samples as sample weights
        return ds.map(lambda x, y:
            (x,
            y,
            tf.gather(binned_weights, tfp.stats.find_bins(x, edges=edges, dtype=tf.int64, extend_upper_interval=True, extend_lower_interval=True)[mt_idx])
            ))

    for fold in range(k):
        spec_dir = glob.glob(targets + f"/{fold}/elementSpec")[0]
        if testset:
            fold_targets = glob.glob(targets + f"/{fold}/test/")
            nameidx = -6
        else:
            fold_targets = glob.glob(targets + f"/{fold}/")
            nameidx = -5
        with open(spec_dir, 'rb') as in_:
            element_spec = pickle.load(in_)

        # drop directory names that are in the list to exclude
        if exclude:
            for substr in exclude:
                fold_targets = [tgt for tgt in fold_targets if substr not in tgt]

        # load data from all datasets in the list of filenames fnames and combine into single dataset
        # this is called to combine the dataset shards of a single crab dataset
        def load_and_merge_data(fnames): 
            n = len(fnames)
            choice_ds = tf.data.Dataset.range(n).repeat()
            datasets = [tf.data.experimental.load(fname, element_spec) for fname in fnames]
            dataset = tf.data.Dataset.choose_from_datasets(datasets, choice_ds, stop_on_empty_dataset=False)
            return dataset

        fnames = list(map(lambda x: glob.glob(x + "final_*/"), fold_targets)) # get all subdirectories containing parts of the dataset
        f_weights = list(map(lambda fnames_ds: match_with_weights(fnames_ds[0], nameidx, fold), fnames)) # match each name with binned weights

        # normalize bakground weights s.t. the average weight for an event will be 1:
        bg_mask = np.array([int(not ("ChargedHiggs" in fname[0])) for fname in fnames])
        bg_unnormalized_w = [weights[0] for i, weights in enumerate(f_weights) if bg_mask[i] == 1]

        fold_datasets = list(map(load_and_merge_data, fnames)) # load datasets
        fold_datasets = list(map(lambda x: add_weights(*x), zip(fold_datasets, f_weights))) # map weights from bins onto events

        #split into collection of signal datasets and bg datasets
        sig_datasets = [ds for i, ds in enumerate(fold_datasets) if "ChargedHiggs" in fnames[i][0]]
        bg_datasets = [ds for i, ds in enumerate(fold_datasets) if "ChargedHiggs" not in fnames[i][0]]

        sig_ds = interleave_signals(sig_datasets) # interleave elements from the different signal datasets to show the nn an equal amount for each type
        bg_ds, bg_lens = shuffle_backgrounds(bg_datasets) # sample from background datasets in proportion to how many events passed the preliminary selection

        bg_len_tot = sum(bg_lens)
        bg_crosssection_tot = np.sum(np.array(bg_unnormalized_w) * np.array(bg_lens))
        scale_factor = tf.cast(bg_len_tot / bg_crosssection_tot, tf.float32)
        bg_ds = bg_ds.map(lambda x, y, w: (x, y, w * scale_factor))

        datasets.append((sig_ds, bg_ds))
    return datasets

def load_hists(datadir, k, energy='13'):
    sig_flag = "ChargedHiggsToTauNu"
    sig_targets = glob.glob(datadir + "/*" + sig_flag + "*/mt_hists.root")
    bkg_targets = set(glob.glob(datadir + "/*/mt_hists.root")) - set(sig_targets)

    bkg_hist = []
    dataset_counts = []
    sig_hists = []
    for i in range(k):
        bkg_hist.append({"bins": [], "values": []})
        dataset_counts.append({})
        for fname in bkg_targets:
            f = uproot3.open(fname)
            name = fname.split("/")[-2]
            dataset_counts[i][name] = int(f["t"][f"ds_{i}_size"].array()[0])
            sigma = crosssection.backgroundCrossSections.crossSection(name, energy)
            hist = f[f"mt_hist_{i}"]
            if bkg_hist[i]["bins"] == []:
                bkg_hist[i]["bins"] = hist.edges
                bkg_hist[i]["values"] = hist.values * sigma / dataset_counts[i][name]
            else:
                bkg_hist[i]["values"] = bkg_hist[i]["values"] + hist.values

        sig_hists.append({})
        for fname in sig_targets:
            f = uproot3.open(fname)
            hist = f[f"mt_hist_{i}"]
            sig_hists[i][fname.split("/")[-2]] = {
                "bins": hist.edges,
                "values": hist.values
            }

    return sig_hists, bkg_hist, dataset_counts

# class to write datasets from multicrab to tf.Datasets
class Writer(processor.ProcessorABC):
    def __init__(self, dataset, m_data, vars, k, outpath=""):
        if dataset is not None:
            self.year = dataset.run[:4]
            self.k = k
            self.skim_counter = Counter.Counters()
            self.skim_counter.book(dataset.histograms["skimCounter"])
        else: k = 0

        self.vars = vars

        self.counter = processor.defaultdict_accumulator(int)

        self.outpath = outpath
        self.m_data = m_data
        self.len_ms = len(m_data)
        self.mt_hists = processor.dict_accumulator(dict(
            [(f"{i}", hist.Hist("Counts", hist.Bin("mt", "transverse mass", MT_BINS))) for i in range(k)]
        ))
        if dataset is not None:
            self.book_histograms()

    @staticmethod
    def get_tf_inputs(events, names, mdata):

        len_ms = len(mdata)

        leading_tau = events.Tau[:,0]
        leading_jet = events.Jet[:,0]
        jet2 = events.Jet[:,1]
        jet3 = events.Jet[:,2]

        def get_taupt():
            return tf.expand_dims(leading_tau.pt.to_numpy(), axis=1)

        def get_pt_miss():
            return tf.expand_dims(events.MET.pt.to_numpy(), axis=1)

        def get_Rtau():
            return tf.expand_dims(leading_tau.leadTkPtOverTauPt, axis=1)

        def get_tau_mt():
            return tf.expand_dims(tf.constant(np.sqrt(leading_tau.mass**2 + leading_tau.pt**2)), axis=1)

        def get_mt():
            ak_mt = TransverseMass.reconstruct_transverse_mass(leading_tau, events.MET)
            return tf.expand_dims(tf.constant(ak_mt), axis=1)

        def get_dPhiTauNu():
            return tf.expand_dims(tf.constant(leading_tau.delta_phi(events.MET).to_numpy()), axis=1)

        def get_dPhiTauJets():
            dPhis = ak.fill_none(ak.pad_none(leading_tau.delta_phi(events.Jet), 4, axis=1),0)[:,:4]
            return tf.constant(dPhis, dtype=tf.float32)

        def get_dPhiNuJets():
            dPhis = ak.fill_none(ak.pad_none(events.MET.delta_phi(events.Jet), 4, axis=1),0)[:,:4]
            return tf.constant(dPhis, dtype=tf.float32)

        def get_btag_all():
            padded_btags = np.clip(ak.fill_none(ak.pad_none(events.Jet.btagCSVV2, 4, axis=1), 0)[:,:4],0.,None)
            return tf.constant(padded_btags, dtype=tf.float32)

        def get_jet_pt_all():
            padded_pts = ak.fill_none(ak.pad_none(events.Jet.pt, 4, axis=1), 0)[:,:4]
            return tf.constant(padded_pts, dtype=tf.float32)

        def get_dPhiJets(): # return the angles between first four most energetic jets in the events, padded with 0s
            padded_jets = ak.pad_none(events.Jet, 4, axis=1)[:,:4]
            dPhis = []
            for i in range(4):
                for j in range(i+1,4):
                    dPhis.append(tf.constant(ak.fill_none(padded_jets[:,i].delta_phi(padded_jets[:,j]), 0), dtype=tf.float32))
            return tf.stack(dPhis, axis=1)

        def get_mtrue():
            s_template = leading_tau.pt.to_numpy().shape + (1,)
            Hplusidx = (events.GenPart.pdgId == 37) | (events.GenPart.pdgId == -37)
            mtrue = events.GenPart.mass[Hplusidx]
            if ak.size(ak.flatten(mtrue)) == 0: # data is background, insert random mass from true distribution
                mtrue = tf.constant(mdata[np.random.randint(len_ms, size=s_template)], dtype=tf.float32)
            else: # data is signal
                mtrue = tf.expand_dims(mtrue[:,0], axis=1)
            return mtrue

        def get_rand_m():
            s_template = leading_tau.pt.to_numpy().shape + (1,)
            return tf.constant(mdata[np.random.randint(len_ms, size=s_template)], dtype=tf.float32)

        def get_empty():
            s_template = leading_tau.pt.to_numpy().shape + (0,)
            return tf.zeros(s_template, dtype=tf.float32)

        if "BJet" in events.fields:
            leading_b = events.BJet[:,0]

            def get_btag():
                return tf.expand_dims(tf.constant(np.clip(events.BJet[:,0].btagCSVV2, 0., None), dtype=tf.float32), axis=1)

            def get_dPhibNu():
                return tf.expand_dims(tf.constant(leading_b.delta_phi(events.MET).to_numpy()), axis=1)

            def get_dPhiTaub():
                return tf.expand_dims(tf.constant(leading_tau.delta_phi(leading_b).to_numpy()), axis=1)

            def get_bjet_pt():
                return tf.expand_dims(leading_b.pt.to_numpy(), axis=1)

            def get_dPhibJet1():
                return tf.expand_dims(leading_b.delta_phi(leading_jet), axis=1)

            def get_dPhibJet2():
                return tf.expand_dims(leading_b.delta_phi(jet2), axis=1)

            def get_dPhibJet3():
                return tf.expand_dims(leading_b.delta_phi(jet3), axis=1)


        funcs = {
            "tau_pt": get_taupt,
            "pt_miss": get_pt_miss,
            "R_tau": get_Rtau,
            "tau_mt": get_tau_mt,
            "mt": get_mt,
            "dPhiTauNu": get_dPhiTauNu,
            "dPhiTauJets": get_dPhiTauJets,
            "dPhiNuJets": get_dPhiNuJets,
            "mtrue": get_mtrue,
            "rand_m": get_rand_m,
            "btag_all": get_btag_all,
            "dPhiAllJets": get_dPhiJets,
            "Jets_pt": get_jet_pt_all,
            "_": get_empty
        }
        if "BJet" in events.fields:
            funcs.update({
            "bjet_pt": get_bjet_pt,
            "dPhiTaub": get_dPhiTaub,
            "dPhibNu": get_dPhibNu,
            "btag": get_btag,
            "dphibJ1": get_dPhibJet1,
            "dphibJ2": get_dPhibJet2,
            "dphibJ3": get_dPhibJet3,
            })
        
        outs = [funcs[name]() for name in names]
        return tf.concat(outs, axis=1)

    def book_histograms(self):
        self.histograms = {}
        self.histograms.update(self.skim_counter.get())
        self.histograms.update(event_counter = self.counter)
        self.histograms.update(mt_hists = self.mt_hists)
        self._accumulator = processor.dict_accumulator(self.histograms)

    @property
    def accumulator(self):
        return self._accumulator

    # calculate NN input variables and convert from ak.Arrays to tf.Dataset
    @staticmethod
    def convert(in_vars, mdata, events, inference=False):
        """
        If writing a neural network for a new analysis, you can subclass Writer and overwrite this method
        and calculate the desired input variables here
        """

        if inference:
            in_vars = [v if v != "mtrue" else "rand_m" for v in in_vars]
            x = Writer.get_tf_inputs(events, in_vars, mdata)
            return x
        else:
            x = Writer.get_tf_inputs(events, in_vars, mdata)

            y = ak.any((events.GenPart.pdgId == 37) | (events.GenPart.pdgId == -37), axis = 1) # was there a H+ in the simulation? 1 or 0    
            y = tf.constant(y)

            ds = tf.data.Dataset.from_tensor_slices((x, y))
            return ds



    # filter events on relaxed selection in order to obtain a clean training dataset, k-fold dataset by event id and save to file as tensorflow dataset
    def process(self, events):

        out = self.accumulator.identity()

        self.skim_counter.setAccumulatorIdentity(out)
        self.skim_counter.setSkimCounter()

        # relaxed selection cuts
        events = Hplus2taunuSelection.tau_identification(events, 50, 2.5)
        events = events[events.TauIsolation] # tau_identification keeps both nominal and fake_tau with the TauIsolation flag specifying the category
        events["Tau"] = events.Tau[events.Tau.isolated] # drop the taus that passed inverted selection (applies only to events with > 1 Tau)
        events = Hplus2taunuSelection.isolated_electron_veto(events)
        events = Hplus2taunuSelection.isolated_muon_veto(events)
        events = Hplus2taunuSelection.hadronic_pf_jets(events)
#        events = Hplus2taunuSelection.b_tagged_jets(events, wp='loose')

        for fold in range(self.k):
            savetime = time.strftime("%S%M%H%d%m%y", time.localtime())
            savepath = self.outpath + f"/data/{fold}/final_{savetime}"

            included = events[events.event % self.k != fold]

            mt_values = TransverseMass.reconstruct_transverse_mass(included.Tau[:,0], included.MET)
            out["mt_hists"][f"{fold}"].fill(mt=mt_values)
            out["event_counter"][f"{fold}"] += len(included)


            tf_dataset_train = self.__class__.convert(self.vars, self.m_data, included)
            tf.data.experimental.save(tf_dataset_train, savepath)

            excluded = events[events.event % self.k == fold]
            tf_dataset_test = self.__class__.convert(self.vars, self.m_data, excluded)
            tf.data.experimental.save(tf_dataset_test, self.outpath + f"/data/{fold}/test/final_{savetime}")
            
            spec_path = self.outpath + f"/data/{fold}/elementSpec"
            with open(spec_path, 'wb') as f:
                pickle.dump(tf_dataset_train.element_spec, f)
        return out
    
    def postprocess(self, accumulator):
        histpath = self.outpath + f"/mt_hists.root"
        if os.path.exists(histpath):
            os.remove(histpath)
        outputfile = uproot3.create(histpath)
        outputfile["t"] = uproot3.newtree(
            {f"ds_{i}_size": int for i in range(self.k)}
        )
        for i in range(self.k):
            outputfile[f"mt_hist_{i}"] = hist.export1d(accumulator["mt_hists"][f"{i}"])
        outputfile["t"].extend({f"ds_{i}_size": [accumulator["unweighted_counter"]["Skim: All events"] // 2] for i in range(self.k)})
        outputfile.close()
        return accumulator


def main():
    np.random.seed(123)


    NCORES = max(multiprocessing.cpu_count() // 2,1)
    t0 = time.time()

    if len(sys.argv) > 3:
        folds = int(sys.argv[3])
    else: folds = 2

    if len(sys.argv) > 4:
        invars = str(sys.argv[4])
    else:
        # invars = DEFAULTVARS
        invars = FULLVARS
    invars = invars.split(",")


    outdir = os.path.abspath(sys.argv[2])
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # write a text file containing the input variables
    # this makes it easier to later use the saved dataset in training where it may
    # be necessary to specify the input variables for their unique preprocessing
    with open(outdir + "/input_variables.txt", "w") as f:
        for var in invars:
            f.write(var + "\n")

    datadir = os.path.abspath(sys.argv[1])
    datasets = multicrabdatasets.getDatasets(datadir)

    m_data = MDATA

    for i, d in enumerate(datasets):
        if d.isData: continue # we must train on simulation data
        suboutpath = os.path.join(outdir, d.name)
        if not os.path.exists(suboutpath):
            os.mkdir(suboutpath)
            os.mkdir(os.path.join(suboutpath,"data"))
        
        # ad-hoc fix to continue interrupted script, comment out if starting from the beginning:
   #     if "ChargedHiggsToTauNu" in d.name:
   #         print(f"dataset {d.name} already saved, skipping...")
   #         continue
   #     if "DYJetsToLL" in d.name:
  #          print(f"dataset {d.name} already saved, skipping...")
  #          continue
 #       if "ST_" in d.name:
#            print(f"dataset {d.name} already saved, skipping...")
#            continue


        samples = {d.name: d.getFileNames()}
        processor.run_uproot_job(
            samples,
            "Events",
            Writer(d,m_data,invars,folds,outpath=suboutpath), ####d.run,d.isData,pileup_data,d.getPileup()),
            processor.iterative_executor,
            {"schema": NanoAODSchema},
            chunksize = 1e6
        )

        dt = time.time()-t0

        print("Processing time %s min %s s"%(int(dt/60),int(dt%60)))
        print("output in",outdir)
    return

if __name__ == "__main__":
    main()
