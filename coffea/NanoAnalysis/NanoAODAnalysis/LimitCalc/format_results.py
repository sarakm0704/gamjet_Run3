import os
from argparse import ArgumentParser
import ROOT

def parse_args():
    parser = ArgumentParser(desc = "placeholder")

    parser.add_argument("result_dir", help="")

    parser.add_argument("outname", help="")

    parser.add_argument("-voi", "--variable_of_interest", dest = "voi", help="", default="mt")

    parser.add_argument("-w", "--reweight", dest="reweight" help="")

    args = parser.parse_args()

def get_channels(resdir):
    return

def get_dataobs(resdir):
    to_merge = [f for f in os.listdir(resdir) if "Tau_Run" in f]

    return

def get_procs(resdir):
    folders = os.listdir(resdir)
    procs = [f for f in folders if "Tau_Run" not in f]
    return procs

def write_channels(outfile):
    return

def get_histos(resultdir, channels, procs):


    def load_proc(p):
        f = ROOT.TFile(os.path.join(resultdir, p, "results", "histograms.root"), "read")
        proc_histos = f.Get("analysis")

        f.Close()


    return {}

def write_channels(tfile, channels):
    return

def write_histos(tfile, histos):
    return

def main(args):

    outfile = ROOT.TFile(args.outname, "RECREATE")
    h_prefix = f"h_{args.voi}"

    channels = get_channels(args.result_dir)
    processes = get_procs(args.result_dir)

    write_channels(outfile)
    histos = get_histos(args.result_dir, channels, processes)

    write_channels(outfile, channels)
    write_histos(outfile, histos)

    outfile.Close()


if __name__ == "__main__":
    args = parse_args()
    main(args):