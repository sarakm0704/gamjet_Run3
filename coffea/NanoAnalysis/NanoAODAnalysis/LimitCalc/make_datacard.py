from argparse import ArgumentParser, RawTextHelpFormatter
import textwrap
import json
import ROOT
import numpy as np

def parse_args():
    # set default options
    ROOTFILE    = ""
    SIGNALNAMES = ""
    BKGNAMES    = ""
    DATANAME    = ""
    SYSTUNCERT  = ""
    CHANNELS    = ""

    parser = ArgumentParser(
        description= textwrap.dedent("""\
        This script takes an input rootfile in the format described below and generates a datard
        to be input into Combine for analysis of limits. Different processes are
        required to be input as arguments in order to specify if they should be treated
        as signal or background. Channel names (Combine bins) to be used in limit are also
        required as an argument.

        The input root file must be formatted s.t. it contains a directory for each bin(channel).
        Each bin directory must then contain histograms named by their process. Shape uncertainty
        histograms must be named like [process_name]_[uncertainty_name].

        The input uncertainty JSON file must contain a nested dictionary of uncertainties. The keys
        of this dict will be used as the uncertainty name. Each key(name) corresponds
        with a "type" field containing the type of the uncertainty ("lnN", "shape", etc.) and multiple
        channels, which then contain all the different processes and their values. Missing
        channels and processes will be interpreted as 1 if the type of the uncertainty is "shape".

        Example datacard:

        ####################################################################################################################################
        
        imax 2
        jmax 2
        kmax 7
        ---------------
        shapes * * shapes_file.root $CHANNEL/$PROCESS  $CHANNEL/$PROCESS_$SYSTEMATIC
        ---------------
        bin ejets mujets
        observation 4734 6448
        ------------------------------
        bin             ejets      ejets   ejets   mujets     mujets    mujets
        process         tprime600  top     ewk     tprime600  top       ewk
        process         0          1       2       0          1         2
        rate            227        4048    760     302        5465.496783      1098.490939
        --------------------------------
        lumi     lnN    1.10        -       -     1.10        -         -
        bgnortop lnN    -          1.114    -     -          1.114      -
        bgnorewk lnN    -           -      1.5    -           -        1.5
        eff_mu   lnN    -           -       -     1.03       1.03      1.03
        eff_e    lnN    1.03       1.03    1.03    -          -         -
        jes    shape    1           1       1      1          1         1   uncertainty on shape due to JES uncertainty
        btgsf  shape    1           1       1      1          1         1   uncertainty on shape due to b-tagging scale factor uncertainty

        ###################################################################################################################################

        The uncertainty JSON to generate the above datacard 
        {
            "lumi": {
                "type": "lnN",
                "ejets": {
                    "tprime600": "1.10"
                },
                "mujets": {
                    "tprime600": "1.10"
                }
            },
            "bgnortop": {
                "type": "lnN",
                "ejets": {
                    "top": "1.114"
                },
                "mujets": {
                    "top": "1.114"
                }
            },
            "bgnorewk": {
                "type": "lnN",
                "ejets": {
                    "ewk": "1.5"
                },
                "mujets": {
                    "ewk": "1.5"
                }
            },
            "eff_mu": {
                "type": "lnN",
                "mujets": {
                    "tprime600": "1.03",
                    "top": "1.03",
                    "ewk": "1.03"
                }
            },
            "eff_e": {
                "type": "lnN",
                "ejets": {
                    "tprime600": "1.03",
                    "top": "1.03",
                    "ewk": "1.03"
                }
            },
            "jes": {
                "type": "shape"
            },
            "btgsf": {
                "type": "shape"
            }
        }

        Example command to generate the above datacard:

        python make_datacard.py datacard.txt -r shapes_file.root -u test.json -s tprime600 -b top,ewk -c ejets,mujets"""),
        formatter_class = RawTextHelpFormatter
    )

    parser.add_argument("out_name", help="name of the datacard file to be generated")

    parser.add_argument("-s", "--signalNames", dest="sig_names", default=SIGNALNAMES,
                        help="list of names for signal processes separated by commas: [default: %s]" % SIGNALNAMES)

    parser.add_argument("-b", "--backgroundNames", dest="bkg_names", default=BKGNAMES,
                        help="list of names for background processes separated by commas: [default: %s]" % BKGNAMES)

    parser.add_argument("-u", "--uncertainties", dest="uncertainties_file", default=SYSTUNCERT,
                        help=":name of the JSON file containing the uncertainty dictionary [default: %s]" % SYSTUNCERT)

    parser.add_argument("-r", "--root_file", dest="rootfile", default=ROOTFILE,
                        help="path to the analysis result folder: [default: %s]" % ROOTFILE)

    parser.add_argument("-c", "--channels", dest="channels", default=CHANNELS,
                        help="list of names of the analysis channels separated by commas: [default: %s]" % CHANNELS)


    return parser.parse_args()

def arg2list(argstr):
    return argstr.strip().split(",")

def load_rate(filename, channel, process):
    tfile = ROOT.TFile.Open(filename, 'read')
    h = tfile.Get(f"{channel}/{process}")
    ret = h.Integral()
    tfile.Close()
    return str(ret)

def get_process_numbering(n_sig, n_bkg, channels):
    return np.tile([*range(0, -n_sig, -1), *range(1,n_bkg+1)], channels).astype(str)

def get_type(u_dict, name):
    return u_dict[name]["type"]

def get_col_space(*texts):
    return np.max([np.sum([len(t) for t in textline]) for textline in zip(*texts)]) + len(texts)

def get_uncertainty(u_dict, u_name, c_name, p_name):
    if u_dict[u_name]["type"] == "shape":
        try:
            val = u_dict[u_name][c_name][p_name]
        except:
            val = "1"
        if val.strip() == "-" or val.strip() == "0": return " - "
        else: return val
    try:
        return u_dict[u_name][c_name][p_name]
    except:
        return " - "

def make_sep(prev):
    n = len(prev[-1])
    prev.append("-" * n)
    return prev

def space(texts, totals):
    return ''.join([text + " " * (total - len(text) + 1) for total, text in zip(totals, texts)])

def rspace(left, right, total):
    w_len = total - len(right) - len(left)
    return left + " " * w_len + right

def main(args):

    args = parse_args()

    # convert argument strings to lists
    channels = arg2list(args.channels)
    sig_processes = arg2list(args.sig_names)
    bkg_processes = arg2list(args.bkg_names)

    # get observed rates
    obs_rates = [load_rate(args.rootfile, ch, "data_obs") for ch in channels]

    # load uncertainty dict from json file
    use_uncerts = args.uncertainties_file != ""
    if use_uncerts:
        with open(args.uncertainties_file, 'r') as f:
            u_dict = json.load(f)
        kmax = len(u_dict.keys())

    imax = len(channels)
    jmax = len(sig_processes) + len(bkg_processes) - 1

    # get datacard row data
    bin_names = np.repeat(channels, jmax+1)
    process_names = np.tile([*sig_processes, *bkg_processes], imax)
    process_nums = get_process_numbering(len(sig_processes), len(bkg_processes), imax)
    rates = [load_rate(args.rootfile, ch, proc) for ch, proc in zip(bin_names, process_names)]

    # get uncertainty columns
    u_names = list(u_dict.keys())
    u_types = [u_dict[name]["type"] for name in u_names]

    # calculate how much total space is needed for each datacard column
    col_space = []
    first_col_len = len("process ")
    if use_uncerts:
        first_col_len = max(first_col_len, get_col_space(u_names, u_types))
    col_space.append(first_col_len)

    # get the space needed for each column for readability
    for i in range(len(bin_names)):
        c = bin_names[i]
        p = process_names[i]
        r = rates[i]
        texts = [c, p, r]
        if use_uncerts:
            texts += [get_uncertainty(u_dict, u_name, c, p) for u_name in u_names]
        col_space.append(get_col_space(texts))

    # format bin counts
    lines = [f"imax {imax}",
             f"jmax {jmax}",
             f"kmax {kmax}"
            ]
    lines = make_sep(lines)
    
    # format shapes line
    lines.append(f"shapes * * {args.rootfile} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC")
    lines = make_sep(lines)

    # format obs rates lines
    lines.append(f"bins {' '.join(channels)}")
    lines.append(f"observation {' '.join(obs_rates)}")
    lines = make_sep(lines)

    # format process/rate lines
    for name, vals in zip(["bin", "process", "process", "rate"], [bin_names, process_names, process_nums, rates]):
        lines.append(
            space([name, *vals], col_space)
        )

    # format uncertainty lines
    if use_uncerts:
        lines = make_sep(lines)

        for n, t in zip(u_names, u_types):
            u_vals = [get_uncertainty(u_dict, n, c, p) for c, p in zip(bin_names, process_names)]
            lines.append(
                space([rspace(n, t, col_space[0]), *u_vals], col_space)
            )

    # add newlines
    lines = "\n".join(lines)

    # write datacard
    with open(args.out_name, "w") as f:
        f.write(lines)

if __name__ == "__main__":
    args = parse_args()
    main(args)