#!/usr/bin/env python3

# Last Updated Oct 10. 2022 by Jieun Choi
# We need to use 6.26 for vectorising bin width for THnT
# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh

import ROOT
from ROOT import *
from array import array
import numpy as np
import os,sys

# doProjection4JEC.py
# Fill 4 Variables in 4d histogram if it is not defined (def roll_in4D)
# Make a projection and unroll 1D responses w.r.t. photon pT and jet1 eta and alpha bins (def unroll_3D)
# Extract Median
# Make a projection and unroll 1D responses vs photon pT w.r.t. jet1 eta and alpha bins (def unroll_3D)

def roll_in4D(infile):
    '''
        If there is no 4D histogram, roll information in 4D hist for Projection.
        To 4D: Photon pT, Leading Jet (abs)eta, Response, Alpha
    '''

    outfile = TFile.Open('hist_rolled_in.root','RECREATE')
    
    bin_info = vector("vector<float>")(4,)
    bin_info[0] = [33, 40, 50, 60, 85, 105, 130, 175, 230, 300, 400, 500, 700, 1000]   # photon_pt bins
    bin_info[1] = [0.0, 0.783, 1.305, 1.93, 2.5, 2.964, 3.2, 5.191]                    # jet1_eta bins
    bin_info[2] = [i/10. for i in range(50+1)]                                         # response (50,0,5)
    bin_info[3] = [0.0,0.01,0.05,0.1,0.15,0.2,0.3]                                     # alpha bins
    
    # 4 in 1
    h_resp4D = THnF('h_resp4D','',4,array('i',(13,7,50,6)),bin_info)
    h_resp4D.GetAxis(0).SetTitle("Photon p_T (GeV)")
    h_resp4D.GetAxis(1).SetTitle("Leading Jet #eta")
    h_resp4D.GetAxis(2).SetTitle("Response")
    h_resp4D.GetAxis(3).SetTitle("#alpha")
    h_resp4D.Sumw2()
    
    # Cross-check
    h_photon_pt = TH1D('h_photon_pt','',13,array('d',[33, 40, 50, 60, 85, 105, 130, 175, 230, 300, 400, 500, 700, 1000]))
    h_photon_pt.GetXaxis().SetTitle("Photon p_T (GeV)")
    h_photon_pt.GetYaxis().SetTitle("Entries")
    h_photon_pt.Sumw2()
    h_photon_pt.StatOverflows(True)
    
    h_jet1_eta = TH1D('h_jet1_eta','',7,array('d',[0,0.783,1.305,1.93,2.5,2.964,3.2,5.191]))
    #h_jet1_eta = TH1D('h_jet1_eta','',14,array('d',[-5.191, -3.2, -2.964, -2.5, -1.93, -1.305, -0.783, 0, 0.783, 1.305, 1.93, 2.5, 2.964, 3.2, 5.191]))
    h_jet1_eta.GetXaxis().SetTitle("Leading Jet #eta")
    h_jet1_eta.GetYaxis().SetTitle("Entires")
    h_jet1_eta.Sumw2()
    
    h_resp = TH1D('h_resp','',50,0,5)
    h_resp.GetXaxis().SetTitle("Response")
    h_resp.GetYaxis().SetTitle("Entries")
    h_resp.Sumw2()
    
    h_alpha = TH1D('h_alpha','',6,array('d',[0.0,0.01,0.05,0.1,0.15,0.2,0.3]))
    h_alpha.GetXaxis().SetTitle("#alpha")
    h_alpha.GetYaxis().SetTitle("Entries")
    h_alpha.Sumw2()
    
    print("Initializing histograms is done")
    
    tree = infile.Get('outTree')
    nevt = tree.GetEntries()
    for i in range(nevt):
        tree.GetEntry(i)
    
        photon_pt = tree.photon_pt
        jet1_eta = tree.jet1_eta
        res_bal = tree.R_balance
        alpha= tree.alpha
    
        h_resp4D.Fill(photon_pt,abs(jet1_eta),res_bal,alpha)
    
        h_photon_pt.Fill(photon_pt)
        h_jet1_eta.Fill(jet1_eta)
        h_resp.Fill(res_bal)
        h_alpha.Fill(alpha)
    
    h_resp4D.Write()
    h_photon_pt.Write()
    h_jet1_eta.Write()
    h_resp.Write() 
    h_alpha.Write()

    outfile.Close()

    return h_resp4D


def roll_out4D(hist4D,outfile):
    '''
        Project 4D histogram into 1D histogram 
        From: 4D hist with Photon pT / jet1 eta / response / alpha
        To:   Response w.r.t. photon pT AND jet1 eta AND alpha binning for extracting median
              Response (median) vs photon pT w.r.t. jet1 eta and alpha binning
                                              
    '''
    
    outhist = TFile.Open(outfile,'RECREATE')
    
    # For inserting title in advance
    # photon pT / jet 1 eta / alpha bin label
    str_photon_pt = [
                      '33 <= photon pT < 40', # additional bin
                      '40 <= photon pT < 50', 
                      '50 <= photon pT < 60', 
                      '60 <= photon pT < 85', 
                      '85 <= photon pT < 105', 
                      '105 <= photon pT < 130', 
                      '130 <= photon pT < 175', 
                      '175 <= photon pT < 230', 
                      '230 <= photon pT < 300',
                      '300 <= photon pT < 400',
                      '400 <= photon pT < 500',
                      '500 <= photon pT < 700',
                      '700 <= photon pT < 1000',
                      'photon pT > 1000',
                    ]
    # non-abs eta bins
    #str_jet1_eta = [
    #                 '5.2 <= jet1 eta < 3.2'
    #                 '3.2 <= jet1 eta < 3.0'
    #                 '3.0 <= jet1 eta < 2.5'
    #                 '2.5 <= jet1 eta < 2.0'
    #                 '2.0 <= jet1 eta < 1.3'
    #                 '1.3 <= jet1 eta < 0.8'
    #                 '0.8 <= jet1 eta < 0', 
    #                 '0 <= jet1 eta < 0.8', 
    #                 '0.8 <= jet1 eta < 1.3', 
    #                 '1.3 <= jet1 eta < 2.0', 
    #                 '2.0 <= jet1 eta < 2.5', 
    #                 '2.5 <= jet1 eta < 3.0', 
    #                 '3.0 <= jet1 eta < 3.2', 
    #                 '3.2 <= jet1 eta < 5.2'
    #                ]
    # abs eta bins
    str_jet1_eta = [
                     'jet1 |eta| < 0.8', 
                     '0.8 <= jet1 |eta| < 1.3', 
                     '1.3 <= jet1 |eta| < 2.0', 
                     '2.0 <= jet1 |eta| < 2.5', 
                     '2.5 <= jet1 |eta| < 3.0', 
                     '3.0 <= jet1 |eta| < 3.2', 
                     '3.2 <= jet1 |eta| < 5.2'
                    ]
    # alpha bins
    str_alpha = [
                   'alpha < 0.01',
                   'alpha < 0.05',
                   # ^^^ Additional bin for checking ideal cases ^^^
                   'alpha < 0.1',
                   'alpha < 0.15',
                   'alpha < 0.20',
                   'alpha < 0.25',
                   'alpha < 0.3',
                  ]

    h_resp4D = hist4D
    
    # Retrive 1D histograms for cross-check
    h_resp0 = h_resp4D.Projection(0, "h_photon_pt_proj0")
    h_resp1 = h_resp4D.Projection(1, "h_jet1_eta_proj1")
    h_resp2 = h_resp4D.Projection(2, "h_response_proj2")
    h_resp3 = h_resp4D.Projection(3, "h_alpha_proj3")

    h_resp0.Write()
    h_resp1.Write()
    h_resp2.Write()
    h_resp3.Write()
    
    nBins0 = h_resp0.GetNbinsX()
    nBins1 = h_resp1.GetNbinsX()
    nBins3 = h_resp3.GetNbinsX()
    
    arr_median = []
    
    for binA in range(nBins3):
        # Temporal histogram for projection w.r.t Alpha
        h_resp4D.GetAxis(3).SetRange(0,binA+1)
        h_alphaBin = h_resp4D.Rebin(1)
        h_tmp_resp3D = h_alphaBin.Projection(0,1,2)
        h_tmp_resp3D.SetTitle(f'{str_alpha[binA]}')
        h_tmp_resp3D.GetXaxis().SetTitle("Photon p_T (GeV)")
        h_tmp_resp3D.GetYaxis().SetTitle("Leading jet #eta")
        h_tmp_resp3D.GetZaxis().SetTitle("Response")
        h_tmp_resp3D.Sumw2()
        #h_tmp_resp3D.Write()
    
    #    # Target histogram 2: Response (median) vs alpha w.r.t. photon pT AND jet1 eta
    #    h_tmp_respAlpha = TH1D(f'h_respT_pTBin{}_etaBin{binY}_alphaBin{binA}',f'{str_jet1_eta[binY]} && {str_alpha[binA]}',13,array('d',[33, 40, 50, 60, 85, 105, 130, 175, 230, 300, 400, 500, 700, 1000]))
    
        for binY in range(nBins1): # jet1 eta
            # Target histogram1 : Response (median) vs photon pT w.r.t. jet1 eta AND alpha
            h_tmp = TH1D(f'h_respT_etaBin{binY}_alphaBin{binA}',f'{str_jet1_eta[binY]} && {str_alpha[binA]}',13,array('d',[33, 40, 50, 60, 85, 105, 130, 175, 230, 300, 400, 500, 700, 1000]))
            h_tmp.GetXaxis().SetTitle("Photon p_T (GeV)")
            h_tmp.GetYaxis().SetTitle("Response")
            h_tmp.Sumw2()
        
            for binX in range(nBins0): # photon pT
                # Response w.r.t. photon pT AND jet1 eta AND alpha
                h_tmp_resp = h_tmp_resp3D.ProjectionZ(f'h_response_ptBin{binX}_etaBin{binY}_alphaBin{binA}',binX,binX,binY,binY)
                h_tmp_resp.SetTitle(f'{str_photon_pt[binX]} && {str_jet1_eta[binY]} && {str_alpha[binA]}')
                h_tmp_resp.Write()
                # Median = 0.5 Quantile
                # GetQuantiles() need double* and np.array allow you to use it 
                # https://root-forum.cern.ch/t/get-quantiles-using-pyroot/34147
                p = np.array([0.5]) 
                median = np.array([-1.])
                if h_tmp_resp.Integral() > 0. : h_tmp_resp.GetQuantiles(1,median,p)
                arr_median.append(median[0])
            
                err = -1.
                if h_tmp_resp.GetEffectiveEntries() > 0.: err = 1.253 * h_tmp_resp.GetRMS() / sqrt(h_tmp_resp.GetEffectiveEntries())
                #print(f"MEDIAN: {median}")
                #print(f"ERROR: {err}")
            
                if h_tmp_resp.Integral() > 0. and h_tmp_resp.GetEffectiveEntries() > 0.: 
                    h_tmp.SetBinContent(binX + 1, median[0])
                    h_tmp.SetBinError(binX + 1, err)    
        
            h_tmp.Write()
    
    #print(f'median_array: {arr_median}')
    
    outhist.Close()


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = '''
    %prog [options] option
    ex) doProjection4JEC.py -i inputFILE -o outputFILE
    do Projection from 4D rolled histogram for JEC
    '''
    parser.add_option("-i", "--inputfile", dest="inputFILE",
                      default=False,
                      type="string",
                      help='put the name of your FILENAME')

    parser.add_option("-o", "--outputfile", dest="outputFILE",
                      default=False,
                      type="string",
                      help='put the name of your outputFILE')
    # TODO draw Data vs MC at once
    #parser.add_option("-d", "--draw", dest="draw",
    #                  action='store_true'
    #                  default=False,
    #                  help='output location')

    (options,args) = parser.parse_args()

    if not options.inputFILE:
        print("inputFILE is not defined, exit.")
        sys.exit()

    if not options.outputFILE:
        print("outputFILE is not defined, exit.")
        sys.exit()

    f_in = TFile(str(options.inputFILE))
    vetolist = ['outTree'] # vetolist if you want
    keys = [k.GetName() for k in f_in.GetListOfKeys()]
    hlist = [k for k in keys if not k in vetolist]

    name_4D = 'h_resp4D'
    is4D = True in (hist == name_4D for hist in hlist)
    # This can be in fancy way, but THn vs TH1/2/3 has no common function for GetNDimensions.... TODO
    # is4D = True in (hist.GetDiemnsion() == 4 for hist in hlist)
    # is4D = True in (f_in.Get(hist).GetNDimensions() == 4 for hist in hlist)
    # print(f"Does any element satisfy specified condition ? : {is4D}")

    if is4D:
        print("Histogram is ready")
        target_hist4D = f_in.Get(name_4D)
    else: 
        print("No 4D histogram defined. Roll in 4D histogram from TTree")
        target_hist4D = roll_in4D(f_in)
        print("Histogram is ready")
    
    roll_out4D(target_hist4D,options.outputFILE)

    print("Projection is done!")
