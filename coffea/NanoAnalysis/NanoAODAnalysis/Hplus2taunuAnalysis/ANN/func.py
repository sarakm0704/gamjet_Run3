# Base script from gitlab.cern.ch/Hplus/HiggsAnalysis keras neural networks func.py
#===o=============================================================================================
# Import modules
#================================================================================================   
import ROOT
import plot
import math
import array
import json
import pandas
import numpy 
import contextlib
import glob
import os
import tensorflow as tf
import joblib
from scipy import sparse
from scipy.spatial.distance import jensenshannon
import ctypes
from jsonWriter import JsonWriter 

import matplotlib.pyplot as plt

#================================================================================================
# Variable definition
#================================================================================================
# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors  
ss = "\033[92m"
ns = "\033[0;0m"
ts = "\033[1;34m"
hs = "\033[0;35m"   
ls = "\033[0;33m"
es = "\033[1;31m"
cs = "\033[0;44m\033[1;37m"

#================================================================================================   
# Function definition
#================================================================================================   
def Print(msg, printHeader=False):
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print("=== ", fName)
        print("\t", msg)
    else:
        print("\t", msg)
    return

def Verbose(msg, printHeader=False, verbose=False):
    if not verbose:
        return
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print("=== ", fName)
        print("\t", msg)
    else:
        print("\t", msg)
    return

def split_list(a_list, firstHalf=True):
    half = len(a_list)//2
    if firstHalf:
        return a_list[:half]
    else:
        return a_list[half:]

def convertHistoToGaph(histo, verbose=False):

    # Lists for values
    x     = []
    y     = []
    xerrl = []
    xerrh = []
    yerrl = []
    yerrh = []
    nBinsX = histo.GetNbinsX()
    
    for i in range (0, nBinsX+1):
        # Get values
        xVal  = histo.GetBinCenter(i)
        xLow  = histo.GetBinWidth(i)
        xHigh = xLow
        yVal  = histo.GetBinContent(i)
        yLow  = histo.GetBinError(i)
        yHigh = yLow
 
        # Store values
        x.append(xVal)
        xerrl.append(xLow)
        xerrh.append(xHigh)
        y.append(yVal)
        yerrl.append(yLow)
        yerrh.append(yHigh)
        
    # Create the TGraph with asymmetric errors
    tgraph = ROOT.TGraphAsymmErrors(len(x),
                                    array.array("d",x),
                                    array.array("d",y),
                                    array.array("d",xerrl),
                                    array.array("d",xerrh),
                                    array.array("d",yerrl),
                                    array.array("d",yerrh))
    if verbose:
        tgraph.GetXaxis().SetLimits(-0.05, 1.0)
    tgraph.SetName(histo.GetName())

    # Construct info table (debugging)
    table  = []
    align  = "{0:>6} {1:^10} {2:>10} {3:>10} {4:>10} {5:^3} {6:<10}"
    header = align.format("#", "x-", "x", "x+", "y", "+/-", "Error")
    hLine  = "="*70
    table.append("")
    table.append(hLine)
    table.append("{0:^70}".format(histo.GetName()))
    table.append(header)
    table.append(hLine)
 
    # For-loop: All values x-y and their errors
    for i, xV in enumerate(x, 0):
        row = align.format(i+1, "%.4f" % xerrl[i], "%.4f" %  x[i], "%.4f" %  xerrh[i], "%.5f" %  y[i], "+/-", "%.5f" %  yerrh[i])
        table.append(row)
    table.append(hLine)

    if 0:
        for i, line in enumerate(table, 1):
            print(line)
    return tgraph                              


def PlotCorrelationMatrix(df, saveDir, saveName, saveFormats):
    # Get list of columns (variable names)
    columns  = df.columns.values
    nColumns = df.columns.values.size

    # Calculate the correlation matrix
    corrMatrix = df.corr()

    margin = 0.20
    labelSize =	12
    if (nColumns > 50):
        margin = 0.15
        labelSize = 9
        
    # Create canvas
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetGrid()

    # Set Margin
    canvas.SetRightMargin(0.10)
    canvas.SetLeftMargin(margin)
    canvas.SetBottomMargin(margin)
    
    # Create histogram
    histoName = saveName
    hCorr = ROOT.TH2F(histoName, histoName, nColumns, 0, nColumns, nColumns, 0, nColumns)

    # Fill bins with the values of the correlation matrix.
    for i in range(nColumns):
        c_i = columns[i]
        for j in range(nColumns):
            c_j = columns[j]
            # Get the value of each element in the matrix
            value_ij =	corrMatrix.loc[c_i][c_j]
            #print(i, j, c_i, c_j, value_ij)
            hCorr.Fill(i, j, value_ij)

        # Shorten variable names
        label = plot.GetLabel(c_i)
        # replace bin label with the name of the variable.
        hCorr.GetXaxis().SetBinLabel(i+1,label);
        hCorr.GetYaxis().SetBinLabel(i+1,label);
        
    hCorr.GetXaxis().SetLabelSize(labelSize)
    hCorr.GetYaxis().SetLabelSize(labelSize)
    hCorr.GetZaxis().SetLabelSize(labelSize)
    # Vertical label
    hCorr.GetXaxis().LabelsOption("v")
    hCorr.GetZaxis().SetRangeUser(-1, 1)    
    hCorr.Draw("colz")
    canvas.Modified()
    canvas.Update()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    return

def PlotOutput(Y_train, Y_test, saveDir, saveName, isSB, saveFormats):
    
    ROOT.gStyle.SetOptStat(0)

    # Create canvas
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()

    # Create histograms
    h1 = ROOT.TH1F('train', '', 50, 0.0, 1.0)
    h2 = ROOT.TH1F('test' , '', 50, 0.0, 1.0)

    # Fill histograms
    for r in Y_train:
        h1.Fill(r)

    for r in Y_test:
        h2.Fill(r)

    if 0:
        h1.Scale(1./h1.Integral())
        h2.Scale(1./h2.Integral())

    ymax = max(h1.GetMaximum(), h2.GetMaximum())

    plot.ApplyStyle(h1, ROOT.kMagenta+1)
    plot.ApplyStyle(h2, ROOT.kGreen+2)

    for h in [h1,h2]:
        h.SetMinimum(100)
        h.SetMaximum(ymax*1.1)
        h.GetXaxis().SetTitle("Output")
        h.GetYaxis().SetTitle("Entries")
        h.Draw("HIST SAME")
    
    # What is this for? Ask soti
    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*1.1])
    graph.Draw("same")
    
    # Create legend
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    if isSB:
        leg.AddEntry(h1, "signal","l")
        leg.AddEntry(h2, "background","l")
    else:
        leg.AddEntry(h1, "train","l")
        leg.AddEntry(h2, "test","l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def PlotInputs(signal, bkg, var, saveDir, saveFormats, pType="sigVbkg", standardise=False, w1=numpy.array([]), w2=numpy.array([])):

    # Define plot type
    if pType == "sigVbkg":
        h1 = "signal"
        h2 = "background"
    elif pType == "testVtrain":
        h1 = "train data"
        h2 = "test data"        
    elif pType == "trainVtest":
        h1 = "test data"
        h2 = "train data"
    else:
        Print("Unknown plot type \"%s\". Using dummie names/legends for histograms" % (pType), True)
        h1 = "h1"
        h2 = "h2"
        
    ROOT.gStyle.SetOptStat(0)

    # Create canvas
    canvas = plot.CreateCanvas()
    canvas.cd()
    
    # Convert weights numpy arrays to list
    w1 = w1.reshape((-1,)).tolist()
    w2 = w2.reshape((-1,)).tolist()

    # Definitions 
    info  = plot.GetHistoInfo(var)
    nBins = info["nbins"]
    xMin  = info["xmin"]
    xMax  = info["xmax"]
    if standardise:
        xMin  = -5.0
        xMax  = +5.0
        nBins = 500

    # Create histograms
    hsignal = ROOT.TH1F(h1, '', nBins, xMin, xMax)
    hbkg    = ROOT.TH1F(h2, '', nBins, xMin, xMax)


    # Fill histogram (signal)
    for i, r in enumerate(signal, 0):
        if len(w1) == 0: 
            hsignal.Fill(r)
        else:
            hsignal.Fill(r, w1[i])
            
    # Fill histogram (bkg)    
    for j, r in enumerate(bkg, 0):
        if len(w2) == 0:
            hbkg.Fill(r)
        else:
            hbkg.Fill(r, w2[j])

    if hsignal.Integral() > 0:
        hsignal.Scale(1./hsignal.Integral())

    if hbkg.Integral() > 0:
        hbkg.Scale(1./hbkg.Integral())

    ymax = max(hsignal.GetMaximum(), hbkg.GetMaximum())

    plot.ApplyStyle(hsignal, ROOT.kAzure-3)
    hsignal.SetFillColor(ROOT.kAzure-3)

    plot.ApplyStyle(hbkg, ROOT.kRed +2 )

    for h in [hsignal, hbkg]:
        h.SetMaximum(ymax*1.2)
        h.GetXaxis().SetTitle(info["xlabel"])
        h.GetYaxis().SetTitle(info["ylabel"])

    hsignal.Draw("HIST")
    hbkg.Draw("HIST SAME")

    # Create legend
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    leg.AddEntry(hsignal, h1, "f") 
    leg.AddEntry(hbkg   , h2, "l") 
    leg.Draw()

    plot.SavePlot(canvas, saveDir, var, saveFormats, False)
    canvas.Close()
    return

def PlotPoivsMassDiffPerDataset(mass_true, poi, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    poi = tf.squeeze(poi)
    mass_true = tf.squeeze(mass_true)
    diffs = (poi - mass_true) / mass_true
    diffs = diffs.numpy()
    mass_true = mass_true.numpy()


    hList = []
    gList = []
    nonempty = []
    xTitle = "Relative difference"
    yTitle = "Entries"
    log = True
    nBins = 50
    xMin = -10
    xMax = 10
    yMax = -1
    yMin = 0.5
    trueMassBins = [0,176.25,182.5,210,230,275,450,550,650,750,900,1250,1750,2250,2750,3250]

    if "xMin" in kwargs:
        xMin = kwargs["xMin"]
    if "xMax" in kwargs:
        xMax = kwargs["xMax"]
    if "nBins" in kwargs:
        nBins = kwargs["nBins"]

    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    event_bins = numpy.digitize(mass_true, trueMassBins) - 1
    for i in range(len(trueMassBins)-1):
        ith_notempty = True
        h = ROOT.TH1F(f"[{trueMassBins[i]},{trueMassBins[i+1]})", "Distributions of errors in mass approximation", nBins, xMin, xMax)
        bin_diffs = diffs[event_bins == i]
        for d in bin_diffs:
            if not ith_notempty: ith_notempty = True
            h.Fill(d)
        yMax = max(yMax, h.GetMaximum())

        nonempty.append(ith_notempty)

        color = i+2 # Skip white
        if color > 7:
            color = color - 7
        plot.ApplyStyle(h, color)
        hList.append(h)


    canvas.SetLogy()

    for i, h in enumerate(hList, 0):
        # h.SetMinimum(yMin_* 0.85)        
        # h.SetMaximum(yMax_* 1.15)
        h.SetMaximum(yMin)  # no guarantees when converted to TGraph!
        h.SetMaximum(yMax)  # no guarantees when converted to TGraph!
        h.GetXaxis().SetTitle(xTitle)
        h.GetYaxis().SetTitle(yTitle)
            
        if i==0:
            h.Draw("HIST ][")
        else:
            h.Draw("HIST SAME ][")

    # Create legend
    leg = plot.CreateLegend(0.70, 0.3, 0.90, 0.90)

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        if nonempty[i]:
            leg.AddEntry(h, h.GetName(), "l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Create TGraph
    for h in hList:
        gList.append(convertHistoToGaph(h))

    # Write the Tgraph into the JSON file
    for gr in gList:
        gName = "%s_%s" % (saveName, gr.GetName())
        jsonWr.addGraph(gName, gr)
    return

def PlotDNNscorePerDataset(mass_true, score, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    hList = []
    gList = []
    nonempty = []
    xTitle = "DNN classification score"
    yTitle = "Entries"
    log = True
    nBins = 100
    xMin = 0
    xMax = 1
    yMax = -1
    yMin = 0.5
    trueMassBins = [0,176.25,182.5,210,230,275,450,550,650,750,900,1250,1750,2250,2750,3250]

    if "xMin" in kwargs:
        xMin = kwargs["xMin"]
    if "xMax" in kwargs:
        xMax = kwargs["xMax"]
    if "nBins" in kwargs:
        nBins = kwargs["nBins"]

    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    event_bins = numpy.digitize(mass_true, trueMassBins) - 1
    for i in range(len(trueMassBins)-1):
        ith_notempty = False
        h = ROOT.TH1F(f"[{trueMassBins[i]},{trueMassBins[i+1]})", "Distributions of DNN score", nBins, xMin, xMax)
        bin_scores = score[event_bins == i]
        for d in bin_scores:
            if not ith_notempty: ith_notempty = True
            h.Fill(d)

        nonempty.append(ith_notempty)
        yMax = max(yMax, h.GetMaximum())

        color = i+2 # Skip white
        if color > 7:
            color = color - 7
        plot.ApplyStyle(h, color)
        hList.append(h)


    canvas.SetLogy()

    for i, h in enumerate(hList, 0):
        # h.SetMinimum(yMin_* 0.85)        
        # h.SetMaximum(yMax_* 1.15)
        h.SetMaximum(yMin)  # no guarantees when converted to TGraph!
        h.SetMaximum(yMax)  # no guarantees when converted to TGraph!
        h.GetXaxis().SetTitle(xTitle)
        h.GetYaxis().SetTitle(yTitle)
            
        if i==0:
            h.Draw("HIST ][")
        else:
            h.Draw("HIST SAME ][")

    # Create legend
    leg = plot.CreateLegend(0.70, 0.3, 0.90, 0.90)

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        if nonempty[i]:
            leg.AddEntry(h, h.GetName(), "l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Create TGraph
    for h in hList:
        gList.append(convertHistoToGaph(h))

    # Write the Tgraph into the JSON file
    for gr in gList:
        gName = "%s_%s" % (saveName, gr.GetName())
        jsonWr.addGraph(gName, gr)
    return


def PlotMassPredictionErrors(mass_true, mass_pred, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    mass_pred = tf.squeeze(mass_pred)
    mass_true = tf.squeeze(mass_true)
    diffs = mass_pred - mass_true

    xBins = (300,0,3100)
    yBins = xBins
#    yBins = (200,-1000,1000)
    xTitle = "true mass"
    yTitle = "mass prediction"
    zTitle = "Entries"
    log = True

    if 'xBins' in kwargs:
        xBins = kwargs['xBins']

    if 'yBins' in kwargs:
        yBins = kwargs['yBins']

    if 'log' in kwargs:
        log = kwargs['log']

    # Create canvas
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    h = ROOT.TH2F("PredErrors", "Distribution of mass predictions", *xBins, *yBins)
    for m, d in zip(mass_true, mass_pred):
        h.Fill(m, d)
#    plot.ApplyStyle(h, 0)
    if log:
        canvas.SetLogz()

    h.GetXaxis().SetTitle(xTitle)
    h.GetYaxis().SetTitle(yTitle)
    h.GetZaxis().SetTitle(zTitle)
    h.Draw("colz")

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

#    g = convertHistoToGaph(h)

    # Write the Tgraph into the JSON file
 #   gName = "%s_%s" % (saveName, g.GetName())
 #   jsonWr.addGraph(gName, g)
    return

def PlotAndWriteJSON(signal, bkg, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    resultsDict = {}
    resultsDict["signal"] = signal
    resultsDict["background"] = bkg

    normalizeToOne = False
    if "normalizeToOne" in kwargs:
        normalizeToOne = kwargs["normalizeToOne"]

   # Create canvas
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    hList  = []
    gList  = []
    yMin   = 100000
    yMax   = 100000
    xMin   = 0.0
    xMax   = 1.0
    nBins  = 50
    xTitle = "DNN output"
    yTitle = "Entries"
    log    = True
    logx   = False
    xBins  = None

    if "log" in kwargs:
        log = kwargs["log"]

    if "logx" in kwargs:
        logx = kwargs["logx"]
    
    if "xTitle" in kwargs:
        xTitle = kwargs["xTitle"]

    if "yTitle" in kwargs:
        yTitle = kwargs["yTitle"]
    if "xMin" in kwargs:
        xMin = kwargs['xMin']
    if "xMax" in kwargs:
        xMax = kwargs['xMax']
    if "yMin" in kwargs:
        yMin = kwargs['yMin']
    if "yMax" in kwargs:
        yMax = kwargs['yMax']
    if "nBins" in kwargs:
        nBins = kwargs['nBins']
    if "xBins" in kwargs:
        xBins = kwargs['xBins']
        nBins = len(xBins) - 1
    
    # For-loop: All results
    for i, key in enumerate(resultsDict.keys(), 0):
        # Print("Constructing histogram %s" % (key), i==0)

        if xBins is not None:
            h = ROOT.TH1F(key, '', nBins, xBins)
        else:
            h = ROOT.TH1F(key, '', nBins, xMin, xMax)
        
        # For-loop: All Entries
        for j, x in enumerate(resultsDict[key], 0):
            h.Fill(x)
            try:
                yMin_ = min(x[0], yMin)
            except:
                pass
                
        # Save maximum
        yMax_ = max(h.GetMaximum(), yMax)

        # Customise & append to list
        plot.ApplyStyle(h, i+1)

        if normalizeToOne:
            if h.Integral()>0.0:
                h.Scale(1./h.Integral())
        hList.append(h)

    if log:
        # print "yMin = %s, yMax = %s" % (yMin, yMax)
        canvas.SetLogy()

    if logx:
        canvas.SetLogx()

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        # h.SetMinimum(yMin_* 0.85)        
        # h.SetMaximum(yMax_* 1.15)
        h.SetMaximum(yMin)  # no guarantees when converted to TGraph!
        h.SetMaximum(yMax)  # no guarantees when converted to TGraph!
        h.GetXaxis().SetTitle(xTitle)
        h.GetYaxis().SetTitle(yTitle)
            
        if i==0:
            h.Draw("HIST")
        else:
            h.Draw("HIST SAME")
    
    # Create legend
    leg = plot.CreateLegend(0.70, 0.76, 0.90, 0.90)
    if "legHeader" in kwargs:
        leg.SetHeader(kwargs["legHeader"])

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        if "legEntries" in kwargs:
            leg.AddEntry(h, kwargs["legEntries"][i], "l")
        else:
            leg.AddEntry(h, h.GetName(), "l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Create TGraph
    for h in hList:
        gList.append(convertHistoToGaph(h))

    # Write the Tgraph into the JSON file
    for gr in gList:
        gName = "%s_%s" % (saveName, gr.GetName())
        jsonWr.addGraph(gName, gr)
    return

def PlotAndWriteJSON_DNNscore(sigOutput, bkgOutput, cutValue, signal, bkg, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    resultsDict = {}
    resultsDict["signal"] = signal
    resultsDict["background"] = bkg

    normalizeToOne = False
    if "normalizeToOne" in kwargs:
        normalizeToOne = kwargs["normalizeToOne"]

   # Create canvas
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    hList  = []
    gList  = []
    yMin   = 0.01
    yMax   = None
    xMin   = 0.0
    xMax   = 1.0
    nBins  = 50
    xTitle = "DNN output"
    yTitle = "Entries"
    log    = True
    if "log" in kwargs:
        log = kwargs["log"]
    if "xTitle" in kwargs:
        xTitle = kwargs["xTitle"]
    if "yTitle" in kwargs:
        yTitle = kwargs["yTitle"]
    if "xMin" in kwargs:
        xMin = kwargs['xMin']
    if "xMax" in kwargs:
        xMax = kwargs['xMax']
    if "nBins" in kwargs:
        nBins = kwargs['nBins']
    if "yMax" in kwargs:
        yMax = kwargs["yMax"]
    if "yMin" in kwargs:
        yMin = kwargs["yMin"]
    
    # For-loop: 
    yMax_ = -100000
    for i, key in enumerate(resultsDict.keys(), 0):

        h = ROOT.TH1F("%s_%s_%s" % (key, saveName, "WP" + str(cutValue)), '', nBins, xMin, xMax)
        for j, x in enumerate(resultsDict[key], 0):

            #print "key = %s, x = %.2f, nBins = %d, xMin = %.2f, xMax = %.2f" % (key, x, nBins, xMin, xMax)
            score  = None
            if key == "signal":
                score = sigOutput[j]
            elif key == "background":
                score = bkgOutput[j]
            else:
                raise Exception("This should not be reached")
    
            # Check if DNN score satisfies cut
            if score < cutValue:
                Verbose("%d) DNN score for %s is %.3f" % (j, key, score), False)
                continue
            else:
                Verbose("%d) DNN score for %s is %.3f" % (j, key, score), False)

            # Fill the histogram
            if len(x) > 0:
                h.Fill(x)
            try:
                yMin_ = min(x[0], yMin)
            except:
                pass
                
        # Save maximum
        yMax_ = max(h.GetMaximum(), yMax_)

        # Customise & append to list
        plot.ApplyStyle(h, i+1)

        if normalizeToOne:
            if h.Integral()>0.0:
                h.Scale(1./h.Integral())
        hList.append(h)

    if yMax == None:
        yMax = yMax_

    if log:
        canvas.SetLogy()

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        #h.SetMinimum(yMin*0.85)        
        #h.SetMaximum(yMax*1.15)
        h.SetMinimum(yMin)
        h.SetMaximum(yMax)

        h.GetXaxis().SetTitle(xTitle)
        h.GetYaxis().SetTitle(yTitle)
            
        if i==0:
            h.Draw("HIST")
        else:
            h.Draw("HIST SAME")
    
    # Create legend
    leg = plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    for h in hList:
        leg.AddEntry(h, h.GetName().split("_")[0],"l")
    leg.Draw()
    
    postfix = "_%s%s" % ("WP", str(cutValue).replace(".", "p"))
    plot.SavePlot(canvas, saveDir, saveName + postfix, saveFormats)
    canvas.Close()

    # Create TGraph
    for h in hList:
        gList.append(convertHistoToGaph(h))

    # Write the Tgraph into the JSON file
    sample = "NA"
    for gr in gList:
        # gr.GetName() is too long since it must be unique (memory replacement issues). Make it simples
        if "signal" in gr.GetName().lower():
            sample = "signal"
        elif "background" in gr.GetName().lower():
            sample = "background"
        else:
            sample = "unknown"            

        #gName = "%s_%s" % (saveName, gr.GetName()) + postfix @ 
        gName = "%s_%s" % (saveName, sample) + postfix
        jsonWr.addGraph(gName, gr)
    return hList

def PlotInputDistortion(preds, inputs, w, name, saveDir, saveFormats, wp = 0.5):

    info  = plot.GetHistoInfo(name)


    nBins = 100
    xMin = inputs.min()
    xMax = inputs.max()
    # Create canvas
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()

    h_all = ROOT.TH1F(f"{name} distribution all", f"Distribution of {name} values", nBins, xMin, xMax)
    h_passed = ROOT.TH1F(f"{name} distribution passed", f"Distribution of {name} values", nBins, xMin, xMax)

    for i, input in enumerate(inputs):
        h_all.Fill(input, w[i])
        if preds[i] > wp:
            h_passed.Fill(input, w[i])

    if h_all.Integral() > 0:
        h_all.Scale(1./h_all.Integral())

    if h_passed.Integral() > 0:
        h_passed.Scale(1./h_passed.Integral())

    ymax = max(h_passed.GetMaximum(), h_passed.GetMaximum())

    plot.ApplyStyle(h_all, 1)
    plot.ApplyStyle(h_passed, ROOT.kAzure)

    for h in [h_all, h_passed]:
        h.SetMaximum(ymax*1.2)
        h.GetXaxis().SetTitle(info["xlabel"])
        h.GetYaxis().SetTitle(info["ylabel"])

    h_passed.Draw("HIST")
    h_all.Draw("HIST SAME")

    # Create legend
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    leg.AddEntry(h_passed, f"after {wp} cut", "f") 
    leg.AddEntry(h_all   , f"all events", "l") 
    leg.Draw()

    plot.SavePlot(canvas, saveDir, name, saveFormats, False)
    canvas.Close()
    return

def PlotTGraph(xVals, xErrs, yVals, yErrs, saveDir, saveName, jsonWr, saveFormats, **kwargs):

    # Create a TGraph object
    graph = plot.GetGraph(xVals, yVals, xErrs, xErrs, yErrs, yErrs)
 
    # Create a TCanvas object
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    #canvas.SetLogy()
    
    # Create the TGraph with asymmetric errors                                                                                                                                                              
    tgraph = ROOT.TGraphAsymmErrors(len(xVals),
                                    array.array("d", xVals),
                                    array.array("d", yVals),
                                    array.array("d", xErrs),
                                    array.array("d", xErrs),
                                    array.array("d", yErrs),
                                    array.array("d", yErrs))
    tgraph.SetName(saveName)
    legName = None
    if "efficiency" in saveName.lower():
        legName = "efficiency"
        if saveName.lower().endswith("sig"):
            plot.ApplyStyle(tgraph, ROOT.kBlue)
        else:
            plot.ApplyStyle(tgraph, ROOT.kRed)
    elif "significance" in saveName.lower():
        legName = "significance"
        if saveName.lower().endswith("sig"):
            plot.ApplyStyle(tgraph, ROOT.kGreen)
        else:
            plot.ApplyStyle(tgraph, ROOT.kGreen+3)
    else:
        plot.ApplyStyle(tgraph, ROOT.kOrange)
        
    # Draw the TGraph
    if "xTitle" in kwargs:
        tgraph.GetXaxis().SetTitle(kwargs["xTitle"])
    if "yTitle" in kwargs:
        tgraph.GetYaxis().SetTitle(kwargs["yTitle"])
    if "xMin" in kwargs and "xMax" in kwargs:
        tgraph.GetXaxis().SetLimits(kwargs["xMin"], kwargs["xMax"])
    else:
        tgraph.GetXaxis().SetLimits(-0.05, 1.0)
    tgraph.Draw("AC")
        
    # Create legend
    leg = plot.CreateLegend(0.60, 0.70, 0.85, 0.80)
    if legName is not None:
        leg.AddEntry(tgraph, legName, "l")
        leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Write the Tgraph into the JSON file
    jsonWr.addGraph(saveName, tgraph)
    return

def GetJSD(var, y_pred, wp, **kwargs):
    # Computes the square root of the Jensen-Shannon divergence.
    #https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html
    #https://machinelearningmastery.com/divergence-between-probability-distributions/
    xMin    = kwargs["xMin"]
    xMax    = kwargs["xMax"]
    nBins   = kwargs["nBins"]
    myBins  = numpy.linspace(xMin, xMax, nBins)
    
    # Concatenate Y (predicted output) and target variable
    # Ymass 0 column:   output
    # Ymass 1st column: target
    Yvar = numpy.concatenate((y_pred, var), axis=1)
    var_all, _ = numpy.histogram(var, myBins)
    #for wp in range(0, 1, 0.025):
    Yvar_sel = Yvar[Yvar[:,0] >= wp]
    #varSel = Yvar_sel = Yvar_sel[:, 1]
    varSel = Yvar_sel[:, 1]
    var_sel, _ = numpy.histogram(varSel, myBins)
        
    if(numpy.sum(var_sel)==0.0):
        jsd = -1
    else:
        jsd = jensenshannon(var_sel, var_all)
    return jsd

def GetEfficiency(histo):

    # Initialize sigma variables
    nbins    = histo.GetNbinsX()
    intErrs  = ctypes.c_double(0.0) #ROOT.Double(0.0)
    intVals  = histo.IntegralAndError(0, nbins+1, intErrs, "")
    xVals    = []
    xErrs    = []
    yVals    = []
    yErrs    = []
    yTmp     = ctypes.c_double(0.0) #ROOT.Double(0.0)
    if intVals == 0.0:
        Print("WARNING! The integral of histogram \"%s\" is zero!" % (histo.GetName()), True)
    
    # For-loop: All bins
    for i in range(0, nbins+1):
        xVal = histo.GetBinCenter(i)
        #if xVal < 0.0:
        #    continue
        xErr = histo.GetBinWidth(i)*0.5
        intBin = histo.IntegralAndError(i, nbins+1, yTmp, "")
        if intVals > 0:
            yVals.append(intBin/intVals)
        else:
            yVals.append(0.0)
        xVals.append(xVal)
        xErrs.append(xErr)
        if intVals > 0:
            yErrs.append(yTmp.value/intVals)
        else:
            yVals.append(0.0)

    return xVals, xErrs, yVals, yErrs


def CalcEfficiency(htest_s, htest_b):

    # Initialize sigma variables
    nbins    = htest_s.GetNbinsX()
    sigmaAll = ctypes.c_double(0.0) #ROOT.Double(0.0)
    sigmaSel = ctypes.c_double(0.0) #ROOT.Double(0.0)
    All_s    = htest_s.IntegralAndError(0, nbins+1, sigmaAll, "")
    All_b    = htest_b.IntegralAndError(0, nbins+1, sigmaAll, "")
    eff_s    = []
    eff_b    = [] 
    xvalue   = []
    error    = []
    
    # For-loop: All bins
    for i in range(0, nbins+1):
        Sel_s = htest_s.IntegralAndError(i, nbins+1, sigmaSel, "")
        Sel_b = htest_b.IntegralAndError(i, nbins+1, sigmaSel, "")

        if (All_s <= 0):
            All_s = 1
            Sel_s = 0
        if (All_b <= 0):
            All_b = 1
            Sel_b = 0

        eff_s.append(Sel_s/All_s)
        eff_b.append(Sel_b/All_b)
        error.append(0)
        xvalue.append(htest_s.GetBinCenter(i))
    
    #print "%d: %s" % (len(xvalue), xvalue)
    #print "%d: %s" % (len(eff_s), eff_s)
    return xvalue, eff_s, eff_b, error

def CalcSignificance(htest_s, htest_b):
    nbins = htest_s.GetNbinsX()
    h_signif0=ROOT.TH1F('signif0', '', nbins, 0.0, 1.)
    h_signif1=ROOT.TH1F('signif1', '', nbins, 0.0, 1.)
    
    sigmaSel_s = ctypes.c_double(0.0) #ROOT.Double(0.0)
    sigmaSel_b = ctypes.c_double(0.0) #ROOT.Double(0.0)
    
    for i in range(0, nbins+1):
        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbins+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbins+1, sigmaSel_b, "")
        # Calculate Significance
        _sign0 = 0
        if (sSel+bSel > 0):
            _sign0 = sSel/math.sqrt(sSel+bSel)

        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))
        h_signif0.Fill(htest_s.GetBinCenter(i), _sign0)        
        h_signif1.Fill(htest_s.GetBinCenter(i), _sign1)        
    return h_signif0, h_signif1

def GetSignificance(htest_s, htest_b):
    nbinsX     = htest_s.GetNbinsX()
    sigmaSel_s = ctypes.c_double(0.0) #ROOT.Double(0.0)
    sigmaSel_b = ctypes.c_double(0.0) #ROOT.Double(0.0)
    xVals      = []
    xErrs      = []
    signif_def = [] # S/sqrt(S+B) - same definition as TMVA
    signif_alt = [] # 2[sqrt(S+B) -sqrt(B)]

    # For-loop: All histogram bins
    for i in range(0, nbinsX+1):
        xVal = htest_s.GetBinCenter(i)
        if xVal < 0.0:
            continue
        xErr = htest_s.GetBinWidth(i)*0.5

        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbinsX+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbinsX+1, sigmaSel_b, "")

        # Calculate Significance
        _sign0 = 0
        if (sSel+bSel > 0):
            _sign0 = sSel/math.sqrt(sSel+bSel)
        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))

        # Append values
        xVals.append(xVal)
        xErrs.append(xErr)
        signif_def.append(_sign0)
        signif_alt.append(_sign1)
    return xVals, xErrs, signif_def, signif_alt

def PlotEfficiency(htest_s, htest_b, saveDir, saveName, saveFormats):
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLeftMargin(0.145)
    canvas.SetRightMargin(0.11)

    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(htest_s, htest_b)
    graph_s = plot.GetGraph(xvalue, eff_s, error, error, error, error)
    graph_b = plot.GetGraph(xvalue, eff_b, error, error, error, error)
    
    plot.ApplyStyle(graph_s, ROOT.kBlue)
    plot.ApplyStyle(graph_b, ROOT.kRed)
    
    # Calculate significance vs output
    h_signif0, h_signif1 = CalcSignificance(htest_s, htest_b)
    
    plot.ApplyStyle(h_signif0, ROOT.kGreen)
    plot.ApplyStyle(h_signif1, ROOT.kGreen+3)
    
    #=== Get maximum of significance
    maxSignif0 = h_signif0.GetMaximum()
    maxSignif1 = h_signif1.GetMaximum()
    maxSignif = max(maxSignif0, maxSignif1)
    
    # Normalize significance
    h_signifScaled0 = h_signif0.Clone("signif0")
    if maxSignif > 0:
        h_signifScaled0.Scale(1./float(maxSignif))

    h_signifScaled1 = h_signif1.Clone("signif1")
    if maxSignif > 0:
        h_signifScaled1.Scale(1./float(maxSignif))

    #Significance: Get new maximum
    ymax = max(h_signifScaled0.GetMaximum(), h_signifScaled1.GetMaximum())

    for obj in [graph_s, graph_b, h_signifScaled0, h_signifScaled1]:
        obj.GetXaxis().SetTitle("Output")
        obj.GetYaxis().SetTitle("Efficiency")
        obj.SetMaximum(ymax*1.1)
        obj.SetMinimum(0)
    #Draw    
    h_signifScaled0.Draw("HIST")
    h_signifScaled1.Draw("HIST SAME")
    graph_s.Draw("PL SAME")
    graph_b.Draw("PL SAME")

    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*1.1])
    graph.Draw("same")

    #Legend
    leg=plot.CreateLegend(0.50, 0.25, 0.85, 0.45)    
    leg.AddEntry(graph_s, "Signal Efficiency", "l")
    leg.AddEntry(graph_b, "Bkg Efficiency", "l")
    leg.AddEntry(h_signifScaled0, "S/#sqrt{S+B}", "l")
    leg.AddEntry(h_signifScaled1, "2#times(#sqrt{S+B} - #sqrt{B})", "l")
    leg.Draw()

    # Define Right Axis (Significance)
    signifColor = ROOT.kGreen+2
    rightAxis = ROOT.TGaxis(1, 0, 1, 1.1, 0, 1.1*maxSignif, 510, "+L")
    rightAxis.SetLineColor ( signifColor )
    rightAxis.SetLabelColor( signifColor )
    rightAxis.SetTitleColor( signifColor )
    rightAxis.SetTitleOffset(1.25)
    rightAxis.SetLabelOffset(0.005)
    rightAxis.SetLabelSize(0.04)
    rightAxis.SetTitleSize(0.045)
    rightAxis.SetTitle( "Significance" )
    rightAxis.Draw()
        
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return
    
def PrintArray(array):
    '''
    Fixes missing leading whitespace (fixme!)
    https://stackoverflow.com/questions/23870301/extra-spaces-in-the-representation-of-numpy-arrays-of-floats
    '''
    _str = str(array)
    
    if ("[ " not in _str):
        _str = _str.replace("[","[ ")
    if (" ]" not in _str):
        _str = _str.replace("]"," ]")
    return _str

def GetROC(htest_s, htest_b):
    '''
    Get ROC curve (signal efficiency vs bkg efficiency)
    '''
    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(htest_s, htest_b)
    graph_roc = plot.GetGraph(eff_s, eff_b, error, error, error, error)
    return graph_roc

def PlotROC(graphMap, saveDir, saveName, saveFormats):
    '''
    Plot ROC curves (signal efficiency vs bkg efficiency)
    '''
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()
    Nhisto = len(graphMap["graph"])
    #leg = plot.CreateLegend(0.25, 0.25, 0.60, 0.65) #0.50, 0.25, 0.85, 0.45

    if (Nhisto < 4):
        leg = plot.CreateLegend(0.25, 0.45, 0.60, 0.65) #0.50, 0.25, 0.85, 0.45
    elif (Nhisto > 15):
        leg = plot.CreateLegend(0.25, 0.25, 0.60, 0.85) #0.50, 0.25, 0.85, 0.45 
    else:
        leg = plot.CreateLegend(0.25, 0.25, 0.60, 0.65) #0.50, 0.25, 0.85, 0.45

    leg = plot.CreateLegend(0.20, 0.65, 0.55, 0.85)
    lineStyle = ROOT.kSolid

    # For-loop: All graphs
    for i, k in enumerate(graphMap["graph"], 0):
        gr = graphMap["graph"][i]
        gr_name = graphMap["name"][i]
        
        color = i+2 # Skip white
        if color > 7:
            color = color - 7
            lineStyle = ROOT.kDashed

        plot.ApplyStyle(gr, color, lineStyle)
        gr.SetMarkerSize(0)
        gr.GetXaxis().SetTitle("Signal Efficiency")
        gr.GetYaxis().SetTitle("Background Efficiency") # "Misidentification rate"
        #gr.SetMinimum(0.0)
        #gr.SetMaximum(1.0)
        gr.GetXaxis().SetRangeUser(0, 1)
        gr.GetYaxis().SetRangeUser(0.001, 1)
        if i == 0:
            gr.Draw("apl")
        else:
            gr.Draw("pl same")
        leg.AddEntry(gr, gr_name, "l")

    leg.Draw("same")

    texcms  = plot.AddCMSText()
    texpre  = plot.AddPreliminaryText()
    texlumi = plot.AddLumiText()
    texcms.Draw("same")
    texpre.Draw("same")
    texlumi.Draw("same")

    
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def PlotOvertrainingTest(Y_train_S, Y_test_S, Y_train_B, Y_test_B, saveDir, saveName, saveFormats, w_train_S, w_test_S, w_train_B, w_test_B):
    def ApplyStyle(histo):
        if "_s" in histo.GetName():
            color = ROOT.kBlue
        else:
            color = ROOT.kRed
            
        plot.ApplyStyle(h, color)
        histo.SetMarkerSize(0.5)
        histo.GetXaxis().SetTitle("Output")
        histo.GetYaxis().SetTitle("Entries")
        histo.SetMinimum(10)
        return
        
    def GetLegendStyle(histoName):
        if "test" in histoName:
            legStyle = "f"
            if "_s" in histoName:
                legText = "signal (test)"
            else:
                legText = "background (test)"
        elif "train" in histoName:
            legStyle = "p"
            if "_s" in histoName:
                legText = "signal (train)"
            else:
                legText = "background (train)"
        return legText, legStyle

    def DrawStyle(histoName):
        if "train" in histoName:
            _style = "P"
        else:
            _style = "HIST"
        return _style

    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()

    hList     = []
    DataList = [Y_train_S, Y_test_S, Y_train_B, Y_test_B]
    WeightList = [w_train_S, w_test_S, w_train_B, w_test_B]
    ymax     = 0
    nbins    = 500
    
    # Create the histograms
    htrain_s = ROOT.TH1F('train_s', '', nbins, 0.0, 1.0)
    htest_s  = ROOT.TH1F('test_s' , '', nbins, 0.0, 1.0)
    htrain_b = ROOT.TH1F('train_b', '', nbins, 0.0, 1.0)
    htest_b  = ROOT.TH1F('test_b' , '', nbins, 0.0, 1.0)

    # Append to list
    hList.append(htrain_s)
    hList.append(htest_s)
    hList.append(htrain_b)
    hList.append(htest_b)
    
    for i in range(len(DataList)):
        for r, w in zip(DataList[i], WeightList[i]):
            hList[i].Fill(r, w)

    # Clone the histograms
    htrain_s1 = htrain_s.Clone("train_s")
    htrain_b1 = htrain_b.Clone("train_b")
    htest_s1  = htest_s.Clone("test_s")
    htest_b1  = htest_b.Clone("test_b")
    drawStyle = "HIST SAME"
    leg=plot.CreateLegend(0.55, 0.68, 0.85, 0.88)

    for h in hList:
        h.Rebin(10)
        # Legend
        legText, legStyle = GetLegendStyle(h.GetName())
        leg.AddEntry(h, legText, legStyle)
        ApplyStyle(h)
        h.Scale(1./h.Integral())

    ymax = max(htrain_s.GetMaximum(), htest_s.GetMaximum(), htrain_b.GetMaximum(), htest_b.GetMaximum())
    for h in hList:
        
        h.SetMaximum(ymax*2)        
        h.Draw(DrawStyle(h.GetName())+" SAME")
    
    leg.Draw()
    
    # Save & close canvas
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return htrain_s1, htest_s1, htrain_b1, htest_b1


def WriteModel(model, model_json, inputList, scaler_attributes, output, verbose=False):
    '''
    Write model weights and architecture in txt file
    '''
    arch = json.loads(model_json)
    with open(output, 'w') as fout:
        # Store input variable names
        fout.write('inputs ' + str(len(inputList)) + '\n')
        for var in inputList:
            fout.write(var + '\n')
            
        # Store scaler type and attributes (needed for variable transformation)
        fout.write(scaler_attributes)

        # Store number of layers
        fout.write( 'layers ' + str(len(model.layers)) + '\n')
        layers = []
        
        # Read models with different structure (Not sure why ".to_json() gives different results)
        if type(arch["config"]).__name__ == "list":
            config = arch["config"]
        else:
            config = arch["config"]["layers"]

        # Iterate over each layer
        for index, l in enumerate(config):
            # Store type of layer
            fout.write(l['class_name'] + '\n')
            #layers += [l['class_name']]

            # Convolution2D layer
            if l['class_name'] == 'Convolution2D':
                # Get weights of layer
                W = model.layers[index].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')

                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model.layers[index].get_weights()[1]) + '\n')

            # Activation layer
            if l['class_name'] == 'Activation':
                # Store activation function
                fout.write(l['config']['activation'] + '\n')

            # MaxPooling2D layer
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')

            # BatchNormalization layer
            if l['class_name'] == 'BatchNormalization':            
                W = model.layers[index].get_weights() #[list of arrays: gamma, beta, mean, variance] (https://github.com/keras-team/keras/issues/1523)
                cfg = model.layers[index].get_config()
                moving_mean  = model.layers[index].moving_mean.numpy()
                moving_var   = model.layers[index].moving_variance.numpy()
                gamma = model.layers[index].gamma.numpy()
                beta  = model.layers[index].beta.numpy()
                fout.write(str(len(W[0])) + '\n')
                fout.write('gamma\n'+ PrintArray(gamma) + '\n')
                fout.write('beta\n'+ PrintArray(beta) + '\n')
                fout.write('moving_mean\n' + PrintArray(moving_mean) + '\n')
                fout.write('moving_variance\n' + PrintArray(moving_var) + '\n')
                fout.write('epsilon\n' + str(cfg['epsilon']) + '\n')
                #fout.write(str(len(W)) + ' ' + str(len(W[0])) + '\n')
                #fout.write(PrintArray(W) + '\n')
                    
            # Dense layer
            if l['class_name'] == 'Dense':
                # Store number of inputs, outputs for each layer
                W = model.layers[index].get_weights()[0]                
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                
                for w in W:
                    # Store weights
                    fout.write(PrintArray(w) + '\n')
                # Store bias values (shifts the activation function : output[i] = (Sum(weights[i,j]*inputs[j]) + bias[i]))
                biases = model.layers[index].get_weights()[1]
                fout.write(PrintArray(biases) + '\n')

        if verbose:
            Print('Writing model in file %s' % (output), True)
        fout.close()
        return

def GetBestModel(directory):
    list_of_files = glob.glob('*%s/*h5' % directory)
    length = len("weights.")
    epochs = []
    for f in list_of_files:
        start = f.index("weights.")
        stop = f.index("-")
        ep = f[start+length:stop]
        ep = int(ep)
        epochs.append(ep)
        
    latest_epoch = str(max(epochs))
    latest_file = glob.glob('*%s/weights.%s*h5' % (directory, latest_epoch))
    
    if (len(latest_file) == 0):
        latest_epoch = "0%s"% latest_epoch
        latest_file = glob.glob('*%s/weights.%s*h5' % (directory, latest_epoch))

    #latest_file = max(list_of_files, key=os.path.getctime)
    #path = os.path.normpath(directory)
    #latest_file = tf.train.latest_checkpoint(path) #fixme! gives None
    return latest_file[0]


# MOVED FROM SEQUENTIAL
def PrintNetworkSummary(opts):
    table    = []
    msgAlign = "{:^10} {:^10} {:>12} {:>10}"
    title    =  msgAlign.format("Layer #", "Neurons", "Activation", "Type")
    hLine    = "="*len(title)
    table.append(hLine)
    table.append(title)
    table.append(hLine)
    for i, n in enumerate(opts.neurons, 0): 
        layerType = "unknown"
        if i == 0:
            #layerType = "input" # is this an input layer or hidden?
            layerType = "hidden"
        elif i+1 == len(opts.neurons):
            layerType = "output"
        else:
            layerType = "hidden"
        table.append( msgAlign.format(i+1, opts.neurons[i], opts.activation[i], layerType) )
    table.append("")

    Print("Will construct a DNN with the following architecture", True)
    for r in table:
        Print(r, False)
    return

def GetDataFramesRowsColumns(df_sig, df_bkg):
    nEntries_sig, nColumns_sig = GetDataFrameRowsColumns(df_sig)
    nEntries_bkg, nColumns_bkg = GetDataFrameRowsColumns(df_bkg)

    if nColumns_sig != nColumns_bkg:
        msg = "The number of columns for signal (%d variables) does not match the correspondigng number for background (%d variables)" % (nColumns_sig, nColumns_bkg)
        raise Exception(es + msg + ns)
    else:
        msg  = "The signal has a total %s%d entries%s for each variable." % (ts, nEntries_sig, ns)
        msg += "The background has %s %d entries%s for each variable. " % (hs, nEntries_bkg, ns)
        msg += "(%d variables in total)" % (nColumns_sig)
        Verbose(msg, True)

    return nEntries_sig, nColumns_sig, nEntries_bkg, nColumns_bkg

def GetDataFrameRowsColumns(df):
    '''
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    '''

    # Get the index (row labels) of the DataFrame object (df).  
    rows = df.index.values
    Verbose("Printing indices (i.e. rows) of DataFrame:\n%s" % (rows), True)

    # Get the data types in the DataFrame
    dtypes = df.dtypes
    Verbose("Printing dtypes of DataFrame:\n%s" % (dtypes), True)

    # Get the columns (labels) of the DataFrame.
    columns = df.columns.values 
    Verbose("Printing column labels of DataFrame:\n%s" % (columns), True)

    nRows    = rows.size    #len(rows)     # NOTE: This is the number of Entries for each TBranch (variable) in the TTree
    nColumns = columns.size #len(columns)  # NOTE: This is the number of TBranches (i.e. variables) in the TTree
    Verbose("DataFrame has %s%d rows%s and %s%d columns%s" % (ts, nRows, ns, hs, nColumns, ns), True)
    return nRows, nColumns

def GetModelWeightsAndBiases(inputList, neuronsList):
    '''
    https://keras.io/models/about-keras-models/

    returns a dictionary containing the configuration of the model. 
    The model can be reinstantiated from its config via:
    model = Model.from_config(config)
    '''
    nParamsT  = 0
    nBiasT    = 0 
    nWeightsT = 0
    nInputs   = len(inputList)

    for i, n in enumerate(neuronsList, 0):
        nParams = 0
        nBias   = neuronsList[i]
        if i == 0:
            nWeights = nInputs * neuronsList[i]
        else:
            nWeights = neuronsList[i-1] * neuronsList[i]
        nParams += nBias + nWeights

        nParamsT  += nParams
        nWeightsT += nWeights
        nBiasT += nBias
    return nParamsT, nWeightsT, nBiasT
        
def GetModelParams(model):
    '''
    https://stackoverflow.com/questions/45046525/how-can-i-get-the-number-of-trainable-parameters-of-a-model-in-keras
    '''
    # For older keras versions check the link above
    trainable_count = numpy.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = numpy.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    total_count = trainable_count + non_trainable_count
    return total_count, trainable_count, non_trainable_count

def GetModelConfiguration(myModel, verbose):
    '''
    NOTE: For some older releases of Keras the
    Model.get_config() method returns a list instead of a dictionary.
    This is fixed in newer releases. See details here:
    https://github.com/keras-team/keras/pull/11133
    '''
    
    config = myModel.get_config()
    for i, c in enumerate(config, 1):
        Verbose( "%d) %s " % (i, c), True)
    return config

def GetKwargs(var, standardise=False):

    kwargs  = {
        "normalizeToOne": True,
        "xTitle" : "DNN output",
        "yTitle" : "a.u.",
        "xMin"   :  0.0,
        "xMax"   : +1.0,
        "nBins"  : 200,
        "log"    : True,
        }

    if "output" in var.lower() or var.lower() == "output":
        kwargs["normalizeToOne"] = True
        kwargs["nBins"]  = 50
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   =  1.0
        kwargs["yMin"]   =  1e-5
        kwargs["yMax"]   =  2.0
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "a.u."
        kwargs["log"]    = True
        if var.lower() == "output":
            kwargs["legHeader"]  = "all data"
        if var.lower() == "outputpred":
            kwargs["legHeader"]  = "all data"
            kwargs["legEntries"] = ["train", "test"]
        if var.lower() == "outputtrain":
            kwargs["legHeader"]  = "training data"
        if var.lower() == "outputtest":
            kwargs["legHeader"]  = "test data"
        return kwargs

    if "efficiency" in var:
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "efficiency" # (\varepsilon)"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if "roc" in var:
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "signal efficiency"
        kwargs["yTitle"] = "background efficiency"
        kwargs["yMin"]   = 1e-4
        kwargs["yMax"]   = 10
        kwargs["log"]    = True

    if var == "significance":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "significance"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "loss":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.
        kwargs["xMax"]   = opts.epochs
        kwargs["nBins"]  = opts.epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "loss"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "acc":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   = opts.epochs
        kwargs["nBins"]  = opts.epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "accuracy"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "auc":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   = opts.epochs
        kwargs["nBins"]  = opts.epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "auc"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False
        
    if var == "trijetMass":
        kwargs["xMin"]   =   0.0
        kwargs["xMax"]   = 900 #900.0
        kwargs["nBins"]  = 180 #450
        kwargs["xTitle"] = "m_{t} [GeV]"
        kwargs["yMin"]   = 1e-3
        kwargs["yMax"]   = 1.0
        #kwargs["yTitle"] = "a.u. / %0.0f GeV"

    if standardise:
        kwargs["xMin"]  =  -5.0
        kwargs["xMax"]  =  +5.0
        kwargs["nBins"] = 500 #1000

    return kwargs

def SaveModelParameters(myModel, opts):
    nParams = myModel.count_params()
    total_count, trainable_count, non_trainable_count  = GetModelParams(myModel)
    nParams, nWeights, nBias = GetModelWeightsAndBiases(opts.inputList, opts.neurons)
    opts.modelParams = total_count
    opts.modelParamsTrain = trainable_count
    opts.modelParamsNonTrainable = non_trainable_count
    opts.modelWeights = nWeights
    opts.modelBiases  = nBias

    ind  = "{:<6} {:<30}"
    msg  = "The model has a total of %s%d parameters%s (neuron weights and biases):" % (hs, opts.modelParams, ns)
    msg += "\n\t" + ind.format("%d" % opts.modelWeights, "Weights")
    msg += "\n\t" + ind.format("%d" % opts.modelBiases, "Biases")
    msg += "\n\t" + ind.format("%d" % opts.modelParamsTrain, "Trainable")
    msg += "\n\t" + ind.format("%d" % opts.modelParamsNonTrainable, "Non-trainable")
    Print(msg, True)
    return

