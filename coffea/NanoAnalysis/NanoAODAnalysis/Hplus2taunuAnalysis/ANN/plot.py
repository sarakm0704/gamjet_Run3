import ROOT
import array
import os
import getpass

def CreateCanvas():
    canvas = ROOT.TCanvas()
    return canvas

def CreateLegend(xmin=0.55, ymin=0.75, xmax=0.85, ymax=0.85):
    leg = ROOT.TLegend(xmin, ymin, xmax, ymax)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.040)
    leg.SetTextFont(42)
    #leg.SetTextSize(0.09)
    leg.SetLineColor(1)
    leg.SetLineStyle(1)
    leg.SetLineWidth(1)
    leg.SetFillColor(0)
    return leg

def Text(text, x=0.3, y=0.8):
    tex = ROOT.TLatex(x, y, text)
    tex.SetNDC()
    tex.SetTextAlign(31)
    tex.SetTextFont(42)
    tex.SetTextSize(0.030)
    tex.SetLineWidth(2)
    return tex

def CreateGraph(gx, gy):
    graph=ROOT.TGraph(2, array.array("d",gx), array.array("d",gy)) 
    graph.SetFillColor(1)
    graph.SetLineColor(ROOT.kBlack)
    graph.SetLineStyle(3)
    graph.SetLineWidth(2)
    return graph

def GetGraph(x, y, xerrl, xerrh, yerrl, yerrh):
    graph = ROOT.TGraphAsymmErrors(len(x), array.array("d",x), array.array("d",y),
                                   array.array("d",xerrl), array.array("d",xerrh),
                                   array.array("d",yerrl), array.array("d",yerrh))
    return graph


# from https://github.com/cms-analysis/CombineHarvester/blob/master/CombineTools/python/plotting.py
def TwoPadSplit(split_point, gap_low, gap_high):
    upper = ROOT.TPad('upper', 'upper', 0., 0., 1., 1.)
    upper.SetBottomMargin(split_point + gap_high)
    upper.SetFillStyle(4000)
    upper.Draw()
    lower = ROOT.TPad('lower', 'lower', 0., 0., 1., 1.)
    lower.SetTopMargin(1 - split_point + gap_low)
    lower.SetFillStyle(4000)
    lower.Draw()
    upper.cd()
    result = [upper, lower]
    return result

def GetRatioStyle(h_ratio, ytitle, xtitle, ymax=2, ymin=0):
    h_ratio.SetMaximum(ymax)
    h_ratio.SetMinimum(ymin)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle(ytitle)
    h_ratio.GetXaxis().SetTitle(xtitle)
    h_ratio.GetYaxis().SetLabelSize(0.09)
    h_ratio.GetXaxis().SetLabelSize(0.09)
    h_ratio.GetYaxis().SetTitleSize(0.095)
    h_ratio.GetXaxis().SetTitleSize(0.095)
    h_ratio.GetXaxis().SetTickLength(0.08)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    return h_ratio

def getDirName(dirName, baseDir=None):
    #print("getDirName() => FIXME! Do not assume full save path")
    usrName = getpass.getuser() 
    usrInit = usrName[0]
    dirName = dirName.replace(".", "p")
    #dirName = "/afs/cern.ch/user/%s/%s/public/html/%s" % (usrInit, usrName, dirName)
    return dirName

def CreateDir(saveDir):
    # Create output directory if it does not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        if 0:
            print("Directory " , saveDir ,  " has been created ")
    else:
        if 0:
            print("Output saved under", saveDir)
    return

def SavePlot(canvas, saveDir, saveName, saveFormats=["pdf"], verbose=False):
    
    # Create output directory if it does not exist
    CreateDir(saveDir)

    savePath = "%s/%s" % (saveDir, saveName)
    usrInit =  getpass.getuser()[0]
    saveURL  = savePath.replace("/afs/cern.ch/user/%s/" %(usrInit),"https://cmsdoc.cern.ch/~")
    saveURL  = saveURL.replace("/public/html/","/")
    
    for ext in saveFormats:
        fileName = "%s.%s" % (savePath, ext)
        canvas.SaveAs( fileName )
        if verbose:
            print("=== ","%s.%s" % (saveURL, ext))
    return

def ApplyStyle(h, color, line=ROOT.kSolid):
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(8)
    h.SetMarkerSize(0.5)
    h.SetLineStyle(line)
    h.SetLineWidth(3)
    h.SetTitle("")
    return


def GetHistoInfo(var):
    h = var.lower()
    xmin = 0
    xmax = 1000
    units = ""
    xlabel = var
    width   = 10 # binwidth
    _format = "%0.0f "

    if "phi" in h:
        _format = "%0.2f "
        xmin = -4
        xmax = 4
        xlabel = "delta #Phi"
        width = 0.10

    if "r_tau" in h:
        _format = "%0.2f "
        xmin = 0
        xmax = 1
        xlabel = "R_{#tau}"
        width = 0.05

    if "eta" in h:
        _format = "%0.2f "
        xmin = -5
        xmax = 5
        xlabel = "#eta"
        width = 0.10
        
    if "mass" in h:
        units   = "GeV/c^{2}"
        xlabel  = "M (%s)" % units
        _format = "%0.0f "
        width   = 10
        
    if "trijetmass" in h:
        units  = "GeV/c^{2}"
        xlabel = "m_{top} (%s)" % units
        xmax = 805 #1005

    if "trijetpt" in h:
        units  = "GeV/c"
        xlabel = "p_{T} (%s)" % units
        xmax = 805 #1005
        
    if "dijetmass" in h:
        xlabel = "m_{W} (%s)" % units
        xmax   = 600
        width  = 10

    if "bjetmass" in h:
        xlabel = "m_{b-tagged jet} (%s)" % units
        width = 1
        xmax = 50

    if "bjetldgjetmass" in h:
        xlabel = "m_{b, ldg jet} (%s)" % units
        xmax = 705

    if "bjetsubldgjetmass" in h:
        xlabel = "m_{b-tagged, subldg jet} (%s)" % units
        xmax = 705

    if "jet_mass" in h:
        xmax = 750

    if "mult" in h:
        _format = "%0.0f "
        width = 1
        xmax = 50
        if "ldg" in h:
            xlabel = "Leading jet mult"
        if "subldg" in h:
            xlabel = "Subleading jet mult"

    if "cvsl" in h:
        _format = "%0.2f "
        width  = 0.01
        xmax = 1
        xmin = -1
        if "ldg" in h:
            xlabel = "Leading jet CvsL"
        if "subldg" in h:
             xlabel = "Subleading jet CvsL"

    if "axis2" in h:
        _format = "%0.3f "
        width = 0.004
        xmax = 0.2
        if "ldg" in h:
            xlabel = "Leading jet axis2"
        if "subldg" in h:
            xlabel = "Subleading jet axis2"

    if "trijetptdr" in h:
        xmax =800
        _format = "%0.0f "
        xlabel = "p_{T}#Delta R_{t}"

    if "dijetptdr" in h:
        xmax =800
        _format = "%0.0f "
        xlabel = "p_{T}#Delta R_{W}"

    if "dgjetptd" in h:
        _format = "%0.2f "
        xlabel = "Leading jet p_{T}D"
        width = 0.01
        xmax = 1
        if "subldg" in h:
            xlabel = "Subleading jet p_{T}D"

    if "bdisc" in h:
        _format = "%0.2f "
        width   = 0.01
        xmax = 1
        if "subldg" in h:
            xlabel = "Subleading jet CSV"
        elif "ldg" in h:
            xlabel = "Leading jet CSV"
        else:
            xlabel = "b-tagged jet CSV"

    if "softdrop" in h:
        xmax = 2
        width = 0.04
        _format = "%0.2f "
        xlabel = "SoftDrop_n2"
    if not (units == ""):
        units = " (%s)" % units
        
    nbins = (xmax - xmin) / width
    nbins = int(nbins)

    _format = _format % width + units
    ylabel = "Arbitrary Units / " + _format
    info     = {
        "xmin"   : xmin,
        "xmax"   : xmax,
        "nbins"  : nbins,
        "units"  : units,
        "xlabel" : xlabel,
        "ylabel" : ylabel,
    }
    return info


def GetLabel(label):
    '''
    Get label for each variable - Used in plotting the correlation matrix
    '''
    label = label.replace("cosW_Jet1Jet2", "cos#omega(j1,j2)")
    label = label.replace("cosW_Jet1BJet", "cos#omega(j1,b)")
    label = label.replace("cosW_Jet2BJet", "cos#omega(j2,b)")
    label = label.replace("LdgJetDeltaPtOverSumPt","#Delta p_{T}(j1,t)/(p_{T,j1}+p_{T,t})")
    label = label.replace("SubldgJetDeltaPtOverSumPt","#Delta p_{T}(j2,t)/(p_{T,j2}+p_{T,t})")
    label = label.replace("bjetDeltaPtOverSumPt","#Delta p_{T}(b,t)/(p_{T,b}+p_{T,t})")
    label = label.replace("LdgJetPtTopCM","p_{T,j1}(topCM)")
    label = label.replace("SubldgJetPtTopCM","p_{T,j2}(topCM)")
    label = label.replace("bjetPtTopCM","p_{T,b}(topCM)")
    label = label.replace("dijetPtOverSumPt", "p_{T,W}/(p_{T,j1}+p_{T,j2})")
    label = label.replace("DEtaDijetwithBJet", "#Delta #eta(W,b)")
    label = label.replace("LdgJetCvsB","CvsB_j1")
    label = label.replace("SubldgJetCvsB","CvsB_j2")
    label = label.replace("bjetCvsB","CvsB_b")
    label = label.replace("LdgJetCvsL","CvsL_j1")
    label = label.replace("SubldgJetCvsL","CvsL_j2")
    label = label.replace("bjetCvsL","CvsL_b")
    label = label.replace("LdgJetMultCharged","multC_j1")
    label = label.replace("SubldgJetMultCharged","multC_j2")
    label = label.replace("bjetMultCharged","multC_b")
    label = label.replace("LdgJetMultNeutral","multN_j1")
    label = label.replace("SubldgJetMultNeutral","multN_j2")
    label = label.replace("bjetMultNeutral","multN_b")        
    label = label.replace("LdgJetMult","mult_j1")
    label = label.replace("SubldgJetMult","mult_j2")
    label = label.replace("bjetMult","mult_b")
    label = label.replace("LdgJetAxis1","axis1_j1")
    label = label.replace("SubldgJetAxis1","axis1_j2")
    label = label.replace("bjetAxis1","axis1_b")
    label = label.replace("LdgJetAxis2","axis2_j1")
    label = label.replace("SubldgJetAxis2","axis2_j2")
    label = label.replace("bjetAxis2","axis2_b")
    label = label.replace("LdgJetPtD","p_{T}D_j1")
    label = label.replace("SubldgJetPtD","p_{T}D_j2")
    label = label.replace("bjetPtD","p_{T}D_b")
    label = label.replace("SoftDrop_n2", "SD")
    label = label.replace("dijetMass","m_{W}")
    label = label.replace("trijetMass","m_{top}")
    label = label.replace("LdgJetMass","m_{j1}")
    label = label.replace("SubldgJetMass","m_{j2}")
    label = label.replace("bjetMass","m_{b}")
    label = label.replace("trijetPtDR", "p_{T} #Delta R_{T}")
    label = label.replace("dijetPtDR", "p_{T} #Delta R_{W}")
    label = label.replace("LdgJetBdisc","bDisc_j1")
    label = label.replace("SubldgJetBdisc","bDisc_j2")
    label = label.replace("bjetBdisc","bDisc_b")
    label = label.replace("LdgJetPullMagnitude", "pullM_j1")
    label = label.replace("SubldgJetPullMagnitude", "pullM_j2")
    label = label.replace("bjetPullMagnitude", "pullM_b")
    label = label.replace("LdgJetPFCharge", "chg_j1")
    label = label.replace("SubldgJetPFCharge", "chg_j2")
    label = label.replace("bjetPFCharge", "chg_b")
    label = label.replace("LdgJetDr2Mean", "Dr2_j1")
    label = label.replace("SubldgJetDr2Mean", "Dr2_j2")
    label = label.replace("bjetDr2Mean", "Dr2_b")
    label = label.replace("LdgJetEneFracCharged","eneFracC_j1")
    label = label.replace("SubldgJetEneFracCharged","eneFracC_j2")
    label = label.replace("bjetEneFracCharged","eneFracC_b")
    label = label.replace("LdgJetEneFracNeutral","eneFracN_j1")
    label = label.replace("SubldgJetEneFracNeutral","eneFracN_j2")
    label = label.replace("bjetEneFracNeutral","eneFracN_b")

    label = label.replace("tetrajetMass", "H^{#pm} mass")
    label = label.replace("tetrajetbjet_topbjet_mass", "m_{b_{t}, b_{H^{#pm}}}")
    label = label.replace("tetrajetbjet_topjet1_mass", "m_{j_{1}^{W}, b_{H^{#pm}}}")
    label = label.replace("tetrajetbjet_topjet2_mass", "m_{h_{2}^{W}, b_{H^{#pm}}}")
    label = label.replace("tetrajetbjetMass", "m_{b{H^{#pm}}}")
    label = label.replace("trijetMass", "m_{t}")
    label = label.replace("tetrajetPt", "H^{#pm} p_{T}")
    label = label.replace("trijetPt", "top p_{T}")
    label = label.replace("absdPt_tetrajet_trijet_Over_SumPt", "absdPt_tetrajet_trijet_Over_SumPt")
    label = label.replace("absdPt_tetrajet_tetrajetbjet_Over_SumPt", "absdPt_tetrajet_tetrajetbjet_Over_SumPt")
    label = label.replace("absdPt_trijet_tetrajetbjet_Over_SumPt", "absdPt_trijet_tetrajetbjet_Over_SumPt")
    label = label.replace("topjet1_HPlusCM", "j_{1}^{W} p_{T} (H^{#pm} CM)")
    label = label.replace("topjet2_HPlusCM", "j_{2}^{W} p_{T} (H^{#pm} CM)")
    label = label.replace("ldgjetPt", "Leading jet p_{T}")
    label = label.replace("WPt", "W p_{T}")
    label = label.replace("dEta_trijet_tetrajetbjet", "#Delta#eta(t, b_{H^{#pm}})")
    label = label.replace("dPhi_tetrajetbjet_trijetbjet", "#Delta#phi(b_{H^{#pm}}, b_{t})")
    label = label.replace("dPhi_trijet_tetrajetbjet", "#Delta#phi(t, b_{H^{#pm}})")
    label = label.replace("dEta_tetrajetbjet_trijetbjet", "#Delta#eta(b_{H^{#pm}}, b_{t})")
    label = label.replace("PhiA", "#phi_{#alpha}")
    label = label.replace("dW_topbjet_tetrajetbjet_HPlusCM", "#Delta#omega(b_{t}, b_{H^{#pm}})")
    label = label.replace("dW_topjet1_tetrajetbjet_HPlusCM", "#Delta#omega(j_{1}^{W}, b_{H^{#pm}})")
    label = label.replace("dW_topjet2_tetrajetbjet_HPlusCM", "#Delta#omega(j_{2}^{W}, b_{H^{#pm}})")
    label = label.replace("chargedMult_tetrajet", "H^{#pm} charged mult")
    label = label.replace("chargedHadronEnFr_tetrajetbjet", "b_{H^{#pm}} charged had. fraction")
    label = label.replace("chargedHadronEnFr_trijet", "top charged had. fraction")
    label = label.replace("neutralMult_tetrajet", "H^{#pm} neutral mult")
    label = label.replace("neutralHadronEnFr_tetrajetbjet", "b_{H^{#pm}} neutral had. fraction")
    label = label.replace("neutralHadronEnFr_trijet", "top neutral had. fraction")
    label = label.replace("tetrajetbjetCvsL", "b_{H^{#pm}} CvsL")
    label = label.replace("ptD_tetrajetbjet", "b_{H^{#pm}} p_{T}D")
    label = label.replace("axis2_tetrajetbjet", "b_{H^{#pm}} axis-2")
    label = label.replace("dR2Mean_tetrajetbjet", "b_{H^{#pm}} #Delta R mean")
    label = label.replace("dR2MeanSum_trijet", "top #Delta R mean")
    
    return label

def AddPreliminaryText():
    # Setting up preliminary text
    tex = ROOT.TLatex(0.,0., 'Preliminary');
    tex.SetNDC();
    tex.SetX(0.27);
    tex.SetY(0.95);
    tex.SetTextFont(53);
    tex.SetTextSize(28);
    tex.SetLineWidth(2)
    return tex

def AddLumiText():
    #tex = ROOT.TLatex(0.85,0.95," (13 TeV)");
    tex = ROOT.TLatex(0.95,0.95," 13 TeV");
    tex.SetNDC();
    tex.SetTextAlign(31);
    tex.SetTextFont(43);  
    tex.SetTextSize(24);
    tex.SetLineWidth(2);   
    return tex
    
def AddCMSText():
    # Settign up cms text
    texcms = ROOT.TLatex(0.,0., 'CMS');
    texcms.SetNDC();
    texcms.SetTextAlign(31);
    texcms.SetX(0.26);
    texcms.SetY(0.95);
    texcms.SetTextFont(63);
    texcms.SetLineWidth(2);
    texcms.SetTextSize(30);
    return texcms
