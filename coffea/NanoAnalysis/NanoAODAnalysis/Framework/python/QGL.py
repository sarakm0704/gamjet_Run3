# QGL code borrowed from K.Kallonen https://github.com/kimmokal/QGValidation/blob/main/scripts/zjets_analysis_data.py
import uproot
import numpy as np
import correctionlib

class qgl():
    def __init__(self,year):
        self.qgl_evaluator = correctionlib.highlevel.CorrectionSet.from_file('pdfQG_AK4chs_13TeV_UL17_ghosts.corr.json')
        self.fIN = uproot.open("pdfQG_ak4_13TeV_UL17_JMEnano_Total_rebinned_PUPPI.root")

        qgl_file = self.fIN
        qgl_rho_bins = qgl_file["rhoBins"].members["fElements"]
        qgl_eta_bins = qgl_file["etaBins"].members["fElements"]
        qgl_pt_bins = qgl_file["ptBins"].members["fElements"]

        qgl_rho_dict = {}
        qgl_eta_dict = {}
        qgl_pt_dict = {}

        for i in range(len(qgl_rho_bins)):
            qgl_rho_dict[i] = qgl_rho_bins[i]
        for i in range(len(qgl_eta_bins)):
            qgl_eta_dict[i] = qgl_eta_bins[i]
        for i in range(len(qgl_pt_bins)):
            qgl_pt_dict[i] = qgl_pt_bins[i]

        self.find_rho_bin = lambda x : self.find_qgl_bin(qgl_rho_dict, x)
        self.find_eta_bin = lambda x : self.find_qgl_bin(qgl_eta_dict, x)
        self.find_pt_bin = lambda x : self.find_qgl_bin(qgl_pt_dict, x)

    def find_qgl_bin(self,bins_dict, value):
        if (value < bins_dict[0]) or (value > bins_dict[len(bins_dict)-1]):
            return -1
        bin_num = 0
        while value > bins_dict[bin_num+1]:
            bin_num = bin_num+1
        return bin_num

    def compute(self,jets):

        jets["rho_bin"] = list(map(self.find_rho_bin, jets["rho"]))
        jets["eta_bin"] = list(map(self.find_eta_bin, np.abs(jets["eta"])))
        jets["pt_bin"] = list(map(self.find_pt_bin, jets["pt"]))

        qgl_jets = jets.to_list()
        #jets["qgl_new"] = list(map(self.compute_jet_qgl, qgl_jets))
        return list(map(self.compute_jet_qgl, qgl_jets))

    def compute_jet_qgl(self, jet):
        jet_rho_bin = jet["rho_bin"]
        jet_eta_bin = jet["eta_bin"]
        jet_pt_bin = jet["pt_bin"]

        if (jet_rho_bin < 0 or jet_eta_bin < 0 or jet_pt_bin < 0):
            return -1.
        if jet["qgl_axis2"] <= 0:
            jet_axis2 = 0
        else:
            jet_axis2 = -np.log(jet["qgl_axis2"])
        jet_mult = jet["qgl_mult"]
        jet_ptD = jet["qgl_ptD"]
            
        quark_likelihood = 1.
        gluon_likelihood = 1.

        for var in ["axis2", "ptD", "mult"]:
            quark_string = "{var_name}/{var_name}_quark_eta{bin1}_pt{bin2}_rho{bin3}".format(
                var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
            gluon_string = "{var_name}/{var_name}_gluon_eta{bin1}_pt{bin2}_rho{bin3}".format(
                var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
                
            if var == "axis2":
                input_var = jet_axis2
            if var == "ptD":
                input_var = jet_ptD
            if var == "mult":
                input_var = float(jet_mult)

            var_quark_likelihood = self.qgl_evaluator[quark_string].evaluate(input_var)
            var_gluon_likelihood = self.qgl_evaluator[gluon_string].evaluate(input_var)

            if (var_quark_likelihood < 0) or (var_gluon_likelihood < 0):
                return -1.

            quark_likelihood = quark_likelihood*var_quark_likelihood
            gluon_likelihood = gluon_likelihood*var_gluon_likelihood
        
        return round(quark_likelihood/(quark_likelihood+gluon_likelihood), 3)

