#!/usr/bin/env python

from coffea import lookup_tools
import numpy as np

class PileupWeight:
    def __init__(self, hDataPU, hMCPU):
        puD = hDataPU.values()
        puD = np.divide(puD,puD.sum()) # normalize area=1

        puMC = hMCPU.values()
        puMC = np.divide(puMC,puMC.sum()) # normalize area=1

        puD = np.where(puMC == 0, 0, puD) # remove elements where MC gives zero, in order to have weight zero
        puMC = np.add(puMC,1.e-20) # add epsilon to prevent division by zero

        self.PUfuncD = lookup_tools.dense_lookup.dense_lookup(puD, hDataPU.axis(0).edges())
        self.PUfuncMC= lookup_tools.dense_lookup.dense_lookup(puMC, hDataPU.axis(0).edges())

        """
        pu = hDataPU.axis(0).edges()
        print("check puD",puD)
        print("check puMC",puMC)
        print("check puWeight",puWeight)
        print("check axis",pu)
        print("check puWeight",self.PUfunc(20))
        """

    def getWeight(self, nInteractions, var=0):
        nData=self.PUfuncD(nInteractions)
        nMC  =self.PUfuncMC(nInteractions)
        weights = np.divide(nData,nMC)
        return weights
