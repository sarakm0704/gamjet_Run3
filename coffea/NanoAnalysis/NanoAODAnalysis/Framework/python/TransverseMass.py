import awkward as ak
import numpy as np

def reconstruct_transverse_mass(tau, met):
    tau_mt = np.sqrt(tau.mass**2 + tau.pt**2)  # get tau transverse mass
    myCosPhi = (tau.x*met.x + tau.y*met.y) / (tau.pt*met.pt) # get cosine of angle between tau and met
    myCosPhi = np.clip(myCosPhi, -1.0, 1.0) # get rid of fp inaccuracy causing unphysical cosine

    mt = np.sqrt(2*tau.pt*met.pt*(1.0-myCosPhi)) # get mt between tau and met
    # mt = ak.where(np.isnan(myMt), tau_mt, myMt)
    # if ak.any(np.isnan(mt)):
    #     msg = f"Encountered nan when calculating event transverse mass with input:\ntaus:\n{tau}\nMET\n{met}"
    #     raise Exception(msg)
    return mt

