import numpy as np
import awkward as ak
import math

def Event(events):
    pickEvents=[]
    pickEvents.append("297056:177:248907287")
    pickEvents.append("297056:178:250163351")
    pickEvents.append("297056:180:252822338")
    pickEvents.append("297056:181:253865652")
    pickEvents.append("297056:181:254020229")

    pickEvents.append("297056:182:255656162")
    pickEvents.append("297056:182:255884447")
    pickEvents.append("297056:183:257097252")
    pickEvents.append("297056:185:259886113")
    pickEvents.append("297056:187:262592061")
    
    pickEvents.append("297056:187:263232787")  
    pickEvents.append("297056:188:264096365")
    pickEvents.append("297056:188:264025010")
    pickEvents.append("297056:189:265645544")
    pickEvents.append("297056:190:267653858")

    phiBB_cut = 0.44 # 0.34

    for i,p in enumerate(pickEvents):
        run,lumi,event = p.split(":")
        run = int(run)
        lumi=int(lumi)
        event=int(event)

        pEvents = events[(events.run == run) & (events.luminosityBlock == lumi) & (events.event == event)]
        print("Event",i,p)
        #print("Muons pt",pEvents.Muon.pt)
        #print("Muons tightId",pEvents.Muon.tightId)
        #print("Muons pfRelIso04_all",pEvents.Muon.pfRelIso04_all)
        print("jets pt,eta,phi",pEvents.Jet.pt,pEvents.Jet.eta,pEvents.Jet.phi)
        jets = pEvents.Jet
        sortedJets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
#        sort_index = ak.sort(pEvents.Jet)
#        print("sort_index",sort_index)
#        sortedJets = pEvents.Jet[pEvents.Jet.pt.argsort()] #jets[sort_index[:,0]]#ak.sort(pEvents.Jet,ascending=False)
        print("jets sorted",sortedJets.pt,sortedJets.eta,sortedJets.phi)
        #print("ljet phi",pEvents.leadingJet.phi)
        #print("Z phi",pEvents.Zboson.phi)
        #print("dphi",pEvents.leadingJet.delta_phi(pEvents.Zboson))

        #phiBB = pEvents.leadingJet.delta_phi(pEvents.Zboson) - math.pi
        #print("check phiBB",phiBB)
        phiBB = abs(pEvents.Zboson.delta_phi(pEvents.leadingJet) - math.pi)
        decision = (phiBB > phiBB_cut) & (phiBB < 2*math.pi-phiBB_cut)
        print("check phiBB",phiBB,pEvents.leadingJet.phi,pEvents.Zboson.phi,decision)

