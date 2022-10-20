from coffea import processor
import awkward as ak

class ModularCounters():
    def __init__(self):
        self.counters = {}
        self.counters["unweighted_counter"] = {}
        self.counters["weighted_counter"]   = {}

    def set(self, label, events):
        self.counters["unweighted_counter"][label] = len(events)
        self.counters["weighted_counter"][label] = ak.sum(events.weight)
        return self

    def get(self):
        return self.counters

    def set_skim(self, counterhisto):
        for i in range(1,counterhisto.GetNbinsX()+1):
            self.counters["unweighted_counter"][counterhisto.GetXaxis().GetBinLabel(i)] = int(counterhisto.GetBinContent(i))
            self.counters["weighted_counter"][counterhisto.GetXaxis().GetBinLabel(i)]   = int(counterhisto.GetBinContent(i))
        return self

    @staticmethod
    def print(accumulator, names_to_print):
        names = [n.lower() for n in names_to_print]
        neg_identifiers = {
            "inclusive": ["pt ", "eta "]
        }
        for name in names:
            if name not in neg_identifiers:
                neg_identifiers[name] = []

        pos_identifiers = names.copy()
        pos_identifiers.remove('inclusive')

        print("    Counters                  unw.           weighted")
        for k in accumulator['unweighted_counter'].keys():
            if any(any(i in k.lower() for i in neg_identifiers[name]) for name in names): continue
            if pos_identifiers and not any(i in k.lower() for i in pos_identifiers): continue
            counter = "     "+k
            while len(counter) < 60:
                counter+=" "
            counter +="%s"%accumulator['unweighted_counter'][k]
            while len(counter) < 75:
                counter+=" "
            counter +="%s"%round(accumulator['weighted_counter'][k],1)
            print(counter)


class Counters():
    def __init__(self):
        self.counters = {}
        self.counters["unweighted_counter"] = {}
        self.counters["weighted_counter"]   = {}
        self.first = True

    def __add__(self, x):
        for key in self.counters["unweighted_counter"].keys():
            self.counters['unweighted_counter'][key] += x.counters['unweighted_counter'][key]
            self.counters['weighted_counter'][key] += x.counters['weighted_counter'][key]
        return self

    def book(self,counterhisto):
        self.counterhisto = counterhisto
            
    def setAccumulatorIdentity(self,identity):
        self.identity = identity

    def setSkimCounter(self):
        if not self.first or not hasattr(self,'counterhisto'): return
        for i in range(1,self.counterhisto.GetNbinsX()+1):
            self.counters["unweighted_counter"][self.counterhisto.GetXaxis().GetBinLabel(i)] = int(self.counterhisto.GetBinContent(i))
            self.counters["weighted_counter"][self.counterhisto.GetXaxis().GetBinLabel(i)]   = int(self.counterhisto.GetBinContent(i))
        self.first = False

    def increment(self,label,events):
        if not label in list(self.counters["unweighted_counter"].keys()):
            self.counters["unweighted_counter"][label] = 0
            self.counters["weighted_counter"][label]   = 0
        self.counters["unweighted_counter"][label] += len(events.event)
        self.counters["weighted_counter"][label]   += ak.sum(events.weight)

    def get(self):
        return self.counters

    def print(self):
        print("    Counters                  unw.           weighted")
        for k in self.counters['unweighted_counter'].keys():
            counter = "     "+k
            while len(counter) < 30:
                counter+=" "
            counter +="%s"%self.counters['unweighted_counter'][k]
            while len(counter) < 45:
                counter+=" "
            counter +="%s"%round(self.counters['weighted_counter'][k],1)
            print(counter)

    def histo(self):
        import ROOT
        n = len(self.counters['unweighted_counter'].keys())
        h_uCounter = ROOT.TH1F("unweighted_counter","",n,1,n)
        h_wCounter = ROOT.TH1F("weighted_counter","",n,1,n)
        for i,k in enumerate(self.counters['unweighted_counter'].keys()):
            h_uCounter.GetXaxis().SetBinLabel(i+1,k)
            h_uCounter.SetBinContent(i+1,self.counters['unweighted_counter'][k])
            h_wCounter.GetXaxis().SetBinLabel(i+1,k)
            h_wCounter.SetBinContent(i+1,self.counters['weighted_counter'][k])
        return h_uCounter,h_wCounter

# work in progress, counters not priority
class VariationCounters(Counters):

    def __init__(self, variations):
        self.variations = variations
        self.counters = {}
        self.counters["unweighted_counter"] = processor.defaultdict_accumulator(processor.defaultdict_accumulator)
        self.counters["weighted_counter"]   = processor.defaultdict_accumulator(processor.defaultdict_accumulator)
        self.first = True

        self.selection = {}
        self.weights = {}

        variations = []
        varSelections = {}

        for key in var.keys():
            variations.append(list(var[key].keys()))
            for k2 in var[key].keys():
                varSelections[k2] = var[key][k2]

        nameBase = name
        for combination in list(itertools.product(*variations)):
            hname = nameBase
            sele = None

            for comb in combination:
                if not (comb == '' or 'incl' in comb or comb == 'PSWeight'):
                    hname = hname + '_%s'%comb

                    if 'weight' in comb or 'Weight' in comb:
                        self.weights[hname] = comb
                        continue
                    if sele == None:
                        sele = varSelections[comb]
                    else:
                        sele = sele + " & " + varSelections[comb]
            self.selection[hname] = sele

        self.selection_re = re.compile("(?P<variable>[\w\.]+)\s*(?P<operator>\S+)\s*(?P<value>\S+)\)")

    def increment(self, label, events):
        for name in self.selection:
            sele = self.select(events, name)
            self.identity["unweighted_counter"][name][label] += len(events.event)
            self.identity["weighted_counter"][name][label]   += ak.sum(events.weight)

    def select(events, key):
        return_sele = (events["event"] > 0)
        if not self.is_unblinded:
            # NOTE: this currently automatically unblinds all events that passed the INVERTED tau isolation check
            return_sele = return_sele & (events["TauIsolation"] <= 0)
        if self.selection[variable] == None:
            return return_sele

        selections = self.selection[variable].split('&')

        for s in selections:
            match = self.selection_re.search(s)
            if match:
                rec_variables = match.group('variable').split('.')
                operator = match.group('operator')
                value    = eval(match.group('value'))

                variable = events
                for field in rec_variables:
                    variable = variable[field]

                if operator == '<':
                    return_sele = return_sele & (variable < value)
                if operator == '>':
                    return_sele = return_sele & (variable > value)
                if operator == '==':
                    return_sele = return_sele & (variable == value)
        return return_sele
