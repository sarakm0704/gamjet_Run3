import sys

class Btag():
    def __init__(self,algorithm,year):
        self.algo = algorithm
        self.year = year
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self.workingPoints = {}
        self.workingPoints['btagDeepB'] = {
            "2016": {'loose': 0.2217, 'medium': 0.6321, 'tight': 0.8953},
            "2017": {'loose': 0.1522, 'medium': 0.4941, 'tight': 0.8001},
            "2018": {'loose': 0.1241, 'medium': 0.4184, 'tight': 0.7527},
            "2022": {'loose': 0.1241, 'medium': 0.4184, 'tight': 0.7527} # FIXME: using 2018 values for 2022
        }

        frac = 0.5
        self.workingPoints['btagDeepC'] = {
            #"2016": {'loose': -0.48, 'medium': -0.1, 'tight': 0.69},
            #"2017": {'loose':  0.05, 'medium': 0.15, 'tight': 0.8},
            #"2018": {'loose':  0.04, 'medium': 0.137,'tight': 0.66}
            # loosening the tight WP to get about same nember of c jets as b jets
            "2016": {'loose': -0.48, 'medium': -0.1, 'tight': -0.1+frac*(0.69+0.1)},
            "2017": {'loose':  0.05, 'medium': 0.15, 'tight': 0.15+frac*(0.8-0.15)},
            "2018": {'loose':  0.04, 'medium': 0.137,'tight': 0.137+frac*(0.66-0.137)},
            "2022": {'loose':  0.04, 'medium': 0.137,'tight': 0.137+frac*(0.66-0.137)} # FIXME: using 2018 values for 2022
        }

    def exists(self):
        if not self.algo in self.workingPoints.keys():
            print("B tagging algo",self.algo,"not found. Available algorithms:",self.workingPoints.keys())
            sys.exit()
        if not self.year in self.workingPoints[self.algo].keys():
            print("B tagging year",self.year,"not found. Available years:",self.algo,":",self.workingPoints[self.algo].keys())
            sys.exit()
        return True

    def loose(self):
        if self.exists():
            return self.workingPoints[self.algo][self.year]['loose']
        return None

    def medium(self):
        if self.exists():
            return self.workingPoints[self.algo][self.year]['medium']
        return None

    def tight(self):
        if self.exists():
            return self.workingPoints[self.algo][self.year]['tight']
        return None
