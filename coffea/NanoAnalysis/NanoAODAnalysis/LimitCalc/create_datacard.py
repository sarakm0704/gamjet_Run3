import json 

def create_datacard():
    datacard = {"lumi":
    {
        "type": "lnN",
         "ejets": {
            "tprime600" : "1.10",
        }, "mujets": {
            "tprime600" : "1.10",
        }
    },
    "bgnortop": {
        "type": "lnN",
         "ejets": {
            "top" : "1.114",
        }, "mujets": {
            "top" : "1.114",
        }
    },
    "bgnorewk": {
        "type": "lnN",
            "ejets": {
                "ewk" : "1.5",
            }, "mujets": {
                "ewk" : "1.5",
            }
    },
    "eff_mu": {
        "type": "lnN",
            "mujets": {
                "tprime600" : "1.03",
                "top": "1.03",
                "ewk": "1.03"
        }
    },
    "eff_e": {
        "type": "lnN",
            "ejets": {
                "tprime600" : "1.03",
                "top": "1.03",
                "ewk": "1.03"
        }
    },
    "jes": {
        "type": "shape"
    },
    "btgsf" : {
        "type": "shape"
    }
    }   

    data = json.dumps(datacard, indent=4)
    print(data)

    with open("test.json", "w") as f:
        f.write(data)

    print(type(data))

    return data

if __name__=="__main__":
    create_datacard()    
