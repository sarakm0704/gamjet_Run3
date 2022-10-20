import sys,os

MYSYSPATH = 'NanoAnalysis/NanoAODAnalysis/Framework/python'

def getDataPath():
    for syspath in reversed(sys.path):
        if MYSYSPATH in syspath:
            return syspath.replace('python','data')

    print("Data path not found, determined in %s"%os.path.join(MYSYSPATH,'DataPath.py'))
    sys.exit()

def getSkimDataPath():
    for syspath in reversed(sys.path):
        if MYSYSPATH in syspath:
            return syspath.replace('NanoAODAnalysis/Framework/python','NanoAODSkim/data')

    print("Data path not found, determined in %s"%os.path.join(MYSYSPATH,'DataPath.py'))
    sys.exit()

def getSkimPythonPath():
    for syspath in reversed(sys.path):
        if MYSYSPATH in syspath:
            return syspath.replace('NanoAODAnalysis/Framework/python','NanoAODSkim/python')

    print("Skim python path not found, determined in %s"%os.path.join(MYSYSPATH,'DataPath.py'))
    sys.exit()
