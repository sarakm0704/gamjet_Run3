# Virtual environment
python -m venv VirtualTestEnvironment
source VirtualTestEnvironment/bin/activate
pip install coffea

# Singularity container
singularity build root.sif docker://rootproject/root:6.26.06-conda
singularity shell --home <path/to/NanoAnalysis> -B /eos/cms/store/user/slehti root.sif
	    # Examples
	    dx7-hip-02 > singularity shell --home $HOME/test/NanoAnalysis -B /jme root.sif
	    lxplus748 > singularity shell --home $TMP/NanoAnalysis -B /eos/cms/store/user/slehti root.sif
pip install coffea

# Getting JEC/JER db's
cd ../Framework/data

git clone https://github.com/cms-jet/JECDatabase
git clone https://github.com/cms-jet/JRDatabase

# Getting Rochester corrections for muons
wget https://twiki.cern.ch/twiki/pub/CMS/RochcorMuon/roccor.Run2.v5.tgz
tar xfvz roccor*.tgz
cd -

analysis.py <multicrabdir>

