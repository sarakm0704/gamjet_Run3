python -m venv VirtualTestEnvironment
source VirtualTestEnvironment/bin/activate
pip install wheel
pip install xrootd
pip install numpy==1.20
pip install coffea

python hplus2taunuAnalysis.py <multicrabdir>

