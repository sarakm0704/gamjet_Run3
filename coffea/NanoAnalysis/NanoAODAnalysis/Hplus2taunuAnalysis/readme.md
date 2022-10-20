## Installation with python venv
```
python -m venv VirtualTestEnvironment
source VirtualTestEnvironment/bin/activate
pip install wheel
pip install xrootd
pip install numpy==1.20
pip install coffea
```


## Alternative: installation with conda environment

1. Install miniconda

        $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        $ bash Miniconda3-latest-Linux-x86_64.sh

2. close and reopen terminal for the changes to take effect

3. (Optional) copy your grid certificate files onto your machine, here is demonstrated the situation where you have your certificates already configured on lxplus

        $ scp -r <CERNusername>@lxplus.cern.ch:~/.globus ~/. 

4. create conda environment from the template included in this repository

        $ conda config --set channel_priority strict
        $ conda env create -f <PATH_TO_YMLFILE>/conda_env.yml
        $ conda activate <my-environment>

5. Install latest coffea

        $ git clone https://github.com/CoffeaTeam/coffea.git
        $ cd coffea
        $ pip install .

6. Local grid access (Not applicable for lxplus)

        $ mkdir localgrid
        $ cd localgrid/
        $ git clone https://github.com/dmwm/dasgoclient.git
        $ cd dasgoclient
        $ make build_all
        $ cd ~
        $ mkdir .grid-security
        $ scp -r <CERNusername>@lxplus.cern.ch:/cvmfs/grid.cern.ch/etc/grid-security/* .grid-security
        $ voms-proxy-init --voms cms --vomses ~/.grid-security/vomses -valid 192:00

7. Install TensorFlow (with optional GPU support)
    1. CUDA support (for NVIDIA GPUS)
        - make sure you have access to a GPU on the machine by running

                $ nvidia-smi

        - The output should show information about your GPU, including power draw, utilization and driver version.
        - if the above step didn't work, make sure you have an nvidia gpu installed on the machine with the appropriate drivers from [nvidia](https://www.nvidia.com/Download/index.aspx) and that your user has access to it.
        - Install Tensorflow-compatible CUDA toolkit and cuDNN versions
                
                $ conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
                $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

        - Automate path configurations
                
                $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
                $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

    2. TensorFlow Installation

            $ pip install --upgrade pip
            $ pip install tensorflow
            $ pip install tensorflow-probability
            $ pip install tensorflow-addons

8. If you find that modules are still missing after following this guide or run into other issues, please open an issue or better yet, modify this readme yourself and create a merge request!

9. Dependencies for scaling analysis to CERN batch (Only on lxplus)

        $ conda install dask dask-jobqueue

## Running the analysis (event selection)

- First activate your environment if it is not already activated

        $ conda activate <CONDA_ENV_NAME>
or
        $ source VirtualTestEnvironment/bin/activate

- Training neural DNN classifier (Optional)

    1. save training data

            $ python ANN/datasetUtils.py <multicrabdir> <outdir> [<k_folds>] [<nn_input_variables>]

    2. train a model

            $ python ANN/disCo.py --dataDir <datasetUtils_outdir> [options]

- For distributed analysis, initilize a voms proxy in advance. The analysis script will transfer it to a location accessible by the workers

        $ voms-proxy-init -voms cms --valid 192:00

- Running the event selection and histograms

        $ python analysis.py <multicrabdir> [<trained_nn_path> <nn_score_cut>]

- Running with memory profiling

        $ pip install memory_profiler
        $ python -m memory_profiler analysis.py <multicrabdir> [...]

## Running limit calculation with the modified frequentist CLs method.

Work in progress
