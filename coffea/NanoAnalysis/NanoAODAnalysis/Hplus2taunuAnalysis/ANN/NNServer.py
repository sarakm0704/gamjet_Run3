import datasetUtils
import sys
import subprocess as sp
import glob
import tensorflow as tf
from  tensorflow import keras as K
from disCo import InputSelector, Sanitizer, MinMaxScaler

CONTAINERNAME = "HplusModelServer"


def kill_server():
    p = sp.Popen("docker exec -it mycontainer", shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate()
    return out, err

def init_server(model_dir, server_name=CONTAINERNAME):
    folds = glob.glob(model_dir + "/fold_*/")
    k = len(folds)
    
    # write models in tf SavedModel format to /tmp
    custom_objects = {"InputSelector": InputSelector, "Sanitizer": Sanitizer, "MinMaxScaler": MinMaxScaler}
    models = [K.models.load_model(d + "model_trained.h5", custom_objects = custom_objects, compile=False) for d in folds]
    [tf.saved_model.save(m, f"/tmp/tfserving/HplusModel{i}/1/") for i, m in enumerate(models)]

    # define model serving config protobuf message to look for all models
    cfg='model_config_list: {\n'\
        '\n'
    for i, d in enumerate(folds):
        m_cfg = '  config: {\n'\
               f'    name: "HplusModel{i}",\n'\
               f'    base_path: "/models/HplusModel{i}",\n'\
                '    model_platform: "tensorflow"\n'\
                '  },\n'
        cfg += m_cfg
    cfg += '}'

    # write model config
    with open("/tmp/tfserving/Hplus_models.config" ,"w") as f:
        f.write(cfg)

    # define the shell command to run the server
    cmd = 'docker run '\
          f'--name {server_name} '\
          '-p 8501:8501 '
    for i in range(k):
        cmd += f'--mount type=bind,source=/tmp/tfserving/HplusModel{i},target=/models/HplusModel{i} '
    cmd+= '--mount type=bind,source=/tmp/tfserving/Hplus_models.config,target=/models/models.config '
    cmd+= '-t tensorflow/serving:latest '\
          '--model_config_file=/models/models.config'

    # start the server
    p = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE, shell=True)

    while True:
        line = p.stdout.readline()
        print(line)
        if not line:
            print("Inference server shutdown before successfully initializing")
            return ""
        elif "event loop" in line:
            print("Inference server ready!")
            break

    return server_name

def main(model_dir):
    server_name = init_server(model_dir)
    if not server_name: raise Exception("inference server initialization failed")
    print("initialized model server")
    return

if __name__ == "__main__":
    model_dir = sys.argv[1]
    main(model_dir)