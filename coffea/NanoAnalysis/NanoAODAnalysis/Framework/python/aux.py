import subprocess

def execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    f = p.stdout
    ret=[]
    for line in f:
        line = line.decode('utf-8')
        #sys.stdout.write(line)                                                                                                                        
        ret.append(line.replace("\n", ""))
    f.close()
    return ret
