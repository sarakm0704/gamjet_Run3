# Base script from gitlab.cern.ch/Hplus/HiggsAnalysis keras neural networks Disco_tf.py
# from builtins import breakpoint
import tensorflow as tf

def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """


    def matricise(var):
        xx = tf.reshape(tf.tile(tf.reshape(var, [-1, 1]), [1,tf.size(var)]), [tf.size(var), tf.size(var)])
        yy = tf.reshape(tf.tile(var, [tf.size(var)]), [tf.size(var),tf.size(var)])
        return xx, yy

    def diff_2(x,y):
        return x - y

    def diff_3(x,y,z):
        return x-y-z

    def diffs_from_mean(var):
        mat = tf.math.abs(diff_2(*matricise(var)))
        matavg = tf.reduce_mean(mat*normedweight, axis=1)
        meandiff = diff_3(mat, *matricise(matavg)) + tf.reduce_mean(matavg*normedweight)
        return meandiff

    Amat = diffs_from_mean(var_1)
    Bmat = diffs_from_mean(var_2)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)

    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
    return dCorr

