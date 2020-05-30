import cupy as cp
import numpy as np
import pickle
import databatch

def dist(A, B):
    dist = cp.linalg.norm(A - B)
    return 1.0 / (1.0 + dist)

def mdist(A, B):
    ret = cp.var(A - B) / (cp.var(A) + cp.var(B))
    return 0.5 * ret

def cossim(A, B):
    num = float(cp.sum(A * B))
    denom = cp.linalg.norm(A) * cp.linalg.norm(B)
    cos = num / denom
    #return 0.5 + 0.5 * cos
    return cos

if __name__ == "__main__":
    with open("/home/data/ljz/data/comment_new/vedio_vector_svm", "rb") as f:
        vedio_vector = pickle.load(f)
    D = []
    with open("./train", "rb") as f: D = pickle.load(f)
    with open("./test", "rb") as f: D.extend(pickle.load(f))
    ret_d = []; ret_c = []
    for f, L1 in D:
        A = vedio_vector[f]
        A = cp.array(A)
        try:
            B = databatch.get_raw_vector(f)
            B = np.array(databatch.zoom_mean(B, databatch.sz)).mean(0)
            #B = databatch.get_new_vector(f)
            #B = np.array(B).mean(0)
            B = cp.array(B)
            ret_d.append(cp.asnumpy(dist(A, B)))
            ret_c.append(cp.asnumpy(cossim(A, B)))
            if len(ret_d) % 10 == 0: print(len(ret_d))
        except:
            pass

    print(min(ret_d), max(ret_d), np.array(ret_d).mean())
    print(min(ret_c), max(ret_c), np.array(ret_c).mean())

#0.19517913 1.0 0.8179025
#0.98712575 1.0 0.9998783