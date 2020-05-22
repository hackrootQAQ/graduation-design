import cupy as cp
import numpy as np
import pickle
import databatch

def dist(A, B):
    dist = cp.linalg.norm(A - B)
    return 1.0 / (1.0 + dist)

def cossim(A, B):
    num = float(cp.sum(A * B))
    denom = cp.linalg.norm(A) * cp.linalg.norm(B)
    cos = num / denom
    return 0.5 + 0.5 * cos

if __name__ == "__main__":
    with open("/home/data/ljz/data/comment_new/vedio_vector_svm", "rb") as f:
        vedio_vector = pickle.load(f)
    D = []
    with open("./train", "rb") as f: D = pickle.load(f)
    with open("./test", "rb") as f: D.extend(pickle.load(f))
    print(len(D))
    ret_d = []; ret_c = []
    """
    for f, L1 in D:
        try:
            A = vedio_vector[f]
            B = databatch.get_raw_vector(f)
            B = np.array(B).mean(0)
            ret_d.append(dist(A, B))
            ret_c.append(cossim(A, B))
        except:
            pass
    """
    f = "82008182"
    A = vedio_vector[f]
    print(A)
    A = cp.array(A)
    B = databatch.get_raw_vector(f)
    B = np.array(B).mean(0)
    print(B)
    B = cp.array(A)
    ret_d.append(cp.asnumpy(dist(A, B)))
    ret_c.append(cp.asnumpy(cossim(A, B)))

    print(min(ret_d), max(ret_d), np.array(ret_d).mean())
    print(min(ret_c), max(ret_c), np.array(ret_c).mean())