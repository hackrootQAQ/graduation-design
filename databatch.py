import tensorflow as tf
from bert_serving.client import BertClient
import pickle, os, random
import define
import numpy as np

## reference: https://github.com/uupers/BiliSpider/wiki/%E8%A7%86%E9%A2%91%E5%88%86%E5%8C%BA%E5%AF%B9%E5%BA%94%E8%A1%A8
relational_table = {
    1   : [24, 25, 47, 27],
    13  : [33, 32, 51, 152, 153, 168, 169, 170, 195],
    3   : [28, 31, 30, 194, 59, 193, 29, 130],
    129 : [20, 154, 156],
    4   : [17, 171, 172, 65, 173, 121, 136, 19],
    36  : [124, 122, 39, 96, 98, 176],
    188 : [95, 189, 190, 191],
    160 : [138, 21, 76, 75, 161, 162, 163, 174],
    119 : [22, 26, 126, 127],
    155 : [157, 158, 164, 159, 192],
    5   : [71, 137, 131],
    181 : [182, 183, 85, 184, 86],
}
label_list = []; m = {}
"""
for L1, L2 in relational_table.items():
    label_list.extend(L2)
for i in range(len(label_list)):
    m[label_list[i]] = i
"""
for L1, L2 in relational_table.items():
    label_list.append(L1)
for i in range(len(label_list)):
    m[label_list[i]] = i

def get_comment(p1, p2):
    path = "/home/data/ljz/data/comment/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        comment = pickle.load(f)
    return (comment.clist, len(comment.clist))

def get_filename(p1):
    path = "/home/data/ljz/data/comment/{}".format(str(p1))
    for r, d, f in os.walk(path):
        return f

def get_new_comment(p1, p2):
    path = "/home/data/ljz/data/comment_new/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        comment = pickle.load(f)
    return (comment.clist, len(comment.clist))

def get_new_filename(p1):
    path = "/home/data/ljz/data/comment_new/{}".format(str(p1))
    if not os.path.exists(path): path = "./data/comment_new/{}".format(str(p1))
    for r, d, f in os.walk(path):
        return f

def get_mince_comment(p1, p2):
    path = "/home/data/ljz/data/comment_mince/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        comment = pickle.load(f)
    return (comment.clist, len(comment.clist))

def get_mince_filename(p1):
    path = "/home/data/ljz/data/comment_mince/{}".format(str(p1))
    for r, d, f in os.walk(path):
        return f

def get_raw_vector(p):
    path = "/home/data/ljz/data/comment_new/vedio_vector0/{}".format(str(p))
    with open(path, "rb") as f:
        return pickle.load(f)

def get_new_vector(p):
    path = "/home/data/ljz/data/comment_new/vedio_vector1/{}".format(str(p))
    with open(path, "rb") as f:
        return pickle.load(f)

def expansion(urtext, sz):
    length = len(urtext); times = sz // length
    urtext_ = []
    for ut in urtext:
        urtext_.extend([ut for i in range(times)])
    if len(urtext_) == sz: return urtext_
    length = len(urtext_); extra = sz - length; interval = (length + extra - 1) // extra
    ret = []
    for i in range(length):
        ret.append(urtext_[i])
        if (i + 1) % interval == 0: ret.append(urtext_[i])
    while len(ret) < sz: ret.append(ret[-1])
    assert len(ret) == sz
    return ret    

def lessen(urtext, sz):
    length = len(urtext); interval = length // sz
    ret = []
    if interval >= 2:
        a = interval * sz + sz - length; b = sz - a
        now = 0
        for i in range(length):
            now += 1
            if a > 0:
                if now == interval: 
                    now = 0; a -= 1; ret.append(urtext[i])
            else:
                if now == interval + 1: 
                    now = 0; b -= 1; ret.append(urtext[i])
    else:
        interval = length // (length - sz)
        a = interval * (length - sz) - sz; b = length - sz - a
        now = 0
        for i in range(length):
            now += 1
            if a > 0:
                if now == interval: now = 0; a -= 1
                else: ret.append(urtext[i])
            else:
                if now == interval + 1: now = 0; b -= 1
                else: ret.append(urtext[i])
    assert len(ret) == sz
    return ret

def zoom(urtext, sz):
    if len(urtext) == sz: return urtext
    if len(urtext) < sz: return expansion(urtext, sz)
    else: return lessen(urtext, sz)

def cut_two_parts(test_size):
    D = []
    for L1, L2 in relational_table.items():
        fn = get_new_filename(L1)
        D.extend([(L1, f) for f in fn])
    random.shuffle(D)
    if test_size == 0: return D, []
    return D[:-test_size], D[-test_size:]    

"""
def cut_two_parts(test_size):
    D = []
    for L1, L2 in relational_table.items():
        for L in L2:
            fn = get_mince_filename(L)
            D.extend([(L, f) for f in fn])
    random.shuffle(D)
    if test_size == 0: return D, []
    return D[:-test_size], D[-test_size:] 
"""

sz = 768
"""
def get_batch(batch_size, D = None):
    now = 0
    while True:
        ret = []; fr = []
        for i in range(batch_size):
            L1, f = D[now]; fr.append(D[now])
            tmpC, tmpL = get_new_comment(L1, f)
            ret.append((zoom(tmpC, sz), m[L1]))
            now = (now + 1) % len(D)
            if now == 0: random.shuffle(D)
        yield ret, fr
"""

def get_batch(batch_size, D = None):
    now = 0
    while True:
        if now == 0: random.shuffle(D)
        _X = []; _Y = []; fr = []
        for i in range(batch_size):
            f, L1 = D[now]; fr.append(D[now])
            _X.append(get_new_vector(f))
            _Y.append(m[L1])
            now = (now + 1) % len(D)
        yield _X, _Y, fr

def expansion_mean(urvec, sz):
    length = urvec.shape[0]; times = sz // length
    urvec_ = []
    for uv in urvec:
        urvec_.extend([uv for i in range(times)])
    if len(urvec_) == sz: return urvec_
    length = len(urvec_); extra = sz - length
    interval = (length + extra - 1) // extra
    ret = []
    for i in range(length):
        ret.append(urvec_[i])
        if (i + 1) % interval == 0: ret.append(urvec_[i])
    while len(ret) < sz: ret.append(ret[-1])
    assert len(ret) == sz
    return ret

def lessen_mean(urvec, sz):
    length = urvec.shape[0]; interval = length // sz
    ret = []; tmp = []
    a = interval * sz + sz - length; b = sz - a
    now = 0
    for i in range(length):
        now += 1
        if a > 0:
            if now == interval: 
                now = 0; a -= 1 
                assert len(tmp) == interval
                ret.append(np.array(tmp).mean())
                print(ret[-1])
                tmp = []
            else: tmp.append(urvec[i])
        else:
            if now == interval + 1: 
                now = 0; b -= 1 
                assert len(tmp) == interval + 1
                ret.append(np.array(tmp).mean())
                tmp = []
            else: tmp.append(urvec[i])
    assert len(ret) == sz
    return ret

def zoom_mean(urvec, sz):
    if urvec.shape[0] == sz: return urvec
    if urvec.shape[0] < sz: return expansion_mean(urvec, sz)
    else: return lessen_mean(urvec, sz)

def get_mean_batch(batch_size, D = None):
    now = 0
    while True:
        if now == 0: random.shuffle(D)
        _X = []; _Y = []; fr = []
        for i in range(batch_size):
            f, L1 = D[now]; fr.append(D[now])
            _X.append(zoom_mean(get_raw_vector(f), sz))
            _Y.append(m[L1])
            now = (now + 1) % len(D)
        yield _X, _Y, fr

if __name__ == "__main__":
    pass