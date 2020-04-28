import tensorflow as tf
from bert_serving.client import BertClient
import pickle, os, random
import define

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
m = {
    1   : 0,
    13  : 1,
    3   : 2,
    129 : 3,
    4   : 4,
    36  : 5,
    188 : 6,
    160 : 7,
    119 : 8,
    155 : 9,
    5   : 10,
    181 : 11,
}

def get_comment(p1, p2):
    path = "./data/comment_new/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        comment = pickle.load(f)
    #print(comment.clist[0].content)
    return (comment.clist, len(comment.clist))

def get_filename(p1):
    path = "./data/comment_new/{}".format(str(p1))
    for r, d, f in os.walk(path):
        return f

def expansion(urtext, sz):
    length = len(urtext); times = sz // length
    urtext_ = []
    for ut in urtext:
        urtext_.extend([ut for i in range(times)])
    if len(urtext_) == sz: return urtext
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
        fn = get_filename(L1)
        D.extend([(L1, f) for f in fn])
    random.shuffle(D)
    return D[:-test_size], D[-test_size:]    

sz = 768
def get_batch(batch_size, D = None):
    P = [i for i in range(len(D))]
    random.shuffle(P)
    now = 0
    while True:
        ret = []; fr = []
        for i in range(batch_size):
            L1, f = D[P[now]]; fr.append(D)
            tmpC, tmpL = get_comment(L1, f)
            ret.append((zoom(tmpC, sz), m[L1]))
            now = now + 1
            if now >= len(D): now = 0
        yield ret, fr

def print_sz_0():
    len_list = []
    for L1, L2 in relational_table.items():
        fn = get_filename(L1)
        for f in fn:
            path = "./data/comment/{}/{}".format(str(L1), str(f))
            with open(path, "rb") as fl:
                comment = pickle.load(fl)
            len_list.append(comment.tlen)
    len_list = sorted(len_list)
    with open("theMsg.txt", "w") as f:
        for l in len_list: f.write(str(l) + "\n")

if __name__ == "__main__":
    print_sz_0()