import databatch
import jieba
import pickle
import math

with open("wl", "rb") as f: W = pickle.load(f)
W_m = {}; i = 0
for x in W: W_m[x[0]] = i; i += 1
sz = 10000
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
for L1, L2 in relational_table.items():
    label_list.append(L1)
for i in range(len(label_list)):
    m[label_list[i]] = i
stop_word = {}
with open("stop", "r") as f:
    for line in f: stop_word[line.strip("\n")] = 1
with open("idf", "rb") as f:
    idf = pickle.load(f)

def tf_idf(L1, f):
    tmpC = databatch.get_new_comment(L1, f)[0]
    ret = [0 for i in range(sz)]
    num = 0
    for C in tmpC:
        sens = jieba.cut(C.content.strip())
        num += len(sens)
        for w in sens:
            if w in stop_word.keys(): continue
            if w not in W_m.keys(): ret[sz - 1] += 1
            else: ret[W_m[w]] += 1
    for i in range(sz): ret[i] = ret[i] / num
    for i in range(sz): ret[i] = ret[i] * idf[i]
    return ret

if __name__ == "__main__":
    """
    word_map = {}; stop_word = {}
    with open("stop", "r") as f:
        for line in f: stop_word[line.strip("\n")] = 1
    i = 0
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        for f in fn:
            tmpC = databatch.get_new_comment(L1, f)[0]
            for C in tmpC:
                sens = jieba.cut(C.content.strip())
                for w in sens:
                    if w not in stop_word.keys(): 
                        if w in word_map: word_map[w] += 1
                        else: word_map[w] = 1
            i += 1
            print(i)
    word_map = list(word_map.items())
    word_map = sorted(word_map, key = lambda x:(-x[1], x[0]))
    with open("./wl", "wb") as f:
        pickle.dump(word_map[:9999], f)    
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        num = 0; i = 0
        h = [-1 for i in range(sz)]
        ret = [0 for i in range(sz)]
        num += len(fn)    
        for f in fn:
            tmpC = databatch.get_new_comment(L1, f)[0]
            for C in tmpC:
                sens = jieba.cut(C.content.strip())
                for w in sens:
                    if w in stop_word.keys(): continue
                    if w not in W_m.keys(): nx = sz - 1
                    else: nx = W_m[w]
                    if h[nx] != i: h[nx] = i; ret[nx] += 1
            i += 1
            print(i)
    print(num)
    for i in range(sz): ret[i] = math.log(num / (ret[i] + 1))
    print(ret)
    with open("./idf", "wb") as f:
        pickle.dump(ret, f)
    """
    for L1, L2 in databatch.relational_table.items():
