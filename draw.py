import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import databatch
import pickle, random, jieba, distance
import define
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def get_num():
    ret = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        for f in fn:
            ret.append(len(databatch.get_new_comment(L1, f)[0]))
        print("{} finished.".format(str(L1)))
    with open("./comment_num", "wb") as f:
        pickle.dump(ret, f)

def draw_1():
    with open("./comment_num", "rb") as f:
        ret = pickle.load(f)
    inp = np.array(ret)
    mean = inp.mean()
    std = inp.std()
    median = np.median(inp)
    print(mean)
    print(std)
    print(median)
    print(inp.max())
    x = np.arange(300, 20000, 50)
    y = normfun(x, mean, std)

    plt.tick_params(labelsize = 12)
    plt.plot(x, color = 'g', linewidth = 3)
    plt.hist(inp, bins = 100, color = 'r', alpha = 0.5, rwidth = 0.9, normed = True)
    font1 = {'family' : 'Fira code', 'weight' : 'bold', 'size' : 14}
    plt.xlabel('Comment count', font1, labelpad = 6)
    plt.ylabel('Probability', font1, labelpad = 6)
    plt.show()

def draw_4():
    with open("./comment_vedio_num", "rb") as f:
        ret = pickle.load(f)
    inp = np.array(ret)
    mean = inp.mean()
    std = inp.std()
    median = np.median(inp)
    print(mean)
    print(std)
    print(median)
    print(inp.max())
    print(inp.min())
    x = np.arange(50, 150000, 50)
    y = normfun(x, mean, std)

    plt.tick_params(labelsize = 12)
    plt.plot(x, y, color = 'g', linewidth = 3)
    plt.hist(inp, bins = 100, color = 'r', alpha = 0.5, rwidth = 0.9, normed = True)
    font1 = {'family' : 'Fira code', 'weight' : 'bold', 'size' : 14}
    plt.xlabel('Charactor count', font1, labelpad = 6)
    plt.ylabel('Probability', font1, labelpad = 6)
    plt.show()

def get_comment_length():
    f_list = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        f_list.extend([(L1, f) for f in fn])
    random.shuffle(f_list)
    print("Start.")
    comment_choose = []
    """
    for i in range(len(f_list)):
        tmpC, tmpL = databatch.get_new_comment(f_list[i][0], f_list[i][1])
        comment_choose.extend([len(tmp.content) for tmp in tmpC])
        if len(comment_choose) > 50000000: break
    with open("./comment_length_char", "wb") as f:
        pickle.dump(comment_choose, f)
    print("Finish.")
    """
    for i in range(len(f_list)):
        tmpC, tmpL = databatch.get_new_comment(f_list[i][0], f_list[i][1])
        comment_choose.extend([len(list(jieba.cut(tmp.content))) for tmp in tmpC])
        if len(comment_choose) > 50000000: break
    with open("./comment_length_word", "wb") as f:
        pickle.dump(comment_choose, f)
    print("Finish.")
    
def get_vedio_length():
    f_list = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        f_list.extend([(L1, f) for f in fn])
    ret = []
    for i in range(len(f_list)):
        tmpC, tmpL = databatch.get_new_comment(f_list[i][0], f_list[i][1])
        num = 0
        for tmp in tmpC: num += len(tmp.content)
        ret.append(num)
        if i % 100 == 0: print(i)
    with open("./comment_vedio_num", "wb") as f:
        pickle.dump(ret, f)

def draw_2():
    with open("./comment_length_word", "rb") as f:
        ret = pickle.load(f)
    mx, mn = max(ret), min(ret)
    x = [i for i in range(mn, mx + 1)]
    print(mx, mn)
    y = [0 for i in range(mn, mx + 1)]
    for l in ret: y[l - mn] += 1
    y = [x / len(ret) for x in y]

    plt.plot(x, y)
    font1 = {'family' : 'Fira code', 'weight' : 'bold'}
    plt.xlabel('Length', font1)
    plt.ylabel('Percentage(%)', font1)
    plt.show()

def get_mean_comment():
    ret = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        for f in fn:
            tmpC, tmpL = databatch.get_new_comment(L1, f)
            num = len(tmpC)
            mean = num / tmpC[-1].time
            ret.append((num, mean))
        print("{} finished.".format(str(L1)))
        print(ret[-1])
    with open("./comment_mean", "wb") as f:
        pickle.dump(ret, f)

def draw_3():
    with open("./comment_mean", "rb") as f:
        ret = pickle.load(f)
    x = [i[0] for i in ret]
    y = [i[1] for i in ret]
    plt.plot(x, y)
    plt.show()

# [(7.944598568288813, 2814115), (8.055715209325419, 5350119), (8.201363982814193, 5024990), (8.45416361755894, 11406343), (9.56378263562433, 299115493)]
def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)

def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer = lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis = 0))
    denominator = np.sum(np.max(vectors, axis = 0))
    cv_ = TfidfVectorizer(tokenizer = lambda s: s.split())
    vectors_ = cv_.fit_transform(corpus).toarray()
    return [1.0 * numerator / denominator, 
        np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1])),
        np.dot(vectors_[0], vectors_[1]) / (norm(vectors_[0]) * norm(vectors_[1]))]

def update(x, d):
    return ((x[0] * x[1] + d) / (x[1] + 1), x[1] + 1)

def get_distance():
    f_list = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        f_list.extend([(L1, f) for f in fn])
    random.shuffle(f_list)
    a = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    b = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    c = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    for i in range(50):
        print("\r{}...".format(str(i + 1)), end = "", flush = True)
        tmpC, tmpL = databatch.get_new_comment(f_list[i][0], f_list[i][1])
        l = len(tmpC)
        if l > 1500: continue
        for i in range(l):
            for j in range(i + 1, l):
                d, d_tf, d_ti = jaccard_similarity(tmpC[i].content, tmpC[j].content)
                dt = abs(tmpC[j].time - tmpC[i].time)
                if dt <= 1: a[0] = update(a[0], d); b[0] = update(b[0], d_tf); c[0] = update(c[0], d_ti)
                elif dt <= 3: a[1] = update(a[1], d); b[1] = update(b[1], d_tf); c[1] = update(c[1], d_ti)
                elif dt <= 5: a[2] = update(a[2], d); b[2] = update(b[2], d_tf); c[2] = update(c[2], d_ti)
                elif dt <= 10: a[3] = update(a[3], d); b[3] = update(b[3], d_tf); c[3] = update(c[3], d_ti)
                else: a[4] = update(a[4], d); b[4] = update(b[4], d_tf); c[4] = update(c[4], d_ti)
    print("\n")
    print(a)
    print(b)
    print(c)

def gen_distance_2():
    f_list = []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
    random.shuffle(f_list)
    a = []; b = []; c = []
    for i in range(50):
        print("\r{}...".format(str(i + 1)), end = "", flush = True)
        l = len(tmpC)
        a1 = []; a2 = []; a3 = []
        for i in range(l):
            for j in range(i + 1, l):
                dt = abs(tmpC[j].time - tmpC[i].time)
                if dt > 5: break
                if dt <= 1: a1.append(edit_distance(tmpC[i].content, tmpC[j].content))
                elif dt <= 3: a2.append(edit_distance(tmpC[i].content, tmpC[j].content))
                elif dt <= 5: a3.append(edit_distance(tmpC[i].content, tmpC[j].content))
        a.append((np.mean(a1), l))
        b.append((np.mean(a2), l))
        c.append((np.mean(a3), l))
    with open("./comment_a", "wb") as f: pickle.dump(a, f)
    with open("./comment_b", "wb") as f: pickle.dump(b, f)
    with open("./comment_c", "wb") as f: pickle.dump(c, f)

if __name__ == "__main__":
    #get_num()
    #draw_1() 
    #get_comment_length()
    #draw_2()
    #get_mean_comment()
    #draw_3()
    #get_distance()
    #s1 = "你在干嘛呢"
    #s2 = "你在干什么呢"
    #print(jaccard_similarity(s1, s2))
    #get_vedio_length()
    #draw_4()
    gen_distance_2()