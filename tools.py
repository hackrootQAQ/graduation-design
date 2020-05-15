import define, pickle, databatch, os, shutil
import tensorflow as tf
from bert_serving.client import BertClient
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

def handle(s):
    return s.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')

def wash():
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_filename(L1)
        for f in fn:
            comments, label = databatch.get_comment(L1, f)
            if len(comments) < 386: continue
            new_comment = []
            for comment in comments:
                tmp = handle(comment.content)
                if len(tmp) < 2: continue
                if len(tmp) > 20: continue
                new_comment.append(
                    define.Comment(comment.time, comment.vtime, tmp)
                )
            page_new = define.PageLite(_clist = new_comment, _label = label)
            path = "./data/comment_new/{}/{}".format(str(L1), str(f))
            with open(path, "wb") as fl:
                    pickle.dump(page_new, fl)

def read_new_data(L1, f, _out):
    comments, label = databatch.get_new_comment(L1, f)
    with open(_out, "w", encoding = "utf-8") as fl:
        for comment in comments:
            fl.write("{} {} {}\n".format(str(comment.time), str(comment.vtime), str(comment.content)))

def read_data(L1, f, _out):
    comments, label = databatch.get_comment(L1, f)
    with open(_out, "w", encoding = "utf-8") as fl:
        for comment in comments:
            fl.write("{} {} {}\n".format(str(comment.time), str(comment.vtime), str(comment.content)))

def del_data():
    with open("wrong_data", "r") as fl:
        for line in fl:
            L1, f = eval(line)[0]
            print("Delete file ({}, {})...".format(str(L1), str(f)), end = "")
            try:
                os.remove("/home/data/ljz/data/comment_new/{}/{}".format(str(L1), str(f)))
                print("Completed!")
            except:
                print("Failed!")

def find_zero():
    D1, D2 = databatch.cut_two_parts(0)
    num = 0; wr = 0
    for L1, f in D1:
        tmpC, tmpL = databatch.get_new_comment(L1, f)
        num += 1
        if len(tmpC) == 0: 
            with open("wrong_data", "a+") as fl:
                fl.write("[({}, \"{}\")]\n".format(str(L1), str(f)))
            wr += 1
        print("Check {}, wrong {}.".format(str(num), str(wr)))

def mince():
    for L1, L2 in relational_table.items():
        for L in L2:
            try:
                os.mkdir("/home/data/ljz/data/comment_mince/{}".format(str(L)))
            except:
                pass
    m = {}
    for key, value in relational_table.items():
        for tid in value:
            path1 = "/home/data/ljz/data/msg/{}/{}".format(str(key), str(tid))
            with open(path1, "rb") as f:
                vedio_msg = pickle.load(f)
            for v in vedio_msg:
                m[v.aid] = tid
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        for f in fn:
            srcfile = "/home/data/ljz/data/comment_new/{}/{}".format(str(L1), str(f))
            if not os.path.exists(srcfile): continue
            dstfile = "/home/data/ljz/data/comment_mince/{}/".format(str(m[int(f)]))
            shutil.copy(srcfile, dstfile)

def gen_vedio_vector():
    bc = BertClient(check_length = False)
    _vedio, _ = databatch.cut_two_parts(0)
    m = {}; num = 0
    """
    for L1, f in _vedio:
        tmpC, tmpL = databatch.get_new_comment(L1, f)
        sens = [comment.content for comment in tmpC]
        bx = np.array(bc.encode(sens))
        m[f] = bx.mean(0)
        num += 1
        print("{}/{}".format(str(num), str(len(_vedio))))
    """
    for L1, f in _vedio:
        tmpC, tmpL = databatch.get_new_comment(L1, f)
        #tmpC = databatch.zoom(tmpC, databatch.sz)
        sens = [comment.content for comment in tmpC]
        vec = bc.encode(sens)
        num += 1
        print("{}/{}".format(str(num), str(len(_vedio))))
        with open("/home/data/ljz/data/comment_new/vedio_vector0/{}".format(str(f)), "wb") as fl:
            pickle.dump(vec, fl)

def divide_train_test(scale = 0.08):
    _train, _test = [], []
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_new_filename(L1)
        num = int(len(fn) * scale)
        _train.extend([(f, L1) for f in fn[:-num]])
        _test.extend([(f, L1) for f in fn[-num:]])
    print(len(_train), len(_test))
    with open("./test", "wb") as fl: pickle.dump(_test, fl)
    with open("./train", "wb") as fl: pickle.dump(_train, fl)

if __name__ == "__main__":
    #read_new_data(119, 7439521, _out = "out.txt")
    #del_data()
    #find_zero()
    #mince()
    #gen_vedio_vector()
    divide_train_test()
    """
    with open("./data/comment_new/vedio_vector", "rb") as f:
        comment = pickle.load(f)
    print(comment)
    """