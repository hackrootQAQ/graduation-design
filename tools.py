import define, pickle, databatch, os

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
    with open("wrong_data.txt", "r") as fl:
        for line in fl:
            L1, f = eval(line)[0]
            print("Delete file ({}, {})...".format(str(L1), str(f)), end = "")
            try:
                os.remove("./data/comment_new/{}/{}".format(str(L1), str(f)))
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
            with open("wrong_data.txt", "a+") as fl:
                fl.write("[({}, \"{}\")]\n".format(str(L1), str(f)))
            wr += 1
        print("Check {}, wrong {}.".format(str(num), str(wr)))

if __name__ == "__main__":
    #read_new_data(119, 7439521, _out = "out.txt")
    del_data()
    #find_zero()