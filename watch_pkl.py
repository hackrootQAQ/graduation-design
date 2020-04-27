import pickle
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

def get_msg(p1, p2, wf):
    path = "./data/msg/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        vedio_msg = pickle.load(f)
    cnt = 0
    for vedio in vedio_msg:
        cnt = max(cnt, vedio.danmaku)
    return cnt

if __name__ == "__main__":
    cnt = 0
    for L1, L2 in relational_table.items():
        for L in L2:
            with open("./theMsg.txt", "w", encoding = "utf-8") as f:
                cnt = max(get_msg(L1, L, f), cnt)
            print(cnt)