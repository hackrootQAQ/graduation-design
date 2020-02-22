import define
import json, time, heapq, pickle
import requests

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

def gen_vedio_url(rid, pn, ps):
    """ Generate the web-interface URL. """
    return "http://api.bilibili.com/x/web-interface/newlist?rid={}&pn={}&ps={}".format(
        str(rid),
        str(pn),
        str(ps))

def fetch_vedio_amount(L2):
    """ Get the vedio amount under L2. """
    url = gen_vedio_url(L2, pn = 1, ps = 1)
    r = requests.get(url = url, timeout = 30).json()
    return int(r["data"]["page"]["count"])

def fetch_vedio(L1, L2):
    """ Get the first 10000 most popular vedios. """
    n = fetch_vedio_amount(L2)
    print("Get {} vedios under ({}, {})...".format(str(n), L1, L2))
    tot = n
    p = 1
    h = define.myHeap(_k = 10000) 
    while n > 0:
        url = gen_vedio_url(L2, pn = p, ps = min(n, 50))
        n, p = n - 50, p + 1

        pro = min(100, (tot - n) * 100 / tot)
        out = "\r{}%: {}".format(format(pro, ".2f"), "=" * int(pro) + ">")
        print(out, end = "", flush = True)

        try:
            r = requests.get(url = url, timeout = 30).json()
            for v in r["data"]["archives"]:
                vedio = define.Vedio(
                    _aid = v["aid"],
                    _tid = (L1, L2),
                    _title = v["title"],
                    _mid = (v["owner"]["mid"], v["owner"]["name"]),
                    _view = v["stat"]["view"],
                    _danmaku = v["stat"]["danmaku"],
                    _reply = v["stat"]["reply"],
                    _favorite = v["stat"]["favorite"],
                    _coin = v["stat"]["coin"],
                    _share = v["stat"]["share"],
                    _like = v["stat"]["like"]
                )
                h.push(vedio)
        except Exception as e:
            pass
        continue
        
    print("\n")
    ret = []
    while h.isEmpty() == False:
        ret.append(h.pop())
    return ret     
    
if __name__ == "__main__":
    for key, value in relational_table.items():
        for tid in value:
            videolist = fetch_vedio(key, tid)
            path = "./data/msg/{}/{}".format(str(key), str(tid))
            with open(path, "wb") as f:
                pickle.dump(videolist, f)