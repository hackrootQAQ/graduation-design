import define
import requests, re, operator, pickle, time, sys
from bs4 import BeautifulSoup as bts
sys.setrecursionlimit(10000000)

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

def getData(url):
    try:
        time.sleep(1)
        ret = requests.get(url, timeout = 30)
        ret.raise_for_status()
        ret.encoding = ret.apparent_encoding
        return ret.text
    except Exception as e:
        return None
        
def calcTime(t):
    return str(int(t) // 60) + ":" + str(int(t) - int(t) // 60 * 60).zfill(2)

def getCid(aid):
    text = getData("https://www.bilibili.com/video/av{}".format(str(aid)))
    if text == None: return -1
    found = re.findall(r'"cid":[\d]*', text)
    if not found:
        fonud = re.findall(r'"cid"=[\d]*', text)
        if not found: return -1
        return int(found[0].split("=")[1])
    else:
        return int(found[0].split(":")[1])
        
def getComment(aid):
    cid = getCid(aid)
    if cid < 0: return -1
    
    url = "https://comment.bilibili.com/" + str(cid) + ".xml"
    page = getData(url)
    soup = bts(page, "html.parser")
    comment = {}
    
    for c in soup.find_all('d'):
        time = float(c.attrs['p'].split(',')[0])
        comment[time] = c.string
        
    comment = sorted(comment.items(), key = lambda x: x[0])
    ret = [define.Comment(
        _time = x[0], 
        _vtime = calcTime(x[0]), 
        _content = x[1]) for x in comment]
    return ret
       
if __name__ == "__main__":
    for key, value in relational_table.items():
        print("Get vedios' comments under {}...".format(str(key)))
        for tid in value:
            path1 = "./data/msg/{}/{}".format(str(key), str(tid))
            with open(path1, "rb") as f:
                vedio_msg = pickle.load(f)
            cnt = 9000; notFound = 0
            for v in vedio_msg[-1000:]:
                cnt += 1
                out = "\r{}: {}/{} ID = {}".format(
                    str(tid), 
                    str(cnt), 
                    str(len(vedio_msg)), 
                    str(v.aid))
                print(out, end = "", flush = True)

                tmp = getComment(v.aid)
                if tmp == -1: notFound += 1; continue
                p = define.Page(_clist = tmp)
                p.prework()

                path2 = "./data/comment/{}/{}".format(str(key), str(v.aid))
                with open(path2, "wb") as f:
                    pickle.dump(p, f)
            print("\nNot found: {}.".format(str(notFound)))
