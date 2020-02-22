import heapq, re
INF = 1000000000
timeSlice = 3

def stringSimilar(s1, s2):
    l1 = len(s1); l2 = len(s2); p1 = 0; p2 = 0
    while p1 < l1 and p2 < l2:
        while p1 + 1 < l1 and s1[p1] == s1[p1 + 1]: p1 += 1
        while p2 + 1 < l2 and s2[p2] == s2[p2 + 1]: p2 += 1
        if s1[p1] != s2[p2]: return False
        p1 += 1; p2 += 1
    if p1 == l1 and p2 == l2: return True
    return False

def printAllMember(func):
    def output(self, file):
        for name, value in vars(self).items():
            print("{}: {}".format(name, value), file = file)
        print("\n", file = file)
        return func(self, file)
    return output

class Comment(object):
    def __init__(self, _time, _vtime, _content):
        self.time = _time
        self.vtime = _vtime
        self.content = _content
        
    def __lt__(self, other):
        return self.time < other.time
        
    def prework(self):
        if self.content[0] == "[" and self.content[-1] == "]":
            self.content = self.content.replace("true", "True")
            self.content = self.content.replace("false", "False")
            self.content = self.content.replace("null", "None")
            self.content = self.content.replace("/n", " ")
            try:
                s = eval(self.content)
                if len(s) == 13: self.content = s[5]
                return True
            except Exception as e:
                self.content = self.content.replace("[", "")
                self.content = self.content.replace("]", "")
                try:
                    s = eval(self.content)
                    return True
                except Exception as e:
                    print(self.content)
                    return False
        return True

class tightComment(object):
    def __init__(self, _st = INF, _ed = -INF, _num = 0, _content = None):
        self.st = _st
        self.ed = _ed
        self.num = _num      
        self.content = _content

    def __lt__(self, other):
        return self.ed < other.ed
        
    def update(self, c):
        self.num += 1
        self.st = min(self.st, c.time)
        self.ed = max(self.ed, c.time)
        return self

class myHeap(object):
    def __init__(self, _k = 10000):
        self.k = _k
        self.h = []
        
    def size(self):
        return len(self.h)

    def isEmpty(self):
        if self.size() > 0: return False
        else: return True

    def up(self, p):
        if p >= self.size(): return
        f = (p - 1) // 2
        while f >= 0:
            if self.h[p] < self.h[f]: 
                #print(p, f)
                self.h[p], self.h[f] = self.h[f], self.h[p]
            else: return
            p = f; f = (p - 1) // 2

    def down(self, p):
        ls, rs = p + p + 1, p + p + 2
        while ls < self.size():
            if rs < self.size() and self.h[rs] < self.h[ls]: nx = rs
            else: nx = ls
            if self.h[p] < self.h[nx]: break
            self.h[p], self.h[nx] = self.h[nx], self.h[p]
            p = nx; ls = p + p + 1; rs = p + p + 2

    def pop(self):
        if not self.isEmpty(): 
            tmp = self.h[0]; self.h[0] = self.h[-1]
            self.h.pop()
            self.down(0)
            return tmp
        else: return None

    def push(self, item):
        self.h.append(item)
        self.up(self.size() - 1)
        if self.size() > self.k: self.pop() 
    
    def top(self):
        if not self.isEmpty(): return self.h[0]
        else: return None
    
    def delete(self, pos):
        if self.size() == 1:
            self.h.pop(); return
        self.h[pos] = self.h[-1]
        self.h.pop()
        self.up(pos); self.down(pos)
        
    def update(self, c):
        if not isinstance(c, Comment): return
        ret = []
        while self.top() != None and c.time - self.top().ed > timeSlice:
            ret.append(self.pop())
        for i in range(len(self.h)):
            if stringSimilar(self.h[i].content, c.content):
                tmp = self.h[i].update(c)
                self.delete(i)
                self.push(tmp)
                return ret
        self.push(tightComment(c.time, c.time, 1, c.content))
        return ret

class Vedio(object):
    def __init__(self, _aid, _tid, _title, _mid, _view, _danmaku, _reply, 
        _favorite, _coin, _share, _like):
        self.aid = _aid
        self.tid = _tid
        self.title = _title
        self.mid = _mid
        self.view = _view
        self.danmaku = _danmaku
        self.reply = _reply
        self.favorite = _favorite
        self.coin = _coin
        self.share = _share
        self.like = _like
        
    def __lt__(self, other):
        return self.danmaku < other.danmaku
        
    @printAllMember
    def print(self, file):
        pass
       
class Page(object):
    def __init__(self, _clist, _label = None):
        self.label = _label
        self.clist = _clist
        self.clen = len(_clist)
        self.tlist = []
        self.tlen = 0
        
    def prework(self):
        h = myHeap(_k = INF)
        for i in range(self.clen):
            c = self.clist[i]
            if not c.prework(): continue
            self.tlist.extend(h.update(c))
        while not h.isEmpty():
            self.tlist.append(h.pop())
        self.tlen = len(self.tlist)

    @printAllMember
    def print(self, file):
        pass
        
if __name__ == "__main__":
    pass
