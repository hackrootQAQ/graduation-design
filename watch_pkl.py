import pickle
import define

def get_msg(p1, p2, wf):
    path = "./data/msg/{}/{}".format(str(p1), str(p2))
    with open(path, "rb") as f:
        vedio_msg = pickle.load(f)
    for v in vedio_msg:
        v.print(file = wf)

if __name__ == "__main__":
    with open("./theMsg.txt", "w", encoding = "utf-8") as f:
        get_msg(1, 24, f)