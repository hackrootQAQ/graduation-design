import os

to_del = [
    (13, '32396504'),
    (4, '2423718'),
    (129, '6896450'),
    (13, '1022200'),
    (119, '835752'),
    (119, '113187'),
    (13, '1022200'),
    (13, '11185914'),
    (4, '25510151'),
    (5, '20100660'),
]

if __name__ == "__main__":
    for x in to_del:
        path = "./data/comment_new/{}/{}".format(str(x[0]), x[1])
        print("Del %s..." % path, end = "")
        try:
            os.remove(path)
            print("Del completed.")
        except:
            print("Failed.")