import os



if __name__ == "__main__":
    for x in to_del:
        path = "./data/comment_new/{}/{}".format(str(x[0]), x[1])
        print("Del %s..." % path, end = "")
        try:
            os.remove(path)
            print("Del completed.")
        except:
            print("Failed.")