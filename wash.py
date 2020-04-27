import define, pickle, databatch

def handle(s):
    return s.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')

if __name__ == "__main__":
    for L1, L2 in databatch.relational_table.items():
        fn = databatch.get_filename(L1)
        for f in fn:
            comments, label = databatch.get_comment(L1, f)
            if len(comments) < 386: continue
            new_comment = []
            for comment in comments:
                tmp = handle(comment.content)
                if len(tmp) < 2: continue
                new_comment.append(
                    define.Comment(comment.time, comment.vtime, tmp)
                )
            page_new = define.PageLite(_clist = new_comment, _label = label)
            path = "./data/comment_new/{}/{}".format(str(L1), str(f))
            with open(path, "wb") as fl:
                    pickle.dump(page_new, fl)