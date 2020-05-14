import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from bert_serving.client import BertClient
import numpy as np
import databatch
import config

CFG = config.CONFIG_S()
bc = BertClient(check_length = False)

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
label_list = []; m = {}
for L1, L2 in relational_table.items():
    label_list.append(L1)
for i in range(len(label_list)):
    m[label_list[i]] = i

def svm_c(x_train, x_test, y_train, y_test):
    svc = SVC(kernel = "rbf", class_weight = "balanced")
    c_range = np.logspace(-5, 15, 11, base = 2)
    gamma_range = np.logspace(-9, 3, 13, base = 2)
    param_grid = [{"kernel" : ["rbf"], "C" : c_range, "gamma" : gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv = 3, n_jobs = -1)
    clf = grid.fit(x_train, y_train)
    score = grid.score(x_test, y_test)
    print("acc %s" % score)

if __name__ == "__main__":
    with open("./data/comment_new/vedio_vector_svm", "rb") as f:
        vedio_vector = pickle.load(f)
    train_D, test_D = databatch.cut_two_parts(CFG.test_size)
    train_X = []; train_Y = []
    for L1, f in train_D:
        train_X.append(vedio_vector[f])
        train_Y.append(m[L1])
    test_X = []; test_Y = []
    for L1, f in test_D:
        test_X.append(vedio_vector[f])
        test_Y.append(m[L1])
    svm_c(train_X, test_X, train_Y, test_Y)