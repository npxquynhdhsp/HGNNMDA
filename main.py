# %%
from params import args

from gen_featureQ import gen_feature

from utils.dataprocessing_ko_join import * #Q QT,cmt

from utils.dataprocessing2 import *
from utils.utilQT import *
from utils.utils import *
from sklearn.metrics import log_loss
import numpy as np
import time
import os

# %%
def combine2(mi_em1, dis_em1, mi_em2, dis_em2, idx_pair_train, idx_pair_test, loop_i, ix):
    from sklearn.preprocessing import MinMaxScaler
    def cosSim2(interaction1, interaction2):
        rows, columns = interaction1.shape
        sim_matrix = np.zeros(rows)

        # Calculate cosine similarity for each row
        for i in range(rows):
            vec1 = interaction1[i, :]
            vec2 = interaction2[i, :]
            norm_row = np.linalg.norm(vec1)
            norm_col = np.linalg.norm(vec2)

            if norm_row == 0 or norm_col == 0:
                sim_matrix[i] = 0
            else:
                sim_matrix[i] = np.dot(vec1, vec2) / (norm_row * norm_col)

        return sim_matrix

    scaler = MinMaxScaler()
    mi_em1T = scaler.fit_transform(mi_em1)
    mi_em2T = scaler.fit_transform(mi_em2)
    dis_em1T = scaler.fit_transform(dis_em1)
    dis_em2T = scaler.fit_transform(dis_em2)

    # Combine
    mi_em1 = mi_em1 * cosSim2(mi_em1T, mi_em2T)[:, np.newaxis] + mi_em2 * (1-cosSim2(mi_em1T, mi_em2T)[:, np.newaxis])
    mi_em2 = mi_em2 * cosSim2(mi_em1T, mi_em2T)[:, np.newaxis] + mi_em1 * (1-cosSim2(mi_em1T, mi_em2T)[:, np.newaxis])
    dis_em1 = dis_em1 * cosSim2(dis_em1T, dis_em2T)[:, np.newaxis] + dis_em2 * (1-cosSim2(dis_em1T, dis_em2T)[:, np.newaxis])
    dis_em2 = dis_em2 * cosSim2(dis_em1T, dis_em2T)[:, np.newaxis] + dis_em1 * (1-cosSim2(dis_em1T, dis_em2T)[:, np.newaxis])

    X_train1T = np.hstack((mi_em1[idx_pair_train[:, 0]].tolist(), dis_em1[idx_pair_train[:, 1]].tolist()))
    X_test1 = np.hstack((mi_em1[idx_pair_test[:, 0]].tolist(), dis_em1[idx_pair_test[:, 1]].tolist()))
    X_train2T = np.hstack((mi_em2[idx_pair_train[:, 0]].tolist(), dis_em2[idx_pair_train[:, 1]].tolist()))
    X_test2 = np.hstack((mi_em2[idx_pair_test[:, 0]].tolist(), dis_em2[idx_pair_test[:, 1]].tolist()))

    return X_train1T, X_test1, X_train2T, X_test2

def KNN(X_train_enc, X_test_enc, y_train):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    # Bước 2: Huấn luyện mô hình KNN
    from sklearn.neighbors import KNeighborsClassifier
    k = 27  # Chọn số lượng láng giềng gần nhất (k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Bước 3: Tạo các đặc trưng bổ sung từ KNN (khoảng cách tới các láng giềng gần nhất)
    # Lấy các khoảng cách đến các láng giềng gần nhất của mỗi điểm
    knn_train_dist, knn_train_indices = knn.kneighbors(
        X_train_scaled)  # khoảng cách và chỉ số các láng giềng
    knn_test_dist, knn_test_indices = knn.kneighbors(X_test_scaled)

    # Bước 4: Kết hợp các đặc trưng của KNN vào X_train và X_test
    # Thêm khoảng cách trung bình của các láng giềng gần nhất vào các đặc trưng

    # X_train_enc = np.hstack([X_train_scaled, knn_train_dist.mean(axis=1).reshape(-1, 1)])
    # X_test_enc = np.hstack([X_test_scaled, knn_test_dist.mean(axis=1).reshape(-1, 1)])
    # --
    X_train_enc = np.hstack([X_train_enc, knn_train_dist.mean(axis=1).reshape(-1, 1)])
    X_test_enc = np.hstack([X_test_enc, knn_test_dist.mean(axis=1).reshape(-1, 1)])

    # X_train_enc = (X_train_enc + knn_train_dist.mean(axis=1).reshape(-1, 1))/2
    # X_test_enc = (X_test_enc + knn_test_dist.mean(axis=1).reshape(-1, 1))/2
    return X_train_enc, X_test_enc

# %%
def main():
    ###
    # Q QT
    ###

    if args.db != 'INDE_TEST':
        print(args.db)
        if args.type_eval == 'KFOLD':  # KFOLD/DIS_K/DENO_MI
            set_ix = np.arange(args.bgf, args.nfold + 1) #Q
            temp = 'FOLD '
            print('read_tr_te_adj', args.read_tr_te_adj)
        elif args.type_eval == 'DIS_K':
            # set_ix = args.dis_set
            set_ix = np.genfromtxt(args.fi_A + 'dis_set2.csv').astype(int) #Q
            temp = 'DIS '
        else:
            mi_set = np.genfromtxt(args.fi_A + 'mi_setT.csv').astype(int).T #Q
            set_ix = mi_set
            temp = 'MIRNA '
    else:
        print('INDEPENDENT TEST')
        set_ix = [1]

    prob_set1, prob_set2, prob_set = [], [], []
    true_set = []

    for loop_i in range(args.bgl, args.nloop + 1):
        print('.......................................... LOOP ', loop_i,'.........................................')
        if (args.db != 'INDE_TEST') and (args.type_eval == 'KFOLD'):
            idx_pair_train_set, idx_pair_test_set, y_trainT_set, y_test_set, train_adj_set = \
                split_kfold_MCB(args.fi_A, args.fi_proc, 'adj_MD.csv', \
                             '_MCB', args.type_test, loop_i)
            print('.')
        for ix in set_ix:
            ###-----------------------
            if (args.db == 'INDE_TEST'):
                idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                    split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_4loai.csv', \
                                    '_MCB', args.type_test, -1, ix, loop_i)
            else:
                if (args.type_eval == 'KFOLD'):
                    if args.read_tr_te_adj == 1:
                        idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                            read_train_test_adj(args.fi_proc, '/md_p', \
                                                '_MCB', args.type_test, ix, loop_i)
                    else:
                        idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                            idx_pair_train_set[ix-1], idx_pair_test_set[ix-1], y_trainT_set[ix-1], \
                                y_test_set[ix-1], train_adj_set[ix-1]
                else:
                    idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                        split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_MD.csv', \
                                        '_MCB', args.type_test, -1, ix, loop_i)

            ###-----------------------
            if args.db != 'INDE_TEST':
                print(temp, ix, '*' * 50)

            print('GEN FEATURE MODEL 1')
            mi_em1, dis_em1 = gen_feature(idx_pair_train, idx_pair_test, \
                                train_adj, ix, loop_i, 1)
            print('GEN FEATURE MODEL 2')
            mi_em2, dis_em2 = gen_feature(idx_pair_train, idx_pair_test, \
                                train_adj, ix, loop_i, 2)


            method_set = ['XG']

            print('Combine cosi')
            X_train1T, X_test1, X_train2T, X_test2 = combine2(mi_em1, dis_em1, mi_em2, dis_em2, idx_pair_train, idx_pair_test, loop_i, ix)

            #--NEGATIVE SAMPLES SELECTION
            print('Negative samples selection')
            from xgboost import XGBClassifier
            from sklearn.model_selection import KFold
            def nega_sample_selection(XtrainT, ytrainT):
                kf = KFold(n_splits = 3, shuffle = True, random_state = 2022)
                X_positive = XtrainT[ytrainT == 1]  # Mẫu dương
                y_positive = np.ones(X_positive.shape[0])
                X_unlabeled = XtrainT[ytrainT == 0]  # Mẫu âm
                y_unlabeled = np.zeros(X_unlabeled.shape[0])
                U_splits = list(kf.split(X_unlabeled))

                Upredictions = np.zeros_like(y_unlabeled, dtype=float)

                for i, (test_idx, train_idx) in enumerate(U_splits):  # chu y co doi vi tri train test cho dung y
                    # Train from P + U_i
                    UX_train = np.vstack([X_positive, X_unlabeled[train_idx]])
                    Uy_train = np.hstack([y_positive, y_unlabeled[train_idx]])

                    UX_test = X_unlabeled[test_idx]

                    Umodel = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=args.xg_lrr, n_estimators=args.xg_ne)

                    if i == 1:
                        from sklearn.linear_model import LogisticRegression
                        Umodel = LogisticRegression()
                    elif i == 2:
                        from sklearn.ensemble import RandomForestClassifier
                        Umodel = RandomForestClassifier(n_estimators=args.rf_ne, max_depth=None, n_jobs=-1)  # tam

                    Umodel.fit(UX_train, Uy_train)

                    Uy_prob = Umodel.predict_proba(UX_test)[:, 1]
                    Upredictions[test_idx] += Uy_prob  # Cộng dồn kết quả dự đoán

                Upredictions /= 2

                # All neg
                negative_indices = np.argwhere(np.array(Upredictions) < 0.5).reshape(-1)

                # Select neg
                selected_indices = np.random.choice(negative_indices, size=y_positive.shape[0], replace=False)

                X_final_neg = X_unlabeled[selected_indices]

                Xtrain_AN_2x4344 = np.vstack([X_positive, X_final_neg])
                ytrain_AN_2x4344 = np.hstack(
                    [np.ones(y_positive.shape[0]), np.zeros(y_positive.shape[0])])
                return Xtrain_AN_2x4344, ytrain_AN_2x4344, selected_indices
            #-----------------------------------------------------

            true_set.append(y_test) # QT
            # #--

            #--EVALUATION
            print('EVALUATION:')
            ###-1 trong 2 ---------------------------------------------------------------------------------

            ###----------------------------------------------------------------------------------
            #--(2) Reliable negative selection
            print('RELIABLE NEGATIVE SAMPLE SELECTION')
            print('Model 1')
            X_train1, y_train, selected_indices1 = nega_sample_selection(X_train1T, y_trainT)
            # Q QX
            print('Model 2')
            X_train2, y_train, selected_indices2 = nega_sample_selection(X_train2T, y_trainT)
            # Q QX

            print('Reli CO KNN')
            print('Model 1')
            X_train1, X_test1 = KNN(X_train1, X_test1, y_train)
            y_prob1 = models_eval(method_set[0], X_train1, X_test1, y_train, y_test, ix, loop_i, 1)
            print('Model 2')
            X_train2, X_test2 = KNN(X_train2, X_test2, y_train)
            y_prob2 = models_eval(method_set[0], X_train2, X_test2, y_train, y_test, ix, loop_i, 2)

            ##--------------
            # code nay da cap nhat
            def learn_alpha(y_prob1, y_prob2, y_true, lr=0.01, epochs=500):
                """
                Học alpha để kết hợp y_prob1 và y_prob2 sao cho khớp tốt nhất với y_true.
                y_prob1, y_prob2, y_true: numpy array hoặc torch tensor shape (n_samples,)
                """
                # Chuyển về tensor float
                if not torch.is_tensor(y_prob1):
                    y_prob1 = torch.tensor(y_prob1, dtype=torch.float32)
                if not torch.is_tensor(y_prob2):
                    y_prob2 = torch.tensor(y_prob2, dtype=torch.float32)
                if not torch.is_tensor(y_true):
                    y_true = torch.tensor(y_true, dtype=torch.float32)

                # Khởi tạo alpha (raw_alpha) và dùng sigmoid để giữ alpha trong [0,1]
                raw_alpha = torch.nn.Parameter(torch.tensor(0.5))  # bắt đầu từ 0.5, thu khac cung ok the
                optimizer = torch.optim.Adam([raw_alpha], lr=lr)
                criterion = torch.nn.MSELoss()

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    alpha = torch.sigmoid(raw_alpha)  # ép alpha vào [0,1]
                    y_pred = alpha * y_prob1 + (1 - alpha) * y_prob2
                    loss = criterion(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                # Trả về alpha cuối cùng
                return torch.sigmoid(raw_alpha).item()

            alpha = learn_alpha(y_prob1, y_prob2, y_test, lr=0.01, epochs=500)
            print(alpha)
            y_prob = alpha * y_prob1 + (1 - alpha) * y_prob2
            # --------

            calculate_score([y_test], [y_prob])

            prob_set1.append(y_prob1)
            prob_set2.append(y_prob2)
            prob_set.append(y_prob)

            #--save
            np.savetxt(
                args.fi_out + 'L' + str(loop_i) + '_yprob_' + method_set[0].lower() + str(
                    ix) + '.csv', y_prob)
            np.savetxt(
                args.fi_out + 'L' + str(loop_i) + '_ytrue' + str(
                    ix) + '.csv', y_test, fmt='%d')

    print('--------------------------------FINAL ALL:-------------------------------')
    savekq(method_set[0], true_set, prob_set) #Q QT doc-utl

# %%
if __name__ == "__main__":
    print('fi_ori_feature', args.fi_ori_feature)
    main()



