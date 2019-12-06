import torch
from sklearn.metrics import f1_score
import warnings
from utils import load_data, EarlyStopping
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import  LogisticRegression
import os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits,embed = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, embed

def my_KNN(x, y, k=5, split_list=[0.2,0.4,0.6,0.8], time=10, show_train=True, shuffle=True):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                #estimator = KNeighborsClassifier(n_neighbors=k)
                estimator = LogisticRegression()
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                time, ss, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))
            print("macro_std",np.std(macro_list),"micro_std",np.std(micro_list))
def my_Kmeans(x, y, k=4, time=10, return_NMI=False):

    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        # print('KMeans exps {}次 æ±~B平å~]~G '.format(time))
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2
def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    y_true=labels.numpy()
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    acm_meta_g1 = [[['pa', 'ap'], ['pf', 'fp']]]
    acm_meta_g2 = [[['pa', 'ap'], ['pt', 'tp']]]
    acm_meta_g3 = [[['pf', 'fp'], ['pt', 'tp']]]
    acm_meta_p1 = [[['pa', 'ap']]]
    acm_meta_p2 = [[['pf', 'fp']]]
    acm_meta_p3 = [[['pt', 'tp']]]
    acm_meta_graphs_1 = [acm_meta_p1, acm_meta_p2]
    ##weighted 1.91.23 91.33 2 91.73 91.72 3 91.83 91.94 4 91.73 91.81
    #unweighted 1.91.55 91.66 2 90.27 90.37 3.90.87 90.97 4 90.73 90.75
    acm_meta_graphs_2 = [acm_meta_g1, acm_meta_p1]  ##91.44 91.37
    acm_meta_graphs_3 = [acm_meta_g1, acm_meta_p2]  ##90.94 90.95
    acm_meta_graphs_4 = [acm_meta_g1, acm_meta_p1, acm_meta_p2]
    #both weighted 1.91.76 91.89 2 92.47 92.56 3 92.72 92.70 4 92.54 92.57 5 91.16 91.38
    #91.97 92.01 7 90.13 90.16 8 91.55 91.68 9 90.91 90.89 10 91.44 91.32
    #both unweighted 91.87 91.86
    #single weighted 1 91.90 92.03 2 91.30 91.28  3 91.87 91.83 4
    acm_meta_graphs_5 = [acm_meta_g1, acm_meta_p1, acm_meta_p2, acm_meta_p3]  ##92.05 92.08
    acm_meta_graphs_6 = [acm_meta_p1, acm_meta_p2, acm_meta_p3]  ##91.37 91.38
    acm_meta_graphs_7 = [acm_meta_p1, acm_meta_p2, acm_meta_p3, acm_meta_g1,
                     acm_meta_g2, acm_meta_g3]  ##91.55,91.55
    acm_meta_graphs_8 = [acm_meta_g3]  ##87.75 87.39

    dblp_meta_p1 = [[['ap', 'pa']]]
    dblp_meta_p2 = [[['ap', 'pc', 'cp', 'pa']]]
    dblp_meta_p3 = [[['ap', 'pt', 'tp', 'pa']]]
    dblp_meta_p4 = [[['ap', 'pa', 'ap', 'pa']]]
    dblp_meta_g1 = [[['ap', 'pa'], ['ap', 'pc', 'cp', 'pa']]]  # co-author and same conference
    dblp_meta_g2 = [[['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]]
    dblp_meta_graph_1 = [dblp_meta_p1, dblp_meta_p2,
                         dblp_meta_p3]  # 92.92 92.35
    ##unweighted 1 92.78 92.26 2 92.92 92.41 3 92.04 91.54 4.92.81 92.33 5 .92.88 92.44
    ## weighted 1 92.74 92.27 2 92.81 92.28 3 93.20 92.69 4 93.55 93.15 5 93.94 93.60
    ##
    dblp_meta_graph_2 = [dblp_meta_p1, dblp_meta_p2,
                         dblp_meta_p3, dblp_meta_g1]  ##92.99 92.44
    dblp_meta_graph_3 = [dblp_meta_p1, dblp_meta_p2,
                         dblp_meta_p3, dblp_meta_g2]  ##93.45 92.95
    ##both
    # 1 94.12 93.71 2 93.84 93.46 3 93.41 92.90 4 93.59 93.10 5 93.13 92.64
    # 6 92.43 91.99 7 93.55 93.14 8 93.80 93.27 9 93.45 93.01 10 93.77 93.38 10
    # 1  93.52 93.13
    ##addition only 1 92.81 92.29 2 92.92 92.42 3 92.36 91.84 4 92.22 91.73
    #93.69 93.23
    dblp_meta_graph_4 = [dblp_meta_p1, dblp_meta_g2]  ##93.34 92.83

    imdb_meta_p1 = [[['md', 'dm']]]
    imdb_meta_p2 = [[['ma', 'am']]]
    imdb_meta_g1 = [[['md', 'dm'], ['ma', 'am']]]
    imdb_meta_graph_1=[imdb_meta_p1,imdb_meta_p2]
    #unweighted 1.59.21 59.18 2 59.99 58.31 3 59.58 59.45
    #weighted 1.60.30 60.25 2 61.19 60.38 3 60.88 60.17
    imdb_meta_graph_2=[imdb_meta_p1,imdb_meta_p2,imdb_meta_g1]
    #unweighted
    #both weighted 1 61.53 61.32 2 60.74 60.68 3 61.09 59.79 4 60.81 60.32
    # 5 61.50 61.73 6 61.67 61.44
    if args['hetero']:
        from model_hetero import HAN
        if args['dataset']=="ACMRaw":
            model = HAN(meta_paths=acm_meta_graphs_4,
                         in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True,
                        weighted=True,
                        embedding_dim=args['embedding_dim']).to(args['device'])
        elif args['dataset']=="DBLP":
            model = HAN(meta_paths=dblp_meta_graph_3,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True ,
                        weighted=True,
                        embedding_dim=args['embedding_dim']).to(args['device'])
        else:
            model = HAN(meta_paths=imdb_meta_graph_2,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True,
                        weighted=True,
                        embedding_dim=args['embedding_dim']).to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])

    name_param=model.named_parameters()
    param_decay=[]
    param_no_decay=[]
    for name_tuple in name_param:
        if "projectjj" not in name_tuple[0] :
            param_decay.append(name_tuple[1])
        else:
            ##pass
            param_no_decay.append(name_tuple[1])
    all_params=[dict(params=param_decay), dict(params=param_no_decay, weight_decay=.0)]
    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=all_params, lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits,embed = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        #print(param_no_decay[0].data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1,embed = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1,embed = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

    my_KNN(embed[test_mask].cpu().numpy(),y_true[test_mask.cpu().numpy().astype(bool)])
    my_Kmeans(embed[test_mask].cpu().numpy(),y_true[test_mask.cpu().numpy().astype(bool)])
if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=2020,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('-d','--dataset', type=str, default="ACM",
                        help="dataset for training ")
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
