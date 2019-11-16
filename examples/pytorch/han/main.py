import torch
from sklearn.metrics import f1_score
import warnings
from utils import load_data, EarlyStopping
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


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
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    acm_meta_g1 = [[['pa', 'ap'], ['pf', 'fp']]]
    acm_meta_g2 = [[['pa', 'ap'], ['pt', 'tp']]]
    acm_meta_g3 = [[['pf', 'fp'], ['pf', 'fp']]]
    acm_meta_p1 = [[['pa', 'ap']]]
    acm_meta_p2 = [[['pf', 'fp']]]
    acm_meta_p3 = [[['pt', 'tp']]]
    acm_meta_graphs_1 = [acm_meta_p1, acm_meta_p2]  ##91.58 91.58
    acm_meta_graphs_2 = [acm_meta_g1, acm_meta_p1]  ##91.44 91.37
    acm_meta_graphs_3 = [acm_meta_g1, acm_meta_p2]  ##90.94 90.95
    acm_meta_graphs_4 = [acm_meta_g1, acm_meta_p1, acm_meta_p2]  ##92.26 92.26 #both 1. 92.29 92.31
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
    ##unweighted 1 92.78 92.26 2 92.92 92.41 3 92.04 91.54 4.92.81 92.33
    ## weighted 1 92.74 92.27 2 92.81 92.28 3 93.20 92.69 4 93.55 93.15 5 93.94 93.60
    dblp_meta_graph_2 = [dblp_meta_p1, dblp_meta_p2,
                         dblp_meta_p3, dblp_meta_g1]  ##92.99 92.44
    dblp_meta_graph_3 = [dblp_meta_p1, dblp_meta_p2,
                         dblp_meta_p3, dblp_meta_g2]  ##93.45 92.95
    ##both
    # 1 94.12 93.71 2 93.84 93.46 3 93.41 92.90 4 93.59 93.10 5 93.13 92.64
    # 6 92.43 91.99 7 93.55 93.14 8 93.80 93.27
    ##addition only 1 92.81 92.29 2 92.92 92.42 3 92.36 91.84 4 92.22 91.73
    #
    dblp_meta_graph_4 = [dblp_meta_p1, dblp_meta_g2]  ##93.34 92.83

    imdb_meta_p1 = [[['md', 'dm']]]
    imdb_meta_p2 = [[['ma', 'am']]]
    imdb_meta_g1 = [[['md', 'dm'], ['ma', 'am']]]
    imdb_meta_graph_1=[imdb_meta_p1,imdb_meta_p2]
    #unweighted 1.59.92 59.87 2 59.62 58.96 3 60.20 59.91 4 59.99 59.84 5 62.28 61.86
    #weighted 1.61.60 61.42 2 61.43 61.21 3 60.85 60.74 4 59.38 59.38 5 60.30 59.86
    imdb_meta_graph_2=[imdb_meta_p1,imdb_meta_p2,imdb_meta_g1]
    #unweighted59.48 59.51
    #weighted
    if args['hetero']:
        from model_hetero import HAN
        if args['dataset']=="ACM":
            model = HAN(meta_paths=acm_meta_graphs_3,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True).to(args['device'])
        elif args['dataset']=="DBLP":
            model = HAN(meta_paths=dblp_meta_graph_1,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True,
                        weighted=False).to(args['device'])
        else:
            model = HAN(meta_paths=imdb_meta_graph_2,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout'],
                        use_both=True,
                        weighted=False).to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=np.random.randint(0, 500000),
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
