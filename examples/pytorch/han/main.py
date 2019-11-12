import torch
from sklearn.metrics import f1_score
import warnings
from utils import load_data, EarlyStopping

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
    meta_g1=[[['pa', 'ap'], ['pf', 'fp']]]
    meta_g2=[[['pa', 'ap'], ['pt', 'tp']]]
    meta_g3=[[['pf', 'fp'], ['pf', 'fp']]]
    meta_p1=[[['pa', 'ap']]]
    meta_p2=[[['pf', 'fp']]]
    meta_p3=[[['pt', 'tp']]]
    meta_graphs_1 = [meta_p1, meta_p2] ##91.58 91.58
    meta_graphs_2 = [meta_g1, meta_p1] ##91.44 91.37
    meta_graphs_3 = [meta_g1, meta_p2] ##90.94 90.95
    meta_graphs_4 = [meta_g1,meta_p1, meta_p2] ##92.26 92.26
    meta_graphs_5 = [meta_g1, meta_p1, meta_p2,meta_p3] ##92.05 92.08
    meta_graphs_6 = [meta_p1, meta_p2,meta_p3]  ##91.37 91.38
    meta_graphs_7 = [meta_p1, meta_p2,meta_p3,meta_g1,meta_g2,meta_g3] ##91.55,91.55
    meta_graphs_8 = [meta_g3] ##87.75 87.39


    dblp_meta_p1=[[['ap','pa']]]
    dblp_meta_p2=[[['ap','pc','cp','pa']]]
    dblp_meta_p3=[[['ap','pt','tp','pa']]]
    dblp_meta_p4=[[['ap','pa','ap','pa']]]
    dblp_meta_g1=[[['ap','pa'],['ap','pc','cp','pa']]] #co-author and same conference
    dblp_meta_g2=[[['ap','pc','cp','pa'],['ap','pt','tp','pa']]]
    dblp_meta_graph_1=[dblp_meta_p1,dblp_meta_p2,dblp_meta_p3] #92.92 92.35
    dblp_meta_graph_2=[dblp_meta_p1,dblp_meta_p2,dblp_meta_p3,dblp_meta_g1] ##92.99 92.44
    dblp_meta_graph_3=[dblp_meta_p1,dblp_meta_p2,dblp_meta_p3,dblp_meta_g2] ##93.45 92.95
    dblp_meta_graph_4=[dblp_meta_p1,dblp_meta_g2] ##93.34 92.83

    if args['hetero']:
        from model_hetero import HAN
        if not args['dblp']:
            model = HAN(meta_paths=meta_graphs_8,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
        else:
            model = HAN(meta_paths=dblp_meta_graph_4,
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
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
    parser.add_argument('-s', '--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--dblp',action='store_true')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
