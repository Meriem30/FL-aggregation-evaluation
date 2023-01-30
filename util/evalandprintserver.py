
import torch
import numpy


def evalandprintserver(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                 best_changed,
                 train_loss, val_loss):
    # evaluation on training data
    for i in range(args.n_clients):
        train_loader_all = train_loaders[i]
    train_loss, train_acc = algclass.server_eval(train_loader_all)
    print(' Server | Train Loss: {:.4f} | Train Acc: {:.4f}'.format( train_loss, train_acc))

    # evaluation on valid data
    for i in range(args.n_clients):
        val_loader_all = val_loaders[i]
    val_loss, val_acc = algclass.server_eval(val_loader_all)
    print(' Server | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(val_loss, val_acc))


    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for i in range(args.n_clients):
            test_loader_all = test_loaders[i]
        _, test_acc = algclass.server_eval(test_loader_all)
        print('Server | Epoch:{} | Test Acc: {:.4f}'.format(best_epoch, test_acc))
        best_tacc = test_acc



    return best_acc, best_tacc, best_changed, train_loss, val_loss