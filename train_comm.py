import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, OLD_IEMOCAPDataset
from comm_model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
from mmfusion import MMFusion
from comm_loss import CoMMLoss
from comet_ml import Experiment
from smurf_decomp import ThreeModalityModel

from comm import CoMM

# Create an experiment with your Comet API key
experiment = Experiment(
    api_key="Fd1aGmcly8SdDO5Ez4DMyCIt5",           # replace with your actual API key
    project_name="comm-smurf-sdt",     # or whatever project name you want
    workspace="mattam301",             # optional: your Comet workspace name
    auto_param_logging=True,
    auto_metric_logging=False,
)
def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multi_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multi_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
def get_OLD_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = OLD_IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = OLD_IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader



def train_or_eval_model(model, loss_function, kl_loss, comm_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=2.0, gamma_4=1.0, gamma_5=3): # prevent gradient explosion
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, z1, z2, corr_loss, comm_true_out = model(textf, visuf, acouf, umask, qmask, lengths)

        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        
        # if train:
        prototype = -1 # TODO: making use of different prototypes later
        comm_loss_values = comm_loss({
            "aug1_embed": z1,
            "aug2_embed": z2,
            "prototype": prototype  # You need to define/select this somewhere
        })

        # corr_loss, L_unco, L_cor = smurf_model.compute_corr_loss(m1, m2, m3)
        # print("comm_loss:", comm_loss_values)
        
        # print("loss checking...")
        # print("loss1:", loss_function(lp_1, labels_, umask).item())
        # print("loss2:", (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask)).item())
        # print("loss3:", (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask)).item())
        
        loss = 0.0
        if loss_mask[0]:  # Use lp_all loss
            loss += gamma_1 * loss_function(lp_all, labels_, umask)
        if loss_mask[1]:  # Use individual modality losses
            loss += gamma_2 * (
                loss_function(lp_1, labels_, umask) +
                loss_function(lp_2, labels_, umask) +
                loss_function(lp_3, labels_, umask)
            )
        if loss_mask[2]:  # Use KL divergence losses
            loss += gamma_3 * (
                kl_loss(kl_lp_1, kl_p_all, umask) +
                kl_loss(kl_lp_2, kl_p_all, umask) +
                kl_loss(kl_lp_3, kl_p_all, umask)
            )
        if loss_mask[3]:  # Use CoMM loss
            # print("CoMM loss:", comm_loss_values["loss"].item())
            loss += gamma_4 * comm_loss_values["loss"]
        if loss_mask[4]:  # Use Smurf loss
            # print("Synergy and Redundancy Distance (S - 2R):", comm_loss_values["modal_loss"].item())
            # scale = min(1.0, epoch / 50.0)  # slowly increase gamma_5 from 0 to 1 over 5 epochs
            loss += gamma_5 * corr_loss.item()
            # print("Smurf loss:", corr_loss.item())
        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf detected in loss")
                print("Loss:", loss)
                print("lp_all:", torch.isnan(lp_all).any().item(), torch.isinf(lp_all).any().item())
                print("labels_:", torch.isnan(labels_).any().item())
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--loss_mask', type=str, default='1111',
                    help='5-bit string to select loss components: 1st - all_loss, 2nd - individual_losses, 3rd - KL loss, 4th - CoMM loss, 5th - modality balancer loss')
    parser.add_argument('--modality_balancer', action='store_true', default=False, help='use modality balancer to CoMM loss')
    parser.add_argument('--augmentation_style', type=str, default='linear', help='style of augmentation to use')
    parser.add_argument('--late_comm', action='store_true', default=False, help='use late comm loss, either comm is applied in early stage (pre-cross modal)')
    parser.add_argument('--use_smurf', action='store_true', default=False, help='use SMURF decomposition')
    args = parser.parse_args()
    loss_mask = [bool(int(c)) for c in args.loss_mask]  # Converts "1011" -> [True, False, True, True]
    print('loss_mask:', loss_mask)
    assert len(loss_mask) == 5, "loss-mask must be a 5-character string of 0s and 1s"   
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    if args.Dataset != 'OLD_IEMOCAP':
        D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
        D_visual = feat2dim['denseface']
        D_text = 1024
    else:
        D_audio = 100
        D_visual = 512
        D_text = 768

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' or 'OLD_IEMOCAP' else 1

    print('temp {}'.format(args.temp))
    
    import torch.nn as nn
    from typing import List

    # Example configuration
    n_modalities = 3
    input_dims = [1024, 1024, 1024]  # Input dim for each modality
    embed_dim = 1024               # Common token embedding dim

    # Create simple linear adapters (e.g., 512 → 256) for each modality
    input_adapters: List[nn.Module] = [
        nn.Linear(in_dim, embed_dim) for in_dim in input_dims
    ]

    # Now create the MMFusion module
    comm_fuse = MMFusion(
        input_adapters=input_adapters,
        embed_dim=embed_dim,
        fusion="concat",   # or "x-attn"
        pool="cls",        # or "mean"
        n_heads=4,
        n_layers=1,
        add_bias_kv=False,
        dropout=0.1
    )

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout, projection=Transformer_Based_Model._build_mlp(512, 512, 256), comm_fuse=comm_fuse, augmentation_style=args.augmentation_style, late_comm=args.late_comm, use_smurf=args.use_smurf) # need to adjust for sure

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()
        
    kl_loss = MaskedKLDivLoss()
    comm_loss = CoMMLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1, # cheating, i'll change from 0.0 -> 0.1
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    elif args.Dataset == 'OLD_IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_OLD_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, comm_loss, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, comm_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, comm_loss, test_loader, e)
        all_fscore.append(test_fscore)
        
        # Log to Comet
        experiment.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_fscore": train_fscore,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_fscore": valid_fscore,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_fscore": test_fscore
        }, step=e)

        import copy
        if best_fscore == None or best_fscore < valid_fscore and e > 70: # warm-up for 70 epochs, avoid 
            print("Updating best model states with valid_fscore: {}".format(valid_fscore))
            best_fscore = valid_fscore
            best_model_state = copy.deepcopy(model.state_dict())
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
            experiment.log_metric("best_test_fscore", max(all_fscore))
            experiment.log_metric("best_epoch", all_fscore.index(max(all_fscore)) + 1)
            experiment.log_text("Final Classification Report", classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))

    # After training, load the best model
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss,comm_loss, test_loader, n_epochs)
    # Print and log confusion matrix and classification report to Comet
    best_label, best_pred, best_mask = test_label, test_pred, test_mask
    print('Best validation F-Score: {}'.format(test_fscore))
    # print out the classification report
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    # print out the confusion matrix
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    # Log the final test results
    experiment.log_metrics({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "final_test_fscore": test_fscore
    })
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(test_fscore))
    print('Accuracy: {}'.format(test_acc))
    print('Loss: {}'.format(test_loss))
    