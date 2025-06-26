import numpy as np
import pandas as pd
import os
import torch
from torch import nn

import json
import time
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer
import sranodec as anom

from pretrainmodel.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



parser = argparse.ArgumentParser()

# Dataset and dataloader
parser.add_argument("--dataset", type=str.upper, default="SMAP")
parser.add_argument("--graph", type=str, default='gat')
parser.add_argument("--num", type=str, default="6", help="Required for ASD dataset. <group_index>")
parser.add_argument('--dset_pretrain', type=str.upper, default='SMAP', help='dataset name')
parser.add_argument("--group", type=str, default="1-6", help="Required for SMD dataset. <group_index>-<index>")
# parser.add_argument('--context_points', type=int, default=512, help='sequence length')
# parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument("--lookback", type=int, default=60)  # 1
parser.add_argument('--topK', type=int, default=5, help='k')  # 1
parser.add_argument('--embed_size', type=int, default=25, help='embed_size')
parser.add_argument("--normalize", type=str2bool, default=True)
parser.add_argument("--spec_res", type=str2bool, default=False)
parser.add_argument('--device', type=str, default='cuda:0', help='')
# 1D conv layer
parser.add_argument("--kernel_size", type=int, default=7)

# --- Train params ---
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--log_tensorboard", type=str2bool, default=True)


parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--target_points', type=int, default=60, help='patch length')
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--epochs', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--init_lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)

args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.lookback)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.epochs) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dataset + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

normalize = args.normalize
group_index = args.group[0]
index = args.group[2:]

# get available GPU devide
set_device()


def get_model(n_features, args):
    """
    c_in: number of variables
    """
    # get number of patches
    # num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1
    num_patch = (max(args.lookback, args.patch_len)-args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    model = PatchTST(c_in=n_features,
                     target_dim=args.lookback,
                     patch_len=args.patch_len,
                     stride=args.stride,
                     num_patch=num_patch,
                     n_layers=args.n_layers,
                     n_heads=args.n_heads,
                     d_model=args.d_model,
                     shared_embedding=True,
                     d_ff=args.d_ff,
                     dropout=args.dropout,
                     head_dropout=args.head_dropout,
                     act='relu',
                     head_type='pretrain',
                     res_attention=False
                     )
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# def find_lr():
#     # get dataloader
#     # dls = get_dls(args)
#     # model = get_model(dls.vars, args)
#     dataset=args.dataset
#     dl = get_data(dataset, normalize=normalize)
#     model = get_model(args)
#     # get loss
#     loss_func = torch.nn.MSELoss(reduction='mean')
#     # get callbacks
#     cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
#     cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
#
#     # define learner
#     learn = Learner(dls, model,
#                         loss_func,
#                         lr=args.lr,
#                         cbs=cbs,
#                         )
#     # fit the data to the model
#     suggested_lr = learn.lr_finder()
#     print('suggested_lr', suggested_lr)
#     return suggested_lr


def pretrain_func(lr=args.init_lr):
    # get dataloader
    # dls = get_dls(args)
    (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    # get model     
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


























#  12月6日注销暂时不用
# if __name__ == '__main__':
#
#     # args.dset = args.dset_pretrain
#     # suggested_lr = find_lr()
#     # # Pretrain
#     # pretrain_func(suggested_lr)
#     # print('pretraining completed')
#     today = time.time()
#     timeInfo = {}
#
#     id = datetime.now().strftime("%d%m%Y_%H%M%S")
#
#     # parser = get_parser()
#     # args = parser.parse_args()
#
#     args = parser.parse_args()
#     # 获取参数，进行初始化
#     device = torch.device(args.device)
#     dataset = args.dataset
#     window_size = args.lookback
#     spec_res = args.spec_res
#     normalize = args.normalize
#     n_epochs = args.epochs
#     batch_size = args.batch_size
#     init_lr = args.init_lr
#     val_split = args.val_split
#     shuffle_dataset = args.shuffle_dataset
#     use_cuda = args.use_cuda
#     print_every = args.print_every
#
#     log_tensorboard = args.log_tensorboard
#     group_index = args.group[0]
#     index = args.group[2:]
#     # threshold = args.threshold
#     num = args.num
#     args_summary = str(args.__dict__)
#     print(args_summary)  # 利用魔术方法打印获取到的参数 通过字典
#
#     if dataset == 'SMD':
#         output_path = f'output/SMD/{args.group}'
#         (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
#     elif dataset == 'ASD':
#         output_path = f'output/ASD/{args.num}'
#         (x_train, _), (x_test, y_test) = get_data(f"omi-{num}", normalize=normalize)
#     elif dataset in ['MSL', 'SMAP', 'SMDALL']:
#         output_path = f'output/{dataset}'
#         # (x_train, _), (x_test, y_test) = get_data(dataset, max_train_size=200, max_test_size=100, normalize=normalize)
#         (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
#     else:
#         raise Exception(f'Dataset "{dataset}" not available.')
#
#     if dataset in ['MSL', 'SMAP']:
#         # less than period
#         amp_window_size = 3
#         # (maybe) as same as period
#         series_window_size = 5
#         # a number enough larger than period
#         score_window_size = 100
#         spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
#         # 獲取異常分數
#         score = spec.generate_anomaly_score(x_train[:, 0])
#         index_changes = np.where(score > np.percentile(score, 90))[0]
#         x_train[:, 0] = anom.substitute(x_train[:, 0], index_changes)
#
#     log_dir = f'{output_path}/logs'
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     save_path = f"{output_path}/{id}"
#
#     x_train = torch.from_numpy(x_train).float()
#     x_test = torch.from_numpy(x_test).float()
#     n_features = x_train.shape[1]
#
#     target_dims = get_target_dims(dataset)
#     if target_dims is None:
#         out_dim = n_features
#         print(f"Will forecast and reconstruct all {n_features} input features")
#     elif type(target_dims) == int:
#         print(f"Will forecast and reconstruct input feature: {target_dims}")
#         out_dim = 1
#     else:
#         print(f"Will forecast and reconstruct input features: {target_dims}")
#         out_dim = len(target_dims)
#
#     train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
#     test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
#
#     train_loader, val_loader, test_loader = create_data_loaders(
#         train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
#     )
#
#     model = get_model(n_features, args)
#     # model = MTAD_GAT(
#     #     n_features,
#     #     window_size,
#     #     out_dim,
#     #     gru_n_layers=args.gru_n_layers,  # 移除GRU
#     #     gru_hid_dim=args.gru_hid_dim,  # 移除GRU
#     #     forecast_n_layers=args.fc_n_layers,
#     #     forecast_hid_dim=args.fc_hid_dim,
#     #     dropout=args.dropout,
#     #     k=args.topK,
#     #     embed_size=args.embed_size,
#     #     in_channels=args.in_channels,
#     #     graph=args.graph
#     # )
#
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         total_params += param
#     print(f'参数： {total_params}')
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
#     forecast_criterion = nn.MSELoss()
#     # recon_criterion = nn.MSELoss()
#
#     trainer = Trainer(
#         model,
#         optimizer,
#         window_size,
#         n_features,
#         target_dims,
#         n_epochs,
#         batch_size,
#         init_lr,
#         forecast_criterion,
#         # recon_criterion,
#         use_cuda,
#         save_path,
#         log_dir,
#         print_every,
#         log_tensorboard,
#         args_summary
#     )
#     # Save config
#     args_path = f"{save_path}/config.txt"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     with open(args_path, "w") as f:
#         json.dump(args.__dict__, f, indent=2)
#
#     before_fit = time.time()
#     timeInfo['before fit'] = before_fit - today
#     trainer.fit(train_loader, val_loader, timeInfo)
#     after_fit = time.time()
#     timeInfo['fit times'] = after_fit - before_fit
#
#     plot_losses(trainer.losses, save_path=save_path, plot=False)
#