# Ref https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py
# Ref https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/utils/Torch_Utility.py
import torch
import torch.nn as nn


def contrastive_loss(feat_1, feat_2, temperature=0.5):
    batch_size = feat_1.size()[0]
    # [2*B, D]
    out = torch.cat([feat_1, feat_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # compute loss
    pos_sim = torch.exp(torch.sum(feat_1 * feat_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def copy_parameters(model, pretrained, verbose=True):
    feat_dict = model.state_dict()
    # load pre_trained self-supervised
    pretrained_dict = pretrained
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()
    }
    if verbose:
        print("=" * 27)
        print("Restored Params and Shapes:")
        for k, v in pretrained_dict.items():
            print(k, ": ", v.size())
        print("=" * 68)
    feat_dict.update(pretrained_dict)
    model.load_state_dict(feat_dict)
    return model


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def init_zeros(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.bias)
        torch.nn.init.zeros_(m.weight)
    else:
        print("Wrong layer TNet")
        exit()
