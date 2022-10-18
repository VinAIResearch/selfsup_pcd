# Ref https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py
# Ref https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/utils/Torch_Utility.py
import torch
import torch.nn as nn


def copy_parameters(model, pretrained, verbose=True, part_seg=False):
    feat_dict = model.state_dict()
    # load pre_trained self-supervised
    pretrained_dict = pretrained
    try:
        pretrained_dict = pretrained_dict["model_state_dict"]
        pretrained_dict = {k[12:]: v for k, v in pretrained_dict.items()}  # remove name module.
    except Exception:
        print("Not OcCo pretrained")
    predict = {}
    if part_seg:
        for k, v in pretrained_dict.items():
            if k == "mlp1.0.weight" or k == "mlp1.0.bias":
                predict[k.replace("mlp1.0", "conv1")] = v
            elif k in [
                "mlp1.1.weight",
                "mlp1.1.bias",
                "mlp1.1.running_mean",
                "mlp1.1.running_var",
                "mlp1.1.num_batches_tracked",
            ]:
                predict[k.replace("mlp1.1", "bn1")] = v
            elif k == "mlp2.3.weight" or k == "mlp2.3.bias":
                predict[k.replace("mlp2.3", "conv2")] = v
            elif k in [
                "mlp2.4.weight",
                "mlp2.4.bias",
                "mlp2.4.running_mean",
                "mlp2.4.running_var",
                "mlp2.4.num_batches_tracked",
            ]:
                predict[k.replace("mlp2.4", "bn2")] = v
            else:
                predict[k] = v
        # pretrained_dict = {k.replace('mlp1.0', 'conv1'): v for k, v in pretrained_dict.items() if k == 'mlp1.0.weight' or k == 'mlp1.0.bias'}
        # pretrained_dict = {k.replace('mlp1.1', 'bn1'): v for k, v in pretrained_dict.items() if k in ['mlp1.1.weight', 'mlp1.1.bias', 'mlp1.1.running_mean', 'mlp1.1.running_var', 'mlp1.1.num_batches_tracked']}
        # pretrained_dict = {k.replace('mlp2.3', 'conv2'): v for k, v in pretrained_dict.items() if k == 'mlp2.3.weight' or k == 'mlp2.3.bias'}
        # pretrained_dict = {k.replace('mlp2.4', 'bn2'): v for k, v in pretrained_dict.items() if k in ['mlp2.4.weight', 'mlp2.4.bias', 'mlp2.4.running_mean', 'mlp2.4.running_var', 'mlp2.4.num_batches_tracked']}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                 k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()}
        pretrained_dict = {
            k: v for k, v in predict.items() if k in feat_dict and predict[k].size() == feat_dict[k].size()
        }
    else:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()
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
        # m.eps =1e-3
        # print(m)


def to_one_hot(y, num_class):
    new_y = torch.eye(num_class)[
        y.cpu().data.numpy(),
    ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            torch.nn.init.zeros_(m.bias)
        except Exception:
            pass
        # print(m)


def init_zeros(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.bias, 1e-5)
        torch.nn.init.zeros_(m.weight)
    else:
        print("Wrong layer TNet")
        exit()
