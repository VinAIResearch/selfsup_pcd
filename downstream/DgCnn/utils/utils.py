import numpy as np
import torch


def copy_parameters(model, pretrained, verbose=True, part_seg=False):
    feat_dict = model.state_dict()
    # load pre_trained self-supervised
    pretrained_dict = pretrained
    try:
        pretrained_dict = pretrained_dict["model_state_dict"]
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}  # remove name module.
    except Exception:
        print("Not OcCo pretrained")
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


def to_one_hot(y, num_class):
    new_y = torch.eye(num_class)[
        y.cpu().data.numpy(),
    ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all
