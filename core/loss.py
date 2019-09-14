import torch
import torch.nn as nn


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def tv(pred_img, real_img, mask):
    output_comp = pred_img*mask + real_img*(1.-mask)
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(output_comp[:, :, :, :-1] - output_comp[:, :, :, 1:])) + \
        torch.mean(torch.abs(output_comp[:, :, :-1, :] - output_comp[:, :, 1:, :]))
    return loss


def hole(pred_img, real_img, mask):
    return nn.L1Loss(mask*pred_img, mask*real_img)


def valid(pred_img, real_img, mask):
    return nn.L1Loss((1.-mask)*pred_img, (1.-mask)*real_img)


def perceptual(pred_feat, real_feat, mask):
    loss = 0.0
    for i in range(3):
        loss += nn.L1Loss(pred_feat[i], real_feat[i])
    return loss 


def style(pred_feat, real_feat, mask):
    loss = 0.0
    for i in range(3):
        loss += nn.L1Loss(gram_matrix(pred_feat[i]), gram_matrix(real_feat[i]))
    return loss

