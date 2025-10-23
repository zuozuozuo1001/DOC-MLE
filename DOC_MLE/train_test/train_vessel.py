import os
from DOC_MLE import Constants
import numpy as np
import torch
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from time import time
from DOC_MLE.data_process.data_load import ImageFolder,get_drive_data
from DOC_MLE.networks.net import UNet_Student_Teacher
from DOC_MLE.train_test.losses import loss_ce
from DOC_MLE.train_test.eval_test import val_vessel
from torch.utils.tensorboard import SummaryWriter
from DOC_MLE.train_test.help_functions import platform_info, check_size
from DOC_MLE.train_test.evaluations import threshold_by_otsu
from torchinfo import summary
from PIL import Image
import math
from DOC_MLE.data_process.data_ultils import data_shuffle
from DOC_MLE.gmodel.utils.gconv_utils import get_basis_params, get_basis_filters, get_rot_info
import torch.nn.functional as F
from torch import autograd
learning_rates = Constants.learning_rates
gcn_model = False


def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))          # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def DiceLoss(predict, target):
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)

    pre = predict.view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
    union = (pre + tar).sum(-1).sum()

    score = 1 - 2 * (intersection + epsilon) / (union + epsilon)

    return score

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def load_model(path):
    net = torch.load(path)
    return net

def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio

def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr


def optimizer_neteq(net, optimizers, criterion, images, masks, ch):
    optimizers.zero_grad()
    pred = net(images, 8)
    loss = loss_ce(pred, masks, criterion,ch)
    loss.backward()
    optimizers.step()
    return pred, loss


def optimizer_net(net1, net2, optimizers, criterion, img_piexllabel, mask_piexllabel, img_unlabel, ch, count, epoch, shuzu):

    ###################################################net1_piexl########################################################
    pred_piexllabel5, pred_piexllabel4, pred_piexllabel3, pred_piexllabel2, pred_piexllabel1, _, _, _, _, _, prot_piexllabel = net1(img_piexllabel, Confin=True)
    pred_unlabel_st5, pred_unlabel_st4, pred_unlabel_st3, pred_unlabel_st2, pred_unlabel_st1, _, _, _, _, _, prot_unlabel_st， loss_ort = net1(img_unlabel, Confin=False， epoch=epoch, total_epochs=50)

    loss_sup_bce5 = loss_ce(pred_piexllabel5, mask_piexllabel, criterion, ch)
    loss_sup_bce4 = loss_ce(pred_piexllabel4, mask_piexllabel, criterion, ch)
    loss_sup_bce3 = loss_ce(pred_piexllabel3, mask_piexllabel, criterion, ch)
    loss_sup_bce2 = loss_ce(pred_piexllabel2, mask_piexllabel, criterion, ch)
    loss_sup_bce1 = loss_ce(pred_piexllabel1, mask_piexllabel, criterion, ch)

    loss_sup_dice5 = DiceLoss(pred_piexllabel5, mask_piexllabel)
    loss_sup_dice4 = DiceLoss(pred_piexllabel4, mask_piexllabel)
    loss_sup_dice3 = DiceLoss(pred_piexllabel3, mask_piexllabel)
    loss_sup_dice2 = DiceLoss(pred_piexllabel2, mask_piexllabel)
    loss_sup_dice1 = DiceLoss(pred_piexllabel1, mask_piexllabel)

    loss_sup = 0.1 * (loss_sup_bce5 + loss_sup_bce4 + loss_sup_bce3 + loss_sup_bce2 + loss_sup_bce1 + loss_sup_dice5 + loss_sup_dice4 + loss_sup_dice3 + loss_sup_dice2 + loss_sup_dice1)
    
    with torch.no_grad():
        noise = torch.clamp(torch.randn_like(img_unlabel) * 0.05, -0.2, 0.2)
        img_unlabel_n = img_unlabel + noise
        pred_unlabel_te5, pred_unlabel_te4, pred_unlabel_te3, pred_unlabel_te2, pred_unlabel_te1, feat_unlabel_te5, feat_unlabel_te4, feat_unlabel_te3, feat_unlabel_te2, feat_unlabel_te1, prot_unlabel_te = net2(img_unlabel_n, Confin=True)


        heihei37 = pred_unlabel_te1.clone()
        heihei37[heihei37 >= 0.7] = 1
        heihei37[heihei37 <= 0.3] = 1
        heihei37[heihei37 < 1] = 0
        heihei37 = torch.sum(heihei37)
        shuzu[1] = shuzu[1] + heihei37


        heihei28 = pred_unlabel_te1.clone()
        heihei28[heihei28 >= 0.8] = 1
        heihei28[heihei28 <= 0.2] = 1
        heihei28[heihei28 < 1] = 0
        heihei28 = torch.sum(heihei28)
        shuzu[2] = shuzu[2] + heihei28

        heihei19 = pred_unlabel_te1.clone()
        heihei19[heihei19 >= 0.9] = 1
        heihei19[heihei19 <= 0.1] = 1
        heihei19[heihei19 < 1] = 0
        heihei19 = torch.sum(heihei19)
        shuzu[3] = shuzu[3] + heihei19


    ##########################################三个标准权重计算######################################################################################
    #原型匹配程度wegiht1
    weight_te5 = abs(F.cosine_similarity(prot_piexllabel, feat_unlabel_te5)).unsqueeze(1)
    weight_te4 = abs(F.cosine_similarity(prot_piexllabel, feat_unlabel_te4)).unsqueeze(1)
    weight_te3 = abs(F.cosine_similarity(prot_piexllabel, feat_unlabel_te3)).unsqueeze(1)
    weight_te2 = abs(F.cosine_similarity(prot_piexllabel, feat_unlabel_te2)).unsqueeze(1)
    weight_te1 = abs(F.cosine_similarity(prot_piexllabel, feat_unlabel_te1)).unsqueeze(1)
    weight_te = torch.cat([weight_te5, weight_te4, weight_te3, weight_te2, weight_te1], dim=1)
    weight_te_softmax = F.softmax(weight_te, dim=1)
    weight1_5, weight1_4, weight1_3, weight1_2, weight1_1 = weight_te_softmax[:,:1,:,: ], weight_te_softmax[:,1:2,:,:], weight_te_softmax[:,2:3,:,: ], weight_te_softmax[:,3:4,:,:], weight_te_softmax[:,4:,:,:]

    # 计算不确定性de权重weight2
    eps = 1e-6
    uncertain5 = 1 + (pred_unlabel_te5 * torch.log(pred_unlabel_te5 + eps) + (1.0 - pred_unlabel_te5) * torch.log((1.0 - pred_unlabel_te5 + eps)))
    uncertain4 = 1 + (pred_unlabel_te4 * torch.log(pred_unlabel_te4 + eps) + (1.0 - pred_unlabel_te4) * torch.log((1.0 - pred_unlabel_te4 + eps)))
    uncertain3 = 1 + (pred_unlabel_te3 * torch.log(pred_unlabel_te3 + eps) + (1.0 - pred_unlabel_te3) * torch.log((1.0 - pred_unlabel_te3 + eps)))
    uncertain2 = 1 + (pred_unlabel_te2 * torch.log(pred_unlabel_te2 + eps) + (1.0 - pred_unlabel_te2) * torch.log((1.0 - pred_unlabel_te2 + eps)))
    uncertain1 = 1 + (pred_unlabel_te1 * torch.log(pred_unlabel_te1 + eps) + (1.0 - pred_unlabel_te1) * torch.log((1.0 - pred_unlabel_te1 + eps)))
    uncertain = torch.cat([uncertain5, uncertain4, uncertain3, uncertain2, uncertain1], dim=1)
    uncertain_softmax = F.softmax(uncertain, dim=1)
    weight2_5, weight2_4, weight2_3, weight2_2, weight2_1 = uncertain_softmax[:,:1,:,: ], uncertain_softmax[:,1:2,:,:], uncertain_softmax[:,2:3,:,: ], uncertain_softmax[:,3:4,:,:], uncertain_softmax[:,4:,:,:]

    #统计正负个数de权重weight3
    pred_unlabel_te5_bi = torch.where(pred_unlabel_te5>=0.5, 1.0, 0.0)
    pred_unlabel_te4_bi = torch.where(pred_unlabel_te4>=0.5, 1.0, 0.0)
    pred_unlabel_te3_bi = torch.where(pred_unlabel_te3>=0.5, 1.0, 0.0)
    pred_unlabel_te2_bi = torch.where(pred_unlabel_te2>=0.5, 1.0, 0.0)
    pred_unlabel_te1_bi = torch.where(pred_unlabel_te1>=0.5, 1.0, 0.0)
    pred_unlabel_te_total = pred_unlabel_te5_bi + pred_unlabel_te4_bi + pred_unlabel_te3_bi + pred_unlabel_te2_bi + pred_unlabel_te1_bi
    pred_unlabel_te_post = pred_unlabel_te_total/5.0
    pred_unlabel_te_nega = 1 - (pred_unlabel_te_total/5.0)
    weight3_post5, weight3_post4, weight3_post3, weight3_post2, weight3_post1 = pred_unlabel_te_post * pred_unlabel_te5_bi, pred_unlabel_te_post * pred_unlabel_te4_bi, \
                                                                                 pred_unlabel_te_post * pred_unlabel_te3_bi, pred_unlabel_te_post * pred_unlabel_te2_bi, pred_unlabel_te_post * pred_unlabel_te1_bi
    weight3_nega5, weight3_nega4, weight3_nega3, weight3_nega2, weight3_nega1 = torch.where(weight3_post5<=0, pred_unlabel_te_nega, weight3_post5), torch.where(weight3_post4<=0, pred_unlabel_te_nega, weight3_post4), \
                                                                                torch.where(weight3_post3<=0, pred_unlabel_te_nega, weight3_post3), torch.where(weight3_post2<=0, pred_unlabel_te_nega, weight3_post2), torch.where(weight3_post1<=0, pred_unlabel_te_nega, weight3_post1)
    weight3_5, weight3_4, weight3_3, weight3_2, weight3_1 = weight3_post5+weight3_nega5, weight3_post4+weight3_nega4, weight3_post3+weight3_nega3, weight3_post2+weight3_nega2, weight3_post1+weight3_nega1
    ##########################################三个标准权重计算######################################################################################

    pred_unlabel_te_new1 = weight1_5 * pred_unlabel_te5 + weight1_4 * pred_unlabel_te4 + weight1_3 * pred_unlabel_te3 + weight1_2 * pred_unlabel_te2 + weight1_1 * pred_unlabel_te1
    pred_unlabel_te_new2 = weight2_5 * pred_unlabel_te5 + weight2_4 * pred_unlabel_te4 + weight2_3 * pred_unlabel_te3 + weight2_2 * pred_unlabel_te2 + weight2_1 * pred_unlabel_te1
    pred_unlabel_te_new3 = weight3_5 * pred_unlabel_te5 + weight3_4 * pred_unlabel_te4 + weight3_3 * pred_unlabel_te3 + weight3_2 * pred_unlabel_te2 + weight3_1 * pred_unlabel_te1

    pred_unlabel_te_new = (pred_unlabel_te_new3 + pred_unlabel_te_new2 + pred_unlabel_te_new1)/3.0

    with torch.no_grad():
        haha37 = pred_unlabel_te_new.clone()
        haha37[haha37 >= 0.7] = 1
        haha37[haha37 <= 0.3] = 1
        haha37[haha37 < 1] = 0
        haha37 = torch.sum(haha37)
        shuzu[4] = shuzu[4] + haha37

        haha28 = pred_unlabel_te_new.clone()
        haha28[haha28 >= 0.8] = 1
        haha28[haha28 <= 0.2] = 1
        haha28[haha28 < 1] = 0
        haha28 = torch.sum(haha28)
        shuzu[5] = shuzu[5] + haha28

        haha19 = pred_unlabel_te_new.clone()
        haha19[haha19 >= 0.9] = 1
        haha19[haha19 <= 0.1] = 1
        haha19[haha19 < 1] = 0
        haha19 = torch.sum(haha19)
        shuzu[6] = shuzu[6] + haha19
        shuzu[0] = shuzu[0]+1

    loss_unsup_confince = nn.MSELoss()(pred_unlabel_st1, pred_unlabel_te_new)

    # loss_unsup_confince = weighted_mse_loss(pred_unlabel_st, pred_unlabel_te, weight_te)

    loss_unsup_cos = torch.mean(1 - F.cosine_similarity(prot_unlabel_st, prot_unlabel_te))

    if epoch<=50:
        consistency_weight = math.exp(-0.01 * (50 - epoch) ** 1.4)
    else:
        consistency_weight = 1.0

    loss_consistency = (loss_unsup_confince + loss_unsup_cos)/2.0

    # print(consistency_weight)
    loss = loss_sup + consistency_weight * loss_consistency + 0.1 * loss_ort
 
    optimizers.zero_grad()
    # with autograd.detect_anomaly():
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(parameters=net1.parameters(), max_norm=10, norm_type=2)
    optimizers.step()

    update_ema_variables(net1, net2, 0.99, count)

    return pred_piexllabel1, loss

def optimizer_net_baseline(net1, net2, optimizers, criterion, img_piexllabel, mask_piexllabel, img_unlabel, ch, count):

    ###################################################net1_piexl########################################################
    pred_piexllabel = net1(img_piexllabel)

    loss_sup_bce = loss_ce(pred_piexllabel, mask_piexllabel, criterion, ch)
    loss_sup_dice = DiceLoss(pred_piexllabel, mask_piexllabel)
    loss_sup = loss_sup_bce + loss_sup_dice


    optimizers.zero_grad()
    loss_sup.backward()
    optimizers.step()


    return pred_piexllabel, loss_sup

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(ema=False):
    model = UNet_Student_Teacher().to(device)
    # gmodel = UNet().to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def visual_preds(preds, is_preds=True):  # This for multi-classification
    rand_arr = torch.rand(size=(preds.size()[1], preds.size()[2], 3))
    color_preds = torch.zeros_like(rand_arr)
    outs = preds.permute((1, 2, 0))  # N H W C
    if is_preds is True:
        outs_one_hot = torch.argmax(outs, dim=2)
    else:
        outs_one_hot = outs.reshape((preds.size()[1], preds.size()[2]))
    for H in range(0, preds.size()[1]):
        for W in range(0, preds.size()[2]):
            if outs_one_hot[H, W] == 1:
                color_preds[H, W, 0] = 255
            if outs_one_hot[H, W] == 2:
                color_preds[H, W, 1] = 255
            if outs_one_hot[H, W] == 3:
                color_preds[H, W, 2] = 255
            if outs_one_hot[H, W] == 4:
                color_preds[H, W, 0] = 255
                color_preds[H, W, 1] = 255
                color_preds[H, W, 2] = 255
    return color_preds.permute((2, 0, 1))

def train_model(learning_rates):
    writer = SummaryWriter(comment=f"MyDRIVETrain01", flush_secs=1)
    tic = time()
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    ch = Constants.BINARY_CLASS
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()

    net1 = create_model(ema=False) #student gmodel
    net2 = create_model(ema=True) #teacher gmodel
    # summary(net1, input_data=torch.rand(Constants.BATCH_SIZE, 1, 288, 288))
    # parameters = list(net1.parameters()) + list(net2.parameters())
    optimizers = optims.Adam(net1.parameters(), lr=learning_rates, betas=(0.9, 0.999), weight_decay=1e-5)
    trains, val = get_drive_data()
    rand_img, rand_label, rand_pred = None, None, None
    count = 0
    shuzu = [0] * 7
    for epoch in range(1, total_epoch + 1):
        trains[0], trains[1] = data_shuffle(trains[0], trains[1])
        trains[2], trains[3] = data_shuffle(trains[2], trains[3])
        dataset = ImageFolder(trains[0], trains[1], trains[2], trains[3])
        data_loader = data.DataLoader(dataset, batch_size=Constants.BATCH_SIZE, shuffle=True, num_workers=8)
        net1.train()
        # net2.train()
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img_piexllabel, mask_piexllabel, img_unlabel, _ in data_loader_iter:
            img_piexllabel = img_piexllabel.to(device)
            mask_piexllabel = mask_piexllabel.to(device)
            img_unlabel = img_unlabel.to(device)
            # pred, train_loss = optimizer_neteq(net1, optimizers, criterion, img_piexllabel, mask_piexllabel,ch)
            pred, train_loss = optimizer_net(net1, net2, optimizers, criterion, img_piexllabel, mask_piexllabel, img_unlabel, ch, count, epoch, shuzu)
            # pred, train_loss = optimizer_net_baseline(net1, net2, optimizers, criterion, img_piexllabel, mask_piexllabel,img_unlabel, ch, count)
            train_epoch_loss += train_loss.item()
            if np.random.rand(1) > 0.4 and np.random.rand(1) < 0.8:
                rand_img, rand_label, rand_pred = img_piexllabel, mask_piexllabel, pred
            count+=1

        C1 = shuzu[1] / (shuzu[0] * 2)
        C2 = shuzu[2] / (shuzu[0] * 2)
        C3 = shuzu[3] / (shuzu[0] * 2)
        C4 = shuzu[4] / (shuzu[0] * 2)
        C5 = shuzu[5] / (shuzu[0] * 2)
        C6 = shuzu[6] / (shuzu[0] * 2)

        s1 = 'epoch:{}, single37:{}, single28:{}, single19:{}, duoge37:{}, duoge28:{}, duoge19:{},'.format(epoch, C1, C2, C3, C4, C5, C6)
        with open(os.path.join('.../DOC_MLE/log', 'tongji.txt'), 'a', encoding='utf-8') as f:
            f.write(s1)
            f.write('\n')

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        if ch ==1:      # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()
            rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()
            writer.add_scalar('Train/acc', rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0], epoch)  # for [N,H,W,1]
        if ch ==2:      # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers)
        # platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers2)
        if epoch % 10 == 1:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 1:  # for [N,1,H,W]
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 2:  # for [N,2,H,W]
                  writer.add_image('Train/image_predictions', torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                             epoch)
        update_lr2(epoch, optimizers)  # modify  lr
        # update_lr2(epoch, optimizers2)


        print('************ start to validate current gmodel {}.iter performance ! ************'.format(epoch))
        acc, sen, f1score, val_loss = val_vessel(net1, val[0], val[1], val[0].shape[0], epoch)
        writer.add_scalar('Val/accuracy', acc, epoch)
        writer.add_scalar('Val/sensitivity', sen, epoch)
        writer.add_scalar('Val/f1score', f1score, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)



        model_name1 = Constants.saved_path + "{}.iter1".format(epoch)
        torch.save(net1, model_name1)





    print('***************** Finish training process ***************** ')

if __name__ == '__main__':
    RANDOM_SEED = 42  # any random number
    # RANDOM_SEED = 3407
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


    set_seed(RANDOM_SEED)
    train_model(learning_rates)
    pass






