import argparse
from math import log10, sqrt
import time
import os
import errno
from os.path import join

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import Dataset
from model import IDN

from myutils import AverageMeter, calc_psnr

torch.manual_seed(1)
my_cuda = True



my_loss='l1'
my_model=""
my_threads=8
#训练设置
my_patch=29

my_start_epoch=1


my_gpus=0

my_images_dir='data'
my_batch_size=16




#    ================>训练周期与lr衰减
my_epochs=1000
my_step=1000


#恢复   Path to checkpoint (default: none)
my_resume=""

#预训练模型的路径（默认值：无）
my_pretrained=None


logmyloss=torch.Tensor()

#优化函数参数
my_lr=0.0001

#模型参数
class modelargs():
    def __init__(self):
        self.n_resgroups=10
        self.n_resblocks=20
        self.n_feats=64
        self.reduction=16
        self.scale=2
        self.rgb_range=255
        self.n_colors=3
        self.res_scale=1
        self.d=16
        self.s=4


def main():




    global my_start_epoch
    cuda = my_cuda
    if cuda:
        print("=> use gpu id: '{}'".format(my_gpus))
        # os.environ["CUDA_VISIBLE_DEVICES"] = my_gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    print("===> Loading datasets")


    train_set = Dataset(my_images_dir, my_patch, 2, False)#pacth=48  scale=2
    training_data_loader = DataLoader(dataset=train_set,
                                     num_workers=my_threads,
                                     batch_size=my_batch_size,
                                     shuffle=True,
									 pin_memory=True,
									 drop_last=True)



    print("===> Building model")

    model = IDN(modelargs())



	#加载
    if my_pretrained is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(my_pretrained, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)


    #损失函数

    if my_loss == 'l1':
        criterion = nn.L1Loss()
    elif my_loss == 'l2':
        criterion = nn.MSELoss()
    else:
        raise Exception('Loss Error: "{}"'.format(opt.loss))

	            #优化函数
    print("===> Setting Optimizer")

    optimizer = optim.Adam(model.parameters(), lr=my_lr)

    
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    logmypsnr = torch.Tensor()
    logmyloss = torch.Tensor()


    print("===> Training")




    for epoch in range(my_start_epoch, my_epochs + 1):
        start_time = time.time()
        train(training_data_loader, optimizer, model, criterion, epoch)
        elapsed_time = time.time() - start_time
        print("Now the {} time is :{}".format(epoch,elapsed_time))
        
        save_checkpoint(model, epoch)




        
    print("All is ok!")
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))





def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = my_lr * (0.5 ** (epoch // my_step))
    return lr



def train(training_data_loader, optimizer, model, criterion, epoch):

    epoch_losses = AverageMeter()

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    epoch_loss = 0
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    #BatchNormalization 和 Dropout
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch


        if my_cuda:
            input = input.cuda()
            target = target.cuda()

        #梯度清除
        optimizer.zero_grad()
       # print(input.size())
       # print(target.size())
       # print("mymodel")

        loss = criterion(model(input), target)
        epoch_loss += loss.item()
       #print("training -------------------")
        #print("loss is{} :".format(loss.item()))
        epoch_losses.update(loss.item(), len(input))

        loss.backward() 


        optimizer.step()

        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                epoch, iteration, len(training_data_loader),
               loss.item()))



    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(
        epoch, epoch_loss / len(training_data_loader)))
    print("ok!--------------------")
    print( '{}_eopch avg loss : {:.8f}'
          .format(epoch,epoch_losses.avg))
    myloss=torch.full((1,1), epoch_losses.avg)

    global logmyloss

    logmyloss = torch.cat([logmyloss, myloss])
    torch.save(logmyloss, os.path.join("loss", 'last_loss.pth'))







def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)

    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(model.state_dict(),model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))




if __name__ == "__main__":
    main()