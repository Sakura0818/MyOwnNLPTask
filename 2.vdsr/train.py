import argparse
from math import log10, sqrt
import time
import os
import errno
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import DatasetFromHdf5
from model import VDSR

from myutils import AverageMeter


my_cuda = True

#测试设置
my_TorF_test= False
my_test_batch_size=128

my_gpuids=[0]
my_model=""
my_threads = 0
#训练设置

my_start_epoch=1

my_epochs=50
my_gpus=0


my_batch_size=128
my_clip=0.4
my_crop_size=64
my_lr=0.1

#    ================>衰减
my_step=10
my_momentum=0.9

my_upscale_factor=2
my_weight_decay=0.0001


#恢复   Path to checkpoint (default: none)
my_resume=""

#预训练模型的路径（默认值：无）
my_pretrained=""


logmyloss=torch.Tensor()


def main():




    global my_start_epoch
    cuda = my_cuda
    if cuda:
        print("=> use gpu id: '{}'".format(my_gpus))
        # os.environ["CUDA_VISIBLE_DEVICES"] = my_gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set,
                                     num_workers=my_threads,
                                     batch_size=my_batch_size,
                                     shuffle=True)

    print("===> Building model")
    model = VDSR()
    criterion = nn.MSELoss(size_average=False)

    
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()




    # optionally resume from a checkpoint
    #继续训练
    if my_resume:
        if os.path.isfile(my_resume):
            print("=> loading checkpoint '{}'".format(my_resume))
            checkpoint = torch.load(my_resume)
            
            my_start_epoch = checkpoint["epoch"] + 1
            print(my_start_epoch)
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(my_resume))

    # optionally copy weights from a checkpoint
    #加载模型
    if my_pretrained:
        if os.path.isfile(my_pretrained):
            print("=> loading model '{}'".format(my_pretrained))
            weights = torch.load(my_pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(my_pretrained))  




            #优化函数
    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(),
                         lr=my_lr,
                         momentum=my_momentum,
                         weight_decay=my_weight_decay)

    print("===> Training")

    for epoch in range(my_start_epoch, my_epochs + 1):
        start_time = time.time()
        train(training_data_loader, optimizer, model, criterion, epoch)
        elapsed_time = time.time() - start_time
        print("Now the {} time is :{}".format(epoch,elapsed_time))
        
        save_checkpoint(model, epoch)
        
    print("All is ok!")




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = my_lr * (0.1 ** (epoch // my_step))
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
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if my_cuda:
            input = input.cuda()
            target = target.cuda()

        #梯度清除
        optimizer.zero_grad()

        loss = criterion(model(input), target)
        epoch_loss += loss.item()
#         print("training -------------------")
#         print("loss is{} :".format(loss.item()))
        epoch_losses.update(loss.item(), len(input))

        loss.backward() 

        nn.utils.clip_grad_norm_(model.parameters(),my_clip) 

        optimizer.step()

        if iteration%1000 == 0:
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
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))




if __name__ == "__main__":
    main()