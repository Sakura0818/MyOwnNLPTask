import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from myutils import AverageMeter, calc_psnr





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)

    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    #输出路径
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))


    #如果不存在，则创建目录
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    #这样可以增加程序的运行效率
    cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(args.seed)


    #设置cpu/gpu
    model = SRCNN().to(device)

    #设置损失函数
    criterion = nn.MSELoss()
    logmypsnr = torch.Tensor()
    logmyloss = torch.Tensor()

    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    #设置dataset
    train_dataset = TrainDataset(args.train_file)

    #DataLoader加载
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    #评测数据集
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)


    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0



    
    for epoch in range(args.num_epochs):
        model.train()#训练
        epoch_losses = AverageMeter()#求epoch的平均loss
       
        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
               
            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))
               
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print( '{}_eopch avg loss : {:.8f}'.format(epoch,epoch_losses.avg))
        myloss=torch.full((1,1), epoch_losses.avg)
        logmyloss = torch.cat([logmyloss, myloss])
        torch.save(logmyloss, os.path.join("loss", 'last_loss.pth'))     
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print( '{}_eopch eval psnr: {:.2f}'.format(epoch,epoch_psnr.avg))
        mypsnr=torch.full((1,1), epoch_psnr.avg)
        logmypsnr = torch.cat([logmypsnr, mypsnr])
        torch.save(logmypsnr, os.path.join("mypsnr", 'psnr_epoch.pth'))
         
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    


