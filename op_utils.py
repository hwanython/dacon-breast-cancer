import tqdm
import torch
import torch.optim as optim
from metric import AverageMeter
import time
from tqdm import tqdm
import numpy as np
import xgboost as xgb
import pickle

def train_loop(train_loader, model, criterion, optimizer, epoch, device):
    """
        Training loop
        Args:
            train_loader  (pytorch dataloader)  :
            model         (pytorch model)       :
            criterion     (pytorch loss)        :
            optimizer     (pytorch optimizer)   :
            device        (str)
        Returns:
            sum_loss    (pytorch Scalar)    : training loss의 평균
            sum_acc     (pytorch Scalar)    : training accuracy의 평균
            model       (pytorch model)     : trained model
    """
    tic = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_acc = 0.0
    train_acc = 0.0

    model.train()
    tbar = tqdm(train_loader, ncols=130)
    for idx, (inputs, labels, tabular) in enumerate(tbar):
        data_time.update(time.time() - tic)
        if device == 'cuda':
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            tabular = tabular.cuda().float()

        # train operation
        optimizer.zero_grad()
        outputs = model(inputs, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        scheduler.step()
        #
        # # SAM optimizer
        # optimizer.first_step(zero_grad=True)
        #
        # # sam optimizer second_steop
        # criterion(model(inputs), labels).backward()
        # optimizer.second_step(zero_grad=True)

        pred = outputs
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        # recording running metric
        total_loss.update(loss.item())
        correct = (pred == labels).float().mean()
        # train_correct += correct
        batch_time.update(time.time() - tic)
        tic = time.time()
        total_acc += correct
        train_acc = total_acc / (idx + 1)
        tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} | B {:.2f} D {:.2f} |'.format(
            epoch, total_loss.average,
            train_acc, batch_time.average, data_time.average))

    total_acc = total_acc / len(train_loader)
    return total_loss.average, total_acc, model


def val_loop(val_loader, model, criterion, epoch, device):
    """
        Validation loop
        Args:
            train_loader  (pytorch dataloader)  :
            model         (pytorch model)       :
            criterion     (pytorch loss)        :
            device        (str)
        Returns:
            sum_loss    (pytorch Scalar)    : training loss의 평균
            sum_acc     (pytorch Scalar)    : training accuracy의 평균
    """
    tic = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_acc = 0.0
    train_acc = 0.0

    model.eval()
    with torch.no_grad():
        tbar = tqdm(val_loader, ncols=130)
        for idx, (inputs, labels, tabular) in enumerate(tbar):
            data_time.update(time.time() - tic)
            if device == 'cuda':
                inputs = inputs.cuda()
                labels = labels.cuda().float()
                tabular = tabular.cuda().float()

            # validation operation
            outputs = model(inputs, tabular)
            loss = criterion(outputs, labels)
            pred = outputs
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            # recording running metric
            total_loss.update(loss.item())
            correct = (pred == labels).float().mean()
            # train_correct += correct
            batch_time.update(time.time() - tic)
            tic = time.time()
            total_acc += correct
            train_acc = total_acc / (idx + 1)

            tbar.set_description('EVAL ({}) | Loss: {:.3f} | Acc {:.2f} | B {:.2f} D {:.2f} |'.format(
                epoch, total_loss.average,
                train_acc, batch_time.average, data_time.average))

    total_acc = total_acc / len(val_loader)
    return total_loss.average, total_acc, pred


def inference(val_loader, model, device):
    """
        Validation loop
        Args:
            train_loader  (pytorch dataloader)  :
            model         (pytorch model)       :
            criterion     (pytorch loss)        :
            device        (str)
        Returns:
            sum_loss    (pytorch Scalar)    : training loss의 평균
            sum_acc     (pytorch Scalar)    : training accuracy의 평균
    """
    tic = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    preds = []
    model.eval()
    with torch.no_grad():
        tbar = tqdm(val_loader, ncols=130)
        for idx, (inputs, tabular) in enumerate(tbar):
            data_time.update(time.time() - tic)
            if device == 'cuda':
                inputs = inputs.cuda()
                tabular = tabular.cuda().float()

            # validation operation
            outputs = model(inputs, tabular)
            pred = outputs
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0


            # recording running metric

            batch_time.update(time.time() - tic)
            tic = time.time()
            tbar.set_description('PRED B {:.2f} D {:.2f} |'.format(batch_time.average, data_time.average))
            preds.extend(pred.cpu().detach().numpy())

    return np.array(preds, dtype=np.float16).flatten()
