import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from KD.base_kd import hinton_distillation, hinton_distillation_wo_ce
from utils import StdOutLog, eval, LoggerForSacred
from cmodels.mnist_net import S_LeNet5
import cmodels.ResNet as ResNet
import KD.od_distiller as od_distiller
import gc
import os
from visdom_logger.logger import VisdomLogger
import DA.DA_datasets as DA_datasets
import cmodels.DAN_model as DAN_model
from utils import get_config_var
vars = get_config_var()

save_dir = vars["SAVE_DIR"]

def hinton_train(model, student_model, T, alpha, optimizer, device, train_loader, is_debug=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if torch.cuda.device_count() > 1:
            teacher_logits = model.module.nforward(data)
            student_logits = student_model.module.nforward(data)
        else:
            teacher_logits = model.nforward(data)
            student_logits = student_model.nforward(data)

        loss = hinton_distillation(teacher_logits, student_logits, target, T, alpha)
        total_loss += float(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_debug:
            break

    del loss
    del teacher_logits
    del student_logits
    # torch.cuda.empty_cache()
    return total_loss / len(train_loader)

def hinton_train_without_label(teacher_model, student_model, T, optimizer, device, train_loader, is_debug=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    teacher_model.train()
    student_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if torch.cuda.device_count() > 1:
            teacher_logits = teacher_model.module.nforward(data)
            student_logits = student_model.module.nforward(data)
        else:
            teacher_logits = teacher_model.nforward(data)
            student_logits = student_model.nforward(data)

        loss = hinton_distillation_wo_ce(teacher_logits, student_logits, T)
        total_loss += float(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_debug:
            break

    del loss
    del teacher_logits
    del student_logits
    # torch.cuda.empty_cache()
    return total_loss / len(train_loader)

def hinton_without_label(model, student_model, T, device, trainloader, testloader, optimizer, epochs, **kwargs):
    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    is_debug = False
    if "is_debug" in kwargs:
        is_debug = kwargs["is_debug"]

    best_acc = 0
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        total_loss = hinton_train_without_label(model, student_model, T, optimizer, device, trainloader, is_debug=is_debug)
        t_acc = eval(model, device, testloader)
        s_acc = eval(student_model, device, testloader)

        if logger is not None:
            logger.log_scalar("training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("teacher_val_acc".format(logger_id), t_acc, epoch)
            logger.log_scalar("student_val_acc".format(logger_id), s_acc, epoch)
        if s_acc > best_acc:
            best_acc = s_acc
            torch.save(model, "./{}/best_{}.p".format(save_dir, logger_id))

    return model, optimizer, best_acc

def hinton_kd(model, student_model, T, alpha, device, trainloader, testloader, optimizer, epochs, **kwargs):
    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    save_name = ""
    if "save_name" in kwargs:
        save_name = kwargs["save_name"]


    best_acc = 0
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        total_loss = hinton_train(model, student_model, T, alpha, optimizer, device, trainloader)
        t_acc = eval(model, device, testloader)
        s_acc = eval(student_model, device, testloader)

        if logger is not None:
            logger.log_scalar("training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("teacher_val_acc".format(logger_id), t_acc, epoch)
            logger.log_scalar("student_val_acc".format(logger_id), s_acc, epoch)
        if s_acc > best_acc:
            best_acc = s_acc
            torch.save(model, "./{}/best_{}_{}.p".format(save_dir, logger_id, save_name))

    return model, optimizer, best_acc

def od_distill_train(device, train_loader, d_net, optimizer):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    loss_ce_temp = 0.
    total_loss_temp = 0.

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.to(device)
        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)

        loss_CE = F.cross_entropy(outputs, targets)
        loss = loss_CE + loss_distill.sum() / batch_size / 10000

        loss_ce_temp += loss_CE.item()
        total_loss_temp += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    del loss_CE
    del loss

    return loss_ce_temp, total_loss_temp

def od_distill_train_without_label(train_loader, d_net, optimizer):
    d_net.train()
    if torch.cuda.device_count() > 1:
        d_net.module.s_net.train()
        d_net.module.t_net.train()
    else:
        d_net.s_net.train()
        d_net.t_net.train()

    loss_ce_temp = 0.
    total_loss_temp = 0.

    for i, (inputs, _) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss = loss_distill.sum() / batch_size / 10000

        total_loss_temp += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    del loss

    return loss_ce_temp, total_loss_temp

def od_kd_without_label(epochs, teacher_net, student_net, distiller_net, optimizer, trainloader, testloader,
                        device, **kwargs):
    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    for epoch in range(1, epochs + 1):

        # train for one epoch
        loss_ce, total_loss = od_distill_train_without_label(trainloader, distiller_net, optimizer)
        t_acc = eval(teacher_net, device, testloader)
        s_acc = eval(student_net, device, testloader)

        if logger is not None:
            logger.log_scalar("training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("teacher_val_acc".format(logger_id), t_acc, epoch)
            logger.log_scalar("student_val_acc".format(logger_id), s_acc, epoch)

        if scheduler:
            scheduler.step()

        # evaluate on validation set
        gc.collect()

def od_kd(epochs, teacher_net, student_net, distiller_net, optimizer, trainloader, testloader, device, **kwargs):
    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    for epoch in range(1, epochs + 1):

        # train for one epoch
        loss_ce, total_loss = od_distill_train(device, trainloader, distiller_net, optimizer)
        t_acc = eval(teacher_net, device, testloader)
        s_acc = eval(student_net, device, testloader)


        if logger is not None:
            logger.log_scalar("training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("teacher_val_acc".format(logger_id), t_acc, epoch)
            logger.log_scalar("student_val_acc".format(logger_id), s_acc, epoch)

        if scheduler:
            scheduler.step()

        # evaluate on validation set
        gc.collect()


def main_hinton_kd():
    batch_size = 64
    test_batch_size = 64
    lr = 0.01
    momentum = 0.9
    epochs = 10
    T = 20
    alpha = 0.3

    device = torch.device("cuda")

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=1)

    teacher_model = torch.load("{}/best_mnist.p".format(save_dir)).to(device)
    t_acc = eval(teacher_model, device, testloader)
    print(t_acc)

    student_model = S_LeNet5().to(device)
    optimizer = torch.optim.SGD(list(teacher_model.parameters()) + list(student_model.parameters()), momentum=momentum,
                                lr=lr)

    hinton_kd(teacher_model, student_model, T, alpha, device, trainloader, testloader, optimizer, epochs,
              logger=StdOutLog(), logger_id="mnist")


def main():
    batch_size = 64
    test_batch_size = 64
    lr = 0.1
    momentum = 0.9
    epochs = 100
    epoch_step = 30
    weight_decay = 1e-4
    teacher_pretrained_path = "{}/dan_resnet50_amazon_2_webcam.pth".format(save_dir)
    student_pretrained = False
    device = torch.device("cuda")

    webcam = os.path.expanduser("~/datasets/webcam/images")
    amazon = os.path.expanduser("~/datasets/amazon/images")
    dslr = os.path.expanduser("~/datasets/dslr/images")

    train_loader_source = DA_datasets.office_loader(amazon, batch_size, 0)
    train_loader_target = DA_datasets.office_loader(webcam, batch_size, 0)
    testloader_target = DA_datasets.office_test_loader(webcam, test_batch_size, 0)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)

    teacher_model = DAN_model.DANNet_ResNet(ResNet.resnet50, True)
    student_model = DAN_model.DANNet_ResNet(ResNet.resnet34, student_pretrained)

    if teacher_pretrained_path != "":
        teacher_model.load_state_dict(torch.load(teacher_pretrained_path))

    if torch.cuda.device_count() > 1:
        teacher_model = torch.nn.DataParallel(teacher_model).to(device)
        student_model = torch.nn.DataParallel(student_model).to(device)

    distiller_model = od_distiller.Distiller_DAN(teacher_model, student_model)

    if torch.cuda.device_count() > 1:
        distiller_model = torch.nn.DataParallel(distiller_model).to(device)

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.SGD(list(student_model.parameters()) + list(distiller_model.module.Connectors.parameters()),
                                    lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.SGD(list(student_model.parameters()) + list(distiller_model.Connectors.parameters()),
                                    lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epoch_step)

    od_kd_without_label(epochs, teacher_model, student_model, distiller_model, optimizer, train_loader_target,
                        testloader_target, device, logger=logger, scheduler=scheduler)


if __name__ == "__main__":
    main()
