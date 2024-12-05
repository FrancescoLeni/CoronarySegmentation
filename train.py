import argparse
import os

import torch
from pathlib import Path

from models import check_load_model
from models.common import Dummy, UNet, UNet3D, UNetBig, ConvNeXtUnet
from models.Rep_ViT import RepViTUnet, RepViTUnet3D
from utils.callbacks import Callbacks, EarlyStopping, Saver
from utils.loaders import load_all
from utils.optimizers import get_optimizer, scheduler
from utils.losses import SemanticLosses, CELoss
from utils.metrics import Metrics
from utils.logger import Loggers
from utils import random_state, increment_path, json_from_parser, my_logger
from models import ModelClass

# setting all random states
random_state(36)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def main(args):

    # unpacking
    folder = args.folder
    name = args.name

    # weighted_loss = args.weighted_loss
    weighted_loss = True

    data_path = args.data_path

    # creating saving location
    p = Path('runs') / 'train'
    if folder:
        p = p / folder
    os.makedirs(p, exist_ok=True)
    save_path = increment_path(p, name)
    name = save_path.stem
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    out_classes = args.n_class + 1  # +1 for bkg

    # saving inputs
    json_from_parser(args, save_path)

    # model (ADJUST)
    if "." not in args.model:
        # means it is not a weight and has to be imported ADJUST => (NEED TO IMPORT IT)
        if args.model == "Dummy":
            model = Dummy()
        elif args.model == 'Unet':
            model = UNet(out_classes)
        elif args.model == 'UnetBig':
            model = UNetBig(n_classes=out_classes)
            # loading model = bla bla bla
        elif args.model == 'RepViT':
            model = RepViTUnet('m2', img_size=args.crop_size,  n_classes=out_classes, fuse=True)
        elif args.model == 'ConvNeXt':
            model = ConvNeXtUnet(n_classes=out_classes)
        elif args.model == 'RepViT3D':
            model = RepViTUnet3D(n_classes=out_classes)
        elif args.model == 'Unet3D':
            model = UNet3D(out_classes)
        else:
            raise TypeError("Model name not recognised")
    else:
        # it is a weight
        model = args.model

    # double-checking whether you parsed weights or model and accounting for transfer learning
    mod = check_load_model(model, args.backbone, my_logger)

    # check for 3D
    flag_3d = any(isinstance(module, torch.nn.Conv3d) for module in model.modules())

    # loading dataset already as iterable torch loaders (train, val ,(optional) test)
    loaders = load_all(data_path, args.reshape_mode, args.crop_size, batch_size=batch_size, test_flag=False,
                       scaler=args.scaler, n_workers=args.n_workers, pin_memory=args.pin_memory, flag_3D=flag_3d)

    # initializing callbacks ( could be handled more concisely i guess...)
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=mod, save_best=True, save_path=save_path, monitor="val_loss", mode='min')
    callbacks = Callbacks([stopper, saver])

    if weighted_loss:
        weights_dict = {64: [0.0188, 0.9812], 128: [0.0056, 0.9944], 256: [0.0019, 0.9981]}
        weights = torch.tensor(weights_dict[args.crop_size], dtype=torch.float32)

        if torch.cuda.is_available() and args.device == 'gpu':
            weights = weights.to('cuda:0')
    else:
        weights = None

    # initializing loss and optimizer
    loss_fn = SemanticLosses(alpha=1, gamma=1, lambdas=(0.5, 0.5), weight=weights)
    # loss_fn = CELoss(weights=weights)

    opt = get_optimizer(mod, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # for encoder only it is just empty, ADJUST for decoder then
    metrics = Metrics(loss_fn=loss_fn, num_classes=out_classes, device=device, top_k=1, thresh=0.5)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, test=False)

    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building model
    model = ModelClass(mod, loaders, info_log=my_logger, loss_fn=loss_fn, device=device, AMP=args.AMP,
                       optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched,
                       grad_clip=args.grad_clip_norm)

    # training the model
    model.train_loop(epochs)


if __name__ == "__main__":

    # list of arguments (ADJUST for student and SAM)
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model', type=str, required=True, help='name of model to train')
    parser.add_argument('--backbone', type=str, default=None, help='path to backbone weights, if present it ONLY loads weights for it')

    # classes (excluding bkg)
    parser.add_argument('--n_class', type=int, default=1, help='the number of classes to segment (excluding bkg)')

    # reshaping BOTH needed
    parser.add_argument('--reshape_mode', type=str, default='crop', choices=[None, 'crop', 'grid'], help=" how to handle resize")
    parser.add_argument('--crop_size', type=int, default=128, help='the finel shape input to model')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'min_max'], help='name of the scaler to use')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--folder', type=str, default=None, help='name of folder to which saving results inside runs/train')
    parser.add_argument('--name', type=str, default="exp", help='name of experiment folder inside folder')
    parser.add_argument('--opt', type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help='name of optimizer to use')
    parser.add_argument('--sched', type=str, default=None, choices=["linear", "cos_lr"], help="name of the lr scheduler")
    parser.add_argument('--lr0', type=float, default=0.0004, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.001, help='final learning rate (multiplicative factor)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (SGD) beta1 (Adam, AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay value')
    parser.add_argument('--lab_smooth', type=float, default=0, help='label smoothing value')
    parser.add_argument('--patience', type=int, default=30, help='number of epoch to wait for early stopping')
    parser.add_argument('--device', type=str, default="gpu", choices=["cpu", "gpu"], help='device to which loading the model')
    parser.add_argument('--AMP', action="store_true", help='whether to use AMP')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)

    # probably not userfull (done by default)
    # parser.add_argument('--weighted_loss', action="store_true", help='whether to weight the loss and weight for classes')

    # datasets (ADJUST)
    parser.add_argument('--data_path', type=str, default='ASOCA_DATASET', help='path to dataset')

    # loaders params
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for parallel dataloading ')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory for more efficient passage to gpu')

    args = parser.parse_args()

    main(args)


