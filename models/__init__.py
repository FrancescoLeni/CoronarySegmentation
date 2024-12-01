import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys
import logging


# SETTING GLOBAL VARIABLES
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def check_load_model(model, backbone_weights):
    if not backbone_weights:
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, str) and Path(model).suffix == ".pt" or ".pth":
            return torch.load(model, map_location=torch.device('cpu'))
        else:
            raise TypeError("model not recognised")
    else:
        # I'm loading only the weights from the backbone

        assert isinstance(model, nn.Module)  # check that the model is something to load weights to

        old = torch.load(backbone_weights)
        filtered_state_dict = {k: old.state_dict()[k] for k in old.state_dict() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)

        return model


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='cpu', callbacks=None, loss_fn=None, optimizer=None, sched=None,
                 metrics=None, loggers=None, AMP=True, freeze=None, info_log=None):
        super().__init__()
        """
        :param
            --model: complete Torch model to train/test
            --loaders: tuple with the Torch data_loaders like (train,val,test)
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
            --loggers: loggers instance
            --AMP: Automatic Mixed Precision 
            --freeze: list containing names of layers to freeze
            --sequences: to handle windowed input sequences
        """

        self.freeze = freeze

        assert isinstance(info_log, logging.Logger), 'provided info_log is not a logger'
        self.my_logger = info_log

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError("model not recognised")

        self.train_loader, self.val_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                self.my_logger.info('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        self.my_logger.info(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.callbacks = callbacks

        self.metrics = metrics
        self.metrics1 = None

        self.loggers = loggers

        if loss_fn:
            self.loss_fun = loss_fn.to(self.device)
            self.opt = optimizer
            self.sched = sched

        if AMP and "cuda" in self.device:
            self.my_logger.info("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False

        total_params = sum(p.numel() for p in self.model.parameters())
        self.my_logger.info(f"'{self.model.name}' - Total parameters: {total_params}")

    def train_one_epoch(self, epoch_index, tot_epochs):
        self.loss_fun.reset()

        # initializing progress bar
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)

            inputs, labs = data
            inputs = inputs.to(self.device)
            labs = labs.to(self.device)

            self.opt.zero_grad()

            if self.AMP:
                with autocast():
                    outputs = self.model(inputs)

                    loss = self.loss_fun(outputs, labs)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                outputs = self.model(inputs)

                loss = self.loss_fun(outputs, labs)
                loss.backward()
                self.opt.step()

            del inputs

            current_loss = self.loss_fun.get_current_value(batch)

            with torch.no_grad():
                # computing training metrics
                self.metrics.on_train_batch_end(outputs.float(), labs, batch)
                # calling callbacks
                self.callbacks.on_train_batch_end(outputs.float(), labs, batch)

            # updating pbar
            # if self.metrics.num_classes != 2:
            #     A = self.metrics.A.t_value_mean
            #     P = self.metrics.P.t_value_mean
            #     R = self.metrics.R.t_value_mean
            #     AUC = self.metrics.AuC.t_value_mean
            # else:
            #     A = self.metrics.A.t_value_mean
            #     P = self.metrics.P.t_value[1]
            #     R = self.metrics.R.t_value[1]
            #     AUC = self.metrics.AuC.t_value[1]

            # pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
            #                             f'train_loss: {last_loss:.4f}, A: {A :.2f}, P: {P :.2f}, R: {R :.2f}, AUC: {AUC :.2f}')

            pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs - 1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                        f'train_loss: {current_loss:.4f}')
            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(batch + 1)

    def val_loop(self, epoch):
        self.loss_fun.reset()

        # resetting metrics for validation
        self.metrics.on_val_start()

        # calling callbacks
        self.callbacks.on_val_start()

        # initializing progress bar
        description = f'Validation'
        pbar_loader = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # Disable gradient computation and reduce memory consumption, and model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            for batch, data in pbar_loader:
                torch.cuda.empty_cache()  # Clear GPU memory

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                _ = self.loss_fun(outputs, labels)

                current_loss = self.loss_fun.get_current_value(batch)

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(outputs.float(), labels, batch)
                # calling callbacks
                self.callbacks.on_val_batch_end(outputs, labels, batch)
                # updating roc and prc
                self.loggers.on_val_batch_end(outputs, labels, batch)

                # # updating pbar
                # if self.metrics.num_classes != 2:
                #     A = self.metrics.A.v_value_mean
                #     P = self.metrics.P.v_value_mean
                #     R = self.metrics.R.v_value_mean
                #     AUC = self.metrics.AuC.v_value_mean
                # else:
                #     A = self.metrics.A.v_value_mean
                #     P = self.metrics.P.v_value[1]
                #     R = self.metrics.R.v_value[1]
                #     AUC = self.metrics.AuC.v_value[1]
                # description = f'Validation: val_loss: {last_loss:.4f}, val_A: {A :.2f}, ' \
                #               f'val_P: {P :.2f}, val_R: {R :.2f}, val_AUC: {AUC :.2f}'
                description = f'Validation: val_loss: {current_loss:.4f}'
                pbar_loader.set_description(description)

        if outputs is not None:
            # updating metrics dict
            self.metrics.on_val_end(batch + 1)

            # updating loggers (roc, prc)
            self.loggers.on_val_end()
            # calling callbacks
            self.callbacks.on_val_end(self.metrics.dict, epoch)

    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            self.metrics.on_epoch_start()

            self.loggers.on_epoch_start(epoch=epoch, max_epoch=num_epochs)

            # self.model.train(True)
            self.check_freeze()  # freezing specific layers (if needed)

            # for name, param in self.model.named_parameters():
            #     self.my_logger.info(name, param.requires_grad)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)

            # validation
            self.val_loop(epoch)

            # logging results
            self.loggers.on_epoch_end(epoch)
            # updating lr scheduler
            if self.sched:
                self.sched.step()
            #resetting metrics
            self.metrics.on_epoch_end()

            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                self.my_logger.info(f"early stopping at epoch {epoch}")
                break

        # self.loggers.on_epoch_end(0)

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model)
        self.callbacks.on_end()

    def check_freeze(self):
        if self.freeze:
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in self.freeze):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            self.model.train(True)




