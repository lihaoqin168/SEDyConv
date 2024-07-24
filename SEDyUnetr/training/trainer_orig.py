# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from base_utils.utils import distributed_all_gather, print_to_log_file, get_temperature
from monai.networks import one_hot

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    train_loader_len = len(loader)
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        if hasattr(args, 'temp_epoch') and epoch < args.temp_epoch and hasattr(model, 'net_update_temperature'):
            temp = get_temperature(idx + 1, epoch, train_loader_len,
                                   temp_epoch=args.temp_epoch, temp_init=args.temp_init)
            model.net_update_temperature(temp)
            print_to_log_file(args.logfile,'net_update_temperature', temp)

        if hasattr(args, 'dyAtt_epoch') and epoch == args.dyAtt_epoch and hasattr(model, 'net_update_dyAttBlocks'):
            model.dy_flg =  True
            print_to_log_file(args.logfile,'net_update_dyAttBlocks')

        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if loss==0.0:#for loss for MO with out background
            loss = torch.tensor(0.0)
        else:
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print_to_log_file(args.logfile,
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, loss_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    run_loss = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
                valid_loss = loss_func(logits, target)
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_onehot = one_hot(target, num_classes=args.out_channels, dim=1)
            val_outputs = (logits == logits.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)

            acc_func.reset()
            acc_func(y_pred=val_outputs, y=val_labels_onehot)
            acc, not_nans = acc_func.aggregate()

            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans],
                                                                 out_numpy=True,
                                                                 is_valid=idx < loader.sampler.valid_length)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

                loss_list = distributed_all_gather([valid_loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=1
                )
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                run_loss.update(valid_loss.cpu().numpy())
            if args.rank == 0:
                print_to_log_file(args.logfile, 'Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                                  'acc', acc.item(),
                                  # 'avg_acc_acc_org', avg_acc_acc_org,
                                  "valid_loss",
                                  valid_loss.item(),
                                  'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
            del data
            del target
            del val_outputs
            del val_labels_onehot
            del logits
            # empty_cache
            torch.cuda.empty_cache()
        return np.mean(run_acc.avg), run_loss.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def save_checkpoint_minLoss(model, epoch, args, filename="model_minloss.pt", min_loss=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_min_loss": min_loss, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    print_to_log_file(args.logfile, "run_training")
    print_to_log_file(args.logfile, args)
    print_to_log_file(args.logfile, model)
    print_to_log_file(args.logfile, loss_func)

    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    val_loss_min = 100.0


    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print_to_log_file(args.logfile, args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args
        )
        if args.rank == 0:
            print_to_log_file(args.logfile,
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        b_new_best_loss = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            #empty_cache
            torch.cuda.empty_cache()
            val_avg_acc, valid_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print_to_log_file(args.logfile,
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "valid_loss",
                    valid_loss,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                    writer.add_scalars("train", {"train_loss":train_loss,"valid_loss":valid_loss,"val_avg_acc":val_avg_acc}, epoch)

                if val_avg_acc > val_acc_max:
                    print_to_log_file(args.logfile,"new best acc ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
                if valid_loss < val_loss_min:
                    print_to_log_file(args.logfile,"new best val_loss_min ({:.6f} --> {:.6f}). ".format(val_loss_min, valid_loss))
                    val_loss_min = valid_loss
                    b_new_best_loss = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint_minLoss(
                            model, epoch, args, min_loss=val_loss_min, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print_to_log_file(args.logfile,"Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
                if b_new_best_loss:
                    print_to_log_file(args.logfile,"Copying to model_minloss.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model_minloss.pt"))


        if scheduler is not None:
            scheduler.step()

    print_to_log_file(args.logfile,"Training Finished !, Best Accuracy: ", val_acc_max)
    print_to_log_file(args.logfile,"Training Finished !, Best min loss: ", val_loss_min)

    return val_acc_max
