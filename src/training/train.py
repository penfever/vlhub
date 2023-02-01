import json
import logging
import math
import os
import time
import datetime
from contextlib import suppress
from itertools import chain
import random

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, tokenize, SIMCLRLoss, IntLoss, ClipLossIQE, ClipLossAlignUnif
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .data import get_total_obj
from .precision import get_autocast

from grad_cache_vl.grad_cache import GradCache

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def train_integer_labels(model, images, labels, device, loss):
    logits = model(images)
    return loss(logits, labels)

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, sim_clr=False):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    #TODO: implement temperature for SIMCLR
    if sim_clr:
        loss = SIMCLRLoss(
            0.1,
            args
        )
    elif args.integer_labels:
        loss = IntLoss(
            args,
            device
        )
    elif args.iqe:
        loss = ClipLossIQE(
            img_weight=args.img_weight,
            text_weight=args.text_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod
        )
    elif args.alignunif:
        loss = ClipLossAlignUnif(
            img_weight=args.img_weight,
            text_weight=args.text_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod
        )
    else:
        loss = ClipLoss(
            img_weight=args.img_weight,
            text_weight=args.text_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)

    #IMAGENET TUNING LOOP
    if args.imagenet_tune_freq > 0 and epoch % args.imagenet_tune_freq == 0:
        intloss = IntLoss(
            args,
            device
        )
        logging.info("ImageNet-tuning of vision head, epoch {}".format(epoch))
        for i, batch in enumerate(data["imagenet-train"].dataloader):
            images, labels = batch
            labels = labels.to(device=device, non_blocking=True)
            images = images.to(device=device, non_blocking=True)
            args.optimizer_tune.zero_grad()
            imagenet_loss = train_integer_labels(unwrap_model(model).visual, images, labels, device, intloss)

            if i % 100 == 0:
                logging.info("Batch {}, total loss {}".format(i, imagenet_loss))

            if scaler is not None:
                scaler.scale(imagenet_loss).backward()
                scaler.step(args.optimizer_tune)
                scaler.update()
            else:
                imagenet_loss.backward()
                args.optimizer_tune.step()

    #MAIN TRAINING LOOP PREP
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    #Gradient Caching
    if args.gc:
        if args.horovod:
            print("horovod is not currently enabled for gradient caching")
            raise NotImplementedError
        if args.precision != 'fp32':
            if args.distributed:
                print("The following combination is not yet supported: gradient caching, mixed precision, DDP")
                print("Please try: gradient caching, fp32, DDP or gradient caching, amp, single GPU")
                raise NotImplementedError
            gc = GradCache(
                models=[model, model], 
                chunk_sizes=args.gpumaxbatch, 
                loss_fn=loss,
                fp16=True,
                scaler=scaler
            )
        else:
            gc = GradCache(
                models=[model , model], 
                chunk_sizes=args.gpumaxbatch, 
                loss_fn=loss,
                fp16=False,
                scaler=scaler
            )

    batchset = list()
    in1k_sm_list = [0]
    in1k_nsm_list = [0]

    #MAIN TRAINING LOOP
    for i, batch in enumerate(dataloader):
        #HOUSEKEEPING
        #     for b in batch[1].tolist():
        #         if b not in batchset:
        #             batchset.append(b)

        #PREP BATCH
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images, texts = batch
        texts = texts.to(device=device, non_blocking=True)
        images = images.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)

        if args.dry_run:
            if args.integer_labels:
                for t in texts:
                    if t[1] == -1:
                        in1k_sm_list[0] += 1
                    else:
                        in1k_nsm_list[0] += 1
            if i % 100 == 0:
                logging.info("Dry run, batch {} of {}".format(i, num_batches_per_epoch))
                if args.integer_labels:
                    logging.info("In1k Matches = {} ({} strict, {} not)".format(in1k_sm_list[0] + in1k_nsm_list[0], in1k_sm_list[0], in1k_nsm_list[0]))
                    # logging.info("Total samples processed: {}".format(get_total_obj() * args.workers))
            continue
        optimizer.zero_grad()

        # LOSS
        with autocast():
            if args.sim_clr:
                #"TEXTS" is actually another image file, in the case of SIMCLR                    
                outputs = unwrap_model(model)(images, texts)
                total_loss = loss(outputs)
                ssl_loss = total_loss['ssl_loss']
                acc = total_loss['ssl_acc']
                if i % 100 == 0:
                    logging.info("SSL ACC: {}".format(acc))
                total_loss = total_loss['loss']
            elif args.integer_labels:
                total_loss = train_integer_labels(unwrap_model(model).visual, images, texts, device, loss)
            elif args.gc:
                if args.alt:
                    raise("gradient caching not supported yet for this model, sorry!")
                total_loss, logit_scale_scalar = gc([images, texts], vl_model=True, no_sync_except_last=args.distributed, lock_img=(args.lock_image_freeze_bn_stats or args.lock_image), scaler=scaler)
            elif args.alt:
                if args.model == "xclip":
                    total_loss = model(
                        texts,
                        images,
                        freeze_image_encoder = args.lock_image,
                        return_loss = True  # set this to True to get the full caption + contrastive loss
                    )                
                else:
                    total_loss = model(
                        text = texts,
                        images = images,
                        return_loss = True  # set this to True to get the full caption + contrastive loss
                    )
            else:                    
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = loss(image_features, text_features, logit_scale)

            #BACKWARD           
            if scaler is not None:
                if args.gc:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.scale(total_loss).backward()
                    if not torch.isfinite(total_loss):
                        logging.info("Loss is NaN -- skipping batch {}".format(i))
                        optimizer.zero_grad()
                        try:
                            torch.cuda.empty_cache()
                        except:
                            print("No cuda cache to free")
                        continue
                    if args.horovod:
                        optimizer.synchronize()
                        scaler.unscale_(optimizer)
                        if args.norm_gradient_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                        with optimizer.skip_synchronize():
                            scaler.step(optimizer)
                    else:
                        if args.norm_gradient_clip is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                        scaler.step(optimizer)
                    scaler.update()
            else:
                if not args.gc:
                    total_loss.backward()
                    if args.norm_gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    if not torch.isfinite(total_loss):
                        logging.info("Loss is NaN -- skipping batch {}".format(i))
                        optimizer.zero_grad()
                        try:
                            torch.cuda.empty_cache()
                        except:
                            print("No cuda cache to free")
                        continue
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if not args.alt and not args.iqe:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        STEP_COUNT = 10 if args.debug else 100
        if is_master(args) and (i % STEP_COUNT == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            if np.isnan(loss_m.val):
                logging.debug("NaN loss in logging function on iteration {}".format(i))
            if args.alt or args.integer_labels:
                logit_scale = torch.tensor([1.0])
            if not args.gc:
                logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            # Early stopping of epoch if ramping condition is met
            if args.ramping:
                if percent_complete > (epoch + 1) * 5:
                    print("Ramping: stopping epoch early")
                    return
    # end for
    #HOUSEKEEPING
    # if args.ds_filter and args.debug:
    #     logging.debug("The model saw {} unique samples this epoch".format(len(batchset)))
    # if args.integer_labels:
    #     logging.info("In1k strict match count was {}, non_strict was {}".format(in1k_sm_list[0], in1k_nsm_list[0]))
    #     if args.wandb:
    #         assert wandb is not None, 'Please install wandb.'
    #         wandb.log({"in1k_strict_match": in1k_sm_list[0], 'in1k_non_strict_match': in1k_nsm_list[0]})

def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)

    
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # FIXME hacking a solution for large batch sizes to avoid evaluation overloading memory
                # this will result in less accurate evaluation metrics
                if len(batch[0]) > args.gpumaxbatch:
                    images = batch[0][:args.gpumaxbatch]
                    texts = batch[1][:args.gpumaxbatch]
                else:
                    images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                
                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics