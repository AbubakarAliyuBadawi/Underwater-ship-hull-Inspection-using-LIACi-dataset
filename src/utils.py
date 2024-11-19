# utils.py
# location (/cluster/home/abubakb/ContMAV/src/utils.py)

import os
import sys

import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=11):
        super().__init__()
        self.n_classes = n_classes
        self.feature_dim = 11
    def forward(self, emb_k, emb_q, labels, epoch, tau=0.1):
        #print(f"emb_k shape: {emb_k.shape}")
        #print(f"emb_q shape: {emb_q.shape}")
        #print(f"labels shape: {labels.shape}")

        if epoch:
            total_loss = torch.tensor(0.0).cuda()
            assert emb_q.shape[0] == labels.shape[0], "mismatch on emb_q and labels shapes!"
            
            # Ensure emb_k has correct shape [n_classes, feature_dim]
            if emb_k.shape[1] != self.feature_dim:
                # Add a reshape or projection to match feature dimensions
                emb_k = F.normalize(emb_k[:, :self.feature_dim] if emb_k.shape[1] > self.feature_dim 
                                  else torch.cat([emb_k, torch.zeros(emb_k.shape[0], self.feature_dim - emb_k.shape[1]).cuda()], dim=1), 
                                  dim=-1)

            # Now emb_k should be [11, 19]
            emb_q = F.normalize(emb_q, dim=1)  # [batch, 19, H, W]

            for i, emb in enumerate(emb_q):
                label = labels[i]
                if not (0 in label.unique() and len(label.unique()) == 1):
                    valid_labels = torch.tensor([l for l in label.unique() if l != 0]).cuda()
                    if len(valid_labels) == 0:
                        continue

                    # Extract features for each valid label
                    label_features = []
                    for lbl in valid_labels:
                        mask = (label == lbl)
                        if mask.sum() == 0:
                            continue
                        # Average features for this class [19]
                        class_features = emb[:, mask].mean(dim=1)
                        label_features.append(class_features)

                    if not label_features:
                        continue

                    # Stack features [num_valid_classes, 19]
                    label_features = torch.stack(label_features)
                    label_features = F.normalize(label_features, dim=1)

                    # Compute similarity
                    similarity = (label_features @ emb_k.T) / tau
                    loss = F.cross_entropy(similarity, torch.arange(len(valid_labels)).cuda())
                    total_loss += loss

            return total_loss / emb_q.shape[0] if emb_q.shape[0] > 0 else torch.tensor(0.0).cuda()

        return torch.tensor(0).cuda()


class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.delta = delta
        self.count = torch.zeros(self.n_classes).cuda()
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        
        # Welford's algorithm for computing running variance
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.var = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        
        self.criterion = torch.nn.L1Loss(reduction="none")
        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.permute(0, 2, 3, 1)
        
        for label in gt_labels:
            if label == 0:  # Skip void class
                continue
                
            # Get true positives for current class
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            
            if tps_current.sum() == 0:
                continue
                
            # Get logits for true positive predictions
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            
            # Calculate mean activation vector
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            
            # Update running mean
            self.features[label] = (self.features[label] * self.count[label] + avg_mav * n_tps)
            self.count[label] += n_tps
            if self.count[label] > 0:
                self.features[label] /= self.count[label]
            
            # Update variance statistics
            centered_logits = logits_tps - avg_mav
            self.ex[label] = torch.mean(centered_logits, dim=0)
            self.ex2[label] = torch.mean(centered_logits ** 2, dim=0)

    def forward(self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: torch.bool) -> torch.Tensor:
        if is_train:
            sem_gt = sem_gt.type(torch.uint8)
            self.cumulate(logits, sem_gt)
            
        if self.previous_features is None:
            return torch.tensor(0.0).cuda()
            
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.permute(0, 2, 3, 1)
        
        acc_loss = torch.tensor(0.0).cuda()
        valid_labels = 0
        
        for label in gt_labels:
            if label == 0:  # Skip void class
                continue
                
            if self.previous_count[label] > 0:
                mav = self.previous_features[label]
                logs = logits_permuted[torch.where(sem_gt == label)]
                
                if logs.shape[0] == 0:
                    continue
                    
                mav = mav.expand(logs.shape[0], -1)
                
                # Calculate L1 loss
                ew_l1 = self.criterion(logs, mav)
                
                # Normalize by variance with safe division
                var = self.var[label].clamp(min=1e-8)
                ew_l1 = ew_l1 / var
                
                # Apply hinge if enabled
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta)
                
                acc_loss += ew_l1.mean()
                valid_labels += 1
        
        # Average loss over valid labels
        if valid_labels > 0:
            acc_loss = acc_loss / valid_labels
            
        return acc_loss.clamp(max=10.0)  # Prevent explosion

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        
        # Update variance using Welford's algorithm
        for c in self.var.keys():
            if self.count[c] > 0:
                self.var[c] = (self.ex2[c] - self.ex[c] ** 2).clamp(min=1e-8)

        # Reset statistics for next epoch
        self.count = torch.zeros(self.n_classes).cuda()
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}

        return self.previous_features, self.var

    def read(self):
        if self.previous_features is None:
            return torch.zeros(self.n_classes, self.n_classes).cuda()
        
        # Get feature dimension from previous features
        feature_dim = next(iter(self.previous_features.values())).shape[0]
        #print(f"MAV feature dimension: {feature_dim}")
        
        mav_tensor = torch.zeros(self.n_classes, feature_dim).cuda()
        for key in range(self.n_classes):
            if key in self.previous_features and self.previous_count[key] > 0:
                mav_tensor[key] = F.normalize(self.previous_features[key], dim=0)
        
        return mav_tensor


class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, sem_gt):
        logits_unk = logits.permute(0, 2, 3, 1)[torch.where(sem_gt == 0)]
        logits_kn = logits.permute(0, 2, 3, 1)[torch.where(sem_gt != 0)]

        if len(logits_unk):
            loss_unk = torch.linalg.norm(logits_unk, dim=1).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - torch.linalg.norm(logits_kn, dim=1)).mean()
        else:
            loss_kn = torch.tensor(0)

        loss = 10 * loss_unk + loss_kn
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="none",
            ignore_index=-1,
        )
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        losses = []
        targets_m = targets.clone()
        if targets_m.sum() == 0:
            import ipdb;ipdb.set_trace()  # fmt: skip
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(
            targets.flatten().type(self.dtype), minlength=self.num_classes
        )
        divisor_weighted_pixel_sum = torch.sum(
            number_of_pixels_per_class[1:] * self.weight
        )  # without void
        if divisor_weighted_pixel_sum > 0:
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
        else:
            losses.append(torch.tensor(0.0).cuda())

        return losses


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(), reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None, reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets_m >= 0)  # only non void pixels

    def compute_whole_loss(self):
        return (
            self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy().item()
        )

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0


def print_log(
    epoch, local_count, count_inter, dataset_size, loss, time_inter, learning_rates
):
    print_string = "Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]".format(
        epoch, local_count, dataset_size, 100.0 * local_count / dataset_size
    )
    for i, lr in enumerate(learning_rates):
        print_string += "   lr_{}: {:>6}".format(i, round(lr, 10))
        print_string += "   Loss: {:0.6f}".format(loss.item())
        print_string += "  [{:0.2f}s every {:>4} data]".format(time_inter, count_inter)
        print(print_string, flush=True)


def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print("{:>2} has been successfully saved".format(path))


def save_ckpt_every_epoch(
    ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mavs, stds
):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_miou_epoch": best_miou_epoch,
        "mavs": mavs,
        "stds": stds,
    }
    ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == "cuda":
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage
            )

        mav_dict = checkpoint["mavs"]
        std_dict = checkpoint["stds"]

        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                model_file, checkpoint["epoch"]
            )
        )
        epoch = checkpoint["epoch"]
        if "best_miou" in checkpoint:
            best_miou = checkpoint["best_miou"]
            print("Best mIoU:", best_miou)
        else:
            best_miou = 0

        if "best_miou_epoch" in checkpoint:
            best_miou_epoch = checkpoint["best_miou_epoch"]
            print("Best mIoU epoch:", best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch, mav_dict, std_dict
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key="mIoU_test"):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, "logs.csv")
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
    assert ckpt_path is not None, f"No trainings found at {ckpt_dir}"
    assert os.path.exists(ckpt_path), f"There is no weights file named {ckpt_path}"
    print(f"Best mIoU: {100*miou:0.2f} at epoch: {epoch}")
    return ckpt_path
