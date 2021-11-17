import math

import torch
from torch.distributions import Normal


class LossFunc():
    def __init__(self, loss_type, task):
        """
        loss_type: [mse, nll]
        task: ["shapenet_3d", "bars", "distractor", "pascal_1d"]
        """
        
        self.loss_type = loss_type
        self.task = task
    
    def calc_loss(self, pr_mu, pr_var, gt_y, test=False):
        if self.loss_type == "mse":
            if self.task == 'distractor':
                loss = torch.sqrt(torch.sum((gt_y - pr_mu) ** 2, dim=-1))
                loss = torch.mean(loss)

            if self.task == "shapenet_3d":
                loss = self.quaternion_loss(gt_y, pr_mu)
            elif self.task == "shapenet_1d":
                if not test:
                    loss = self.azimuth_loss(gt_y, pr_mu)
                if test:
                    loss = self.degree_loss(gt_y, pr_mu)

            elif self.task == "pascal_1d":
                loss = self.mean_square_loss(gt_y, pr_mu)
            return loss

        elif self.loss_type == "nll":
            if self.task == 'distractor':
                # gt_y = gt_y[..., 1:]
                gt_y = gt_y[..., 0:]
            elif self.task == 'bars':
                gt_y = gt_y[..., 0:]
            elif self.task == "shapenet_3d":
                gt_y = gt_y

            sample_num = pr_mu.size(2)

            if self.task == 'distractor':
                gt_y = gt_y[:, :, None, :].repeat(1, 1, sample_num, 1)
                # loss = torch.abs(gt_y - labels_gen)
                # loss = torch.sum(loss, dim=-1)  # sum loss over dim
                # loss = torch.sqrt(torch.sum((gt_y - labels_gen) ** 2, dim=-1))

                # negative joint log-likelihood loss for sampling-based
                # p = Normal(labels_gen, scale_fix)
                pr_var = pr_var[:, :, None, :].repeat(1, 1, sample_num, 1)
                p = Normal(pr_mu, pr_var)
                log_p = p.log_prob(gt_y)
                log_p = torch.sum(log_p, dim=1)  # sum over test samples
                loss = -torch.logsumexp(log_p, dim=1)  # over latent samples
                loss = torch.sum(loss, dim=-1)  # sum two log over output dimension (x and y)
                loss = torch.mean(loss, dim=-1) / self.test_num + torch.log(
                    torch.Tensor([sample_num]).to(self.device)) / self.test_num

                # # mean predictive log-likelihood for deterministic method (parameter-based)
                # p = Normal(labels_gen, scale_gen)
                # log_p = p.log_prob(gt_y)
                # loss = -torch.sum(log_p, dim=-1)
                # loss = torch.mean(loss)

                # loss = torch.mean(loss, dim=-1)
                # loss = torch.stack([torch.abs(gt_y - labels_gen), torch.abs(gt_y - (360 + labels_gen)), torch.abs(labels_gen - (360 + gt_y))], dim=-1)
                # loss = torch.min(loss, dim=-1)[0]

                # gt_y = torch.deg2rad(gt_y[:, :, None])
                # labels_gen = torch.deg2rad(labels_gen)
                # loss = (torch.sin(gt_y) - torch.sin(labels_gen)) ** 2 + (torch.cos(gt_y) - torch.cos(labels_gen)) ** 2
                # loss = torch.abs(torch.cos(gt_y) - torch.cos(labels_gen)) +\
                #        torch.abs(torch.sin(gt_y) - torch.sin(labels_gen))
            elif self.task == 'shapenet_3d':
                gt_y = gt_y[:, :, None, :].repeat(1, 1, sample_num, 1)
                # loss = torch.abs(gt_y - labels_gen)
                # loss = torch.sum(loss, dim=-1)  # sum loss over dim
                # loss = torch.sqrt(torch.sum((gt_y - labels_gen) ** 2, dim=-1))

                # negative joint log-likelihood loss for sampling-based
                # p = Normal(labels_gen, scale_fix)
                pr_var = pr_var[:, :, None, :].repeat(1, 1, sample_num, 1)
                p = Normal(pr_mu, pr_var)
                log_p = p.log_prob(gt_y)
                log_p = torch.sum(log_p, dim=1)  # sum over test samples
                loss = -torch.logsumexp(log_p, dim=1)  # over latent samples
                loss = torch.sum(loss, dim=-1)  # sum two log over output dimension (x and y)
                loss = torch.mean(loss, dim=-1) / self.test_num + torch.log(
                    torch.Tensor([sample_num]).to(self.device)) / self.test_num

            elif self.task == 'bars':
                gt_y = gt_y[:, :, None, :].repeat(1, 1, sample_num, 1)
                loss = (gt_y - pr_mu) ** 2
                loss = torch.mean(loss, dim=-1)  # mean loss over dim
                loss = torch.mean(loss, dim=-1)  # mean loss over samples
                loss = torch.mean(loss, dim=-1)  # mean loss over images per task
                loss = torch.mean(loss, dim=-1)  # mean loss over tasks
            return loss

    def quaternion_loss(self, q_gt, q_pr):
        q_pr_norm = torch.sqrt(torch.sum(q_pr ** 2, dim=-1, keepdim=True))
        q_pr = q_pr / q_pr_norm
        pos_gt_loss = torch.abs(q_gt - q_pr).sum(dim=-1)
        neg_gt_loss = torch.abs(-q_gt - q_pr).sum(dim=-1)
        L1_loss = torch.minimum(pos_gt_loss, neg_gt_loss)
        L1_loss = L1_loss.mean()
        return L1_loss

    def azimuth_loss(self, q_gt, q_pr):
        loss = torch.mean(torch.sum((q_gt[..., :2] - q_pr) ** 2, dim=-1))
        return loss

    def degree_loss(self, q_gt, q_pr):
        q_gt = torch.rad2deg(q_gt[..., -1])
        pr_cos = q_pr[..., 0]
        pr_sin = q_pr[..., 1]
        ps_sin = torch.where(pr_sin >= 0)
        ng_sin = torch.where(pr_sin < 0)
        pr_deg = torch.acos(pr_cos)
        pr_deg_ng = -torch.acos(pr_cos) + 2 * math.pi
        pr_deg[ng_sin] = pr_deg_ng[ng_sin]
        pr_deg = torch.rad2deg(pr_deg)
        errors = torch.stack((torch.abs(q_gt - pr_deg), torch.abs(q_gt + 360.0 - pr_deg), torch.abs(q_gt - (pr_deg + 360.0))), dim=-1)
        errors, _ = torch.min(errors, dim=-1)
        losses = torch.mean(errors)
        return losses

    def mean_square_loss(self, q_gt, q_pr):
        loss = torch.mean((q_gt - q_pr)**2)
        return loss