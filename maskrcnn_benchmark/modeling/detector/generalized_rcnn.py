# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from fDAL import fDALDivergenceHead


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        from maskrcnn_benchmark.config import cfg
        if self.roi_heads:
            if hasattr(self, 'fdalhead') is False:
                in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # this is specificied in the faster-cnn config.
                
                # Creating manually the auxiliary head. In this example the auxiliary head is a discriminator.
                aux_head = nn.Sequential(
                    nn.Conv2d(in_channels, 512, kernel_size=1, stride=1), nn.ReLU(),
                    nn.Conv2d(512, 1, kernel_size=1, stride=1)
                )
                
                # set the divergence (i.e pearson or jensen),pass the auxiliary head, set n_classes=-1 since we are ignoring class information in this example.
                self.fdalhead = fDALDivergenceHead(divergence_name='pearson', aux_head=aux_head, n_classes=-1, reg_coef=0.1,
                                                grl_params={'auto_step': True, 'hi': 1.0, 'max_iters': 10})

            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # print(len(features[0]))
        # print(features[0])
        n = len(features[0])
        half = int(n/2) 
        features_s, features_t = features[0][:half], features[0][n-half:]
        # features_s, features_t = features.chunk(2, dim=0)
        loss_fdal = self.fdalhead(features_s, features_t, None, None)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update({'fdal_loss': loss_fdal})
            return losses

        return result
