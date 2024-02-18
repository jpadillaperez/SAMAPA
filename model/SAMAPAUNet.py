import os
import wandb
import imageio
import numpy as np
import nibabel as nib
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import partial
from einops import rearrange
from typing import Any, Optional, Tuple
from utils.config import train_config, test_config
from model.sam.image_encoder import ImageEncoderViT
from model.sam.prompt_encoder import PromptEncoder
from model.sam.mask_decoder import MaskDecoder
from model.sam.transformer import TwoWayTransformer

from utils.wandb import save3DImagetoWandb, save2DImagetoWandb

class SAMAPAUNet(nn.Module):
    def __init__(self, partial_train = ["axial", "coronal", "sagittal"]):
        super(SAMAPAUNet, self).__init__()

        self.encoder = APAEncoder(partial_train=partial_train)
        self.partial_train = partial_train
        self.full_training = (partial_train == ["axial", "coronal", "sagittal"])

        if self.full_training:
            self.mlp_decoder = MLP(input_size=train_config["input_shape"][0] * train_config["input_shape"][1] * 3,
                                output_size=train_config["input_shape"][0] * train_config["input_shape"][1] * train_config["input_shape"][2])

    def forward(self, x):
        self.axial_out, self.coronal_out, self.sagittal_out = self.encoder(x)

        if self.full_training:
            input_mlp = torch.cat((h_attn, w_attn, d_attn), dim=1)
            out = self.mlp_decoder(input_mlp.view(input_mlp.size(0), -1).float())
            out = out.view(out.size(0), 1, train_config["input_shape"][0], train_config["input_shape"][1], train_config["input_shape"][2])
        else:
            if "axial" in self.partial_train:
                out = self.axial_out
            elif "coronal" in self.partial_train:
                out = self.coronal_out
            elif "sagittal" in self.partial_train:
                out = self.sagittal_out

        return out

#-----------------APA Encoder-----------------#
class APAEncoder(nn.Module):
    def __init__(self, partial_train = ["axial", "coronal", "sagittal"]):
        super(APAEncoder, self).__init__()
        self.partial_train = partial_train
        
        for proj in partial_train:
            setattr(self, proj + "_att", APA(projection=proj))
        
    def forward(self, input):
        self.axial_attn = None
        self.coronal_attn = None
        self.sagittal_attn = None

        for proj in self.partial_train:
            attn = getattr(self, proj + "_att")(input)
            setattr(self, proj + "_attn", attn)

        return self.axial_attn, self.coronal_attn, self.sagittal_attn

#-----------------Axial Attention Modules-----------------#
#TODO: The modules could be the same #DONE
#TODO: We could optimize the number of them by doing DRR on the fly

class APA(nn.Module):
    def __init__(self, projection):
        super(APA, self).__init__()
        self.projection = projection.lower()
        self.attention = InnerBlock(projection)

    def forward(self, input):
        self.drr = input["drr_" + self.projection]
        self.drr_label = input["drr_" + self.projection + "_label"]
        attn = self.attention(self.drr, self.drr_label)
        return attn

#----------------- Basic Blocks -----------------#

class InnerBlock(nn.Module):
    def __init__(self, projection):
        super(InnerBlock, self).__init__()
        self.projection = projection
        self.original_size = train_config["input_shape"]

        #------------------------ SAM Architecture ---------------------------
        self.image_size = 1024
        self.patch_size = 16
        self.encoder_embed_dim = 768
        self.encoder_num_heads = 12
        self.encoder_depth = 12
        self.image_embedding_size = self.image_size // self.patch_size
        self.prompt_embed_dim = 256
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)
        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)

        self.mask_threshold = 0.4

        self.image_encoder = ImageEncoderViT(
            args = None,
            depth= self.encoder_depth,
            embed_dim= self.encoder_embed_dim,
            img_size= self.image_size,
            mlp_ratio= 4,
            norm_layer= partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads= self.encoder_num_heads,
            patch_size= self.patch_size,
            qkv_bias= True,
            use_rel_pos= True,
            global_attn_indexes = [2, 5, 8, 11],
            window_size=14,
            out_chans = self.prompt_embed_dim,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim= self.prompt_embed_dim,
            image_embedding_size= [self.image_embedding_size, self.image_embedding_size],
            input_image_size= [self.image_size, self.image_size],
            mask_in_chans= 16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )


    def forward(self, drr, drr_label):
        #------------------------ SAM Architecture ---------------------------    
        drr_preprocessed = self.preprocess(drr.squeeze(0))
        drr_preprocessed = self.add_three_channels(drr_preprocessed).float()
        drr_label_preprocessed = self.preprocess(drr_label)

        image_embeddings = self.image_encoder(drr_preprocessed)
        points_input, labels_input = self.get_points(torch.zeros_like(drr_label_preprocessed, dtype=torch.float32, device=drr.device), drr_label_preprocessed, num_points=1)
        
        if train_config["debugging_mode"]["active"]:
            save2DImagetoWandb(drr_preprocessed[0, 0], f"Input_{self.projection}")
            save2DImagetoWandb(drr_label_preprocessed[0, 0], f"Target_{self.projection}")
            save2DImagetoWandb(image_embeddings[0, 0], f"ImageEmbedding_{self.projection}")

            #Create Input with points
            input_with_points = drr_preprocessed.clone()
            for point in points_input:
                #Set 5 pixels around the point to 1
                input_with_points[0, 0, int(point[0])-2:int(point[0])+3, int(point[1])-2:int(point[1])+3] = 1
            save2DImagetoWandb(input_with_points[0, 0], f"InputWithPoints_{self.projection}")

        batched_input = {
            "image": drr_preprocessed,
            "original_size": [self.original_size[0], self.original_size[1]],
            "point_coords": points_input,
            "point_labels": labels_input,
        }

        #------------------------ Iteration Prompt and Mask ---------------------------
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            if train_config["debugging_mode"]["active"]:
                save2DImagetoWandb(low_res_masks[0, 0], f"LowResMask_{self.projection}")

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(self.image_encoder.img_size, self.image_encoder.img_size),
                original_size=(self.original_size[0], self.original_size[1]),
            )

            if train_config["debugging_mode"]["active"]:
                save2DImagetoWandb(masks[0, 0], f"Mask_{self.projection}")

            masks = masks.float()
        return masks


    #------------------------ Helper Functions ---------------------------


    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolates to a square input """
        x = F.interpolate(x, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False)
        return x

    def add_three_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Add three channels to the input."""
        return x.repeat(1, 3, 1, 1)


    def get_points(self, prev_masks, ground_truth, mask_threshold= 0.4, depth_threshold=0, num_points=1):
        """
        Get the points and labels for the next click.
        
        Arguments:
        prev_masks (torch.Tensor): The predicted masks from the previous
            click, in BxHxW format.
        ground_truth (torch.Tensor): The ground truth masks, in BxHxW format.
        mask_threshold (float): The threshold for the predicted masks.
        depth_threshold (float): The threshold for the ground truth masks.

        Returns:
        (torch.Tensor, torch.Tensor): The point and label for the next click.
        """
        
        batch_points = []
        batch_labels = []

        if ground_truth.ndim == 4:
            ground_truth = ground_truth.squeeze(0).squeeze(0)
        if prev_masks.ndim == 4:
            prev_masks = prev_masks.squeeze(0).squeeze(0)

        # Apply value threshold to the predicted masks
        prev_masks = (prev_masks > mask_threshold)

        # Apply depth threshold to the ground truth masks
        true_masks = (ground_truth > depth_threshold)

        # Get the false positives and false negatives
        fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_masks))

        if len(torch.where(fn_masks)[0]) > 0:
            points_source = torch.where(fn_masks)
        else:
            points_source = torch.where(true_masks)

        for i in range(num_points):
            idx = np.random.choice(len(points_source[0]))
            point_input = torch.tensor([points_source[0][idx], points_source[1][idx]], dtype=torch.float32, device=prev_masks.device)
            label_input = torch.tensor([1], dtype=torch.int64, device=prev_masks.device)
            batch_points.append(point_input)
            batch_labels.append(label_input)

        batch_points = torch.stack(batch_points, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)

        return batch_points, batch_labels


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # Define the MLP layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.layers(x)
