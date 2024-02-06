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
    def __init__(self, num_classes=1):
        super(SAMAPAUNet, self).__init__()

        self.encoder = APAEncoder()

        self.mlp_decoder = MLP(input_size=train_config["input_shape"][0] * train_config["input_shape"][1] +
                                        train_config["input_shape"][0] * train_config["input_shape"][2] +
                                        train_config["input_shape"][1] * train_config["input_shape"][2],

                                output_size=train_config["input_shape"][0] * train_config["input_shape"][1] * train_config["input_shape"][2])
                                
        self.sigmoid = torch.nn.Sigmoid()

        #override train and eval methods to keep weights of image encoder frozen
        #self.train = partial(self.train, freeze_encoder=False)
        #self.eval = partial(self.eval, freeze_encoder=True)

    def forward(self, x):
        if train_config["debugging_mode"]["active"]:
            save3DImagetoWandb(x["image"][0,0,:,:,:], "input_image")
            save3DImagetoWandb(x["masks"][0,0,:,:,:], "input_mask")
            save2DImagetoWandb(x["drr_axial"], "input_drr_axial")
            save2DImagetoWandb(x["drr_coronal"], "input_drr_coronal")
            save2DImagetoWandb(x["drr_sagittal"], "input_drr_sagittal")

        h_attn, w_attn, d_attn = self.encoder(x)

        if train_config["debugging_mode"]["active"]:
            save2DImagetoWandb(h_attn[0,0,:,:], "h_attn")
            save2DImagetoWandb(w_attn[0,0,:,:], "w_attn")
            save2DImagetoWandb(d_attn[0,0,:,:], "d_attn")

        out = self.mlp_decoder(torch.cat((h_attn, w_attn, d_attn), dim=1))
        out = self.sigmoid(out)

        if train_config["debugging_mode"]["active"]:
            save3DImagetoWandb(out[0,0,:,:,:], "output_masks")

        return out


#-----------------APA Encoder-----------------#
class APAEncoder(nn.Module):
    def __init__(self):
        super(APAEncoder, self).__init__()
        
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)

        self.axial_att = APA("axial")
        self.coronal_att = APA("coronal")
        self.sagittal_att = APA("sagittal")
        
    def forward(self, input):
        input["image"] = input["image"].repeat(1,3,1,1,1)
        input["masks"] = input["masks"].repeat(1,3,1,1,1)

        axial_attn = self.axial_att(input)
        coronal_attn = self.coronal_att(input)
        sagittal_attn = self.sagittal_att(input)

        return axial_attn, coronal_attn, sagittal_att

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
        self.checkpoint = train_config["SAM_checkpoint"]

        #------------------------ SAM Architecture ---------------------------
        self.input_size = 64 
        #TODO: include this number in the config file

        self.patch_size = 16
        self.encoder_embed_dim = 768
        self.image_embedding_size = self.input_size // self.patch_size

        self.prompt_embed_dim = 256
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)
        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)

        #TODO: calculate mean and std of the dataset (or from config file)
        self.pixel_mean = 123.675
        self.pixel_std = 58.395
        self.pixel_mean = torch.tensor(self.pixel_mean).to("cuda:0")
        self.pixel_std = torch.tensor(self.pixel_std).to("cuda:0")


        self.image_encoder = ImageEncoderViT(
            args = None,
            depth= 12,
            embed_dim= self.encoder_embed_dim,
            img_size= self.input_size,
            mlp_ratio= 4,
            norm_layer= partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads= 12,
            patch_size= self.patch_size,
            qkv_bias= True,
            use_rel_pos= True,
            global_attn_indexes = [2, 5, 8, 11],
            window_size=14,
            out_chans = 256,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim= self.prompt_embed_dim,
            image_embedding_size= [self.image_embedding_size, self.image_embedding_size],
            input_image_size= [self.input_size, self.input_size],
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

        with open(self.checkpoint, "rb") as f:
            state_dict = torch.load(f)

        #------------------------ Load Weights ---------------------------
        ##load all the weights
        self.image_encoder.load_state_dict(state_dict, strict=False)  
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        for name, param in self.image_encoder.named_parameters():
            if name.find('MLP_Adapter') != -1:
                param.requires_grad = True
                #print('---------Unfreezing layer: ', name)
            elif name.find('Space_Adapter') != -1:
                param.requires_grad = True
                #print('---------Unfreezing layer: ', name)
        self.prompt_encoder.load_state_dict(state_dict, strict=False)
        self.mask_decoder.load_state_dict(state_dict, strict=False)

        #------------------------ Extras ---------------------------
        self.num_clicks = 2 
        #TODO: include this number in the config file


    def forward(self, drr, drr_label):
        #------------------------ SAM Architecture ---------------------------
        drr_preprocessed = torch.stack([self.preprocess(x) for x in drr], dim=0)    
        image_embeddings = self.image_encoder(drr_preprocessed)

        #------------------------ Iteration Prompt and Mask ---------------------------
        #to create a 3D Mask with every click
        for num_click in range(self.num_clicks):
            points_input, labels_input = self.get_points(prev_masks, drr_label, self.projection)
            if num_click == self.num_clicks - 1:
                #One last forward pass to get the final mask
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=low_res_masks,
                        )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                    
            else:
                # Forward pass after adding the click
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,
                    masks=low_res_masks,
                    )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

            prev_masks = F.interpolate(low_res_masks, size=(self.image_encoder.img_size, self.image_encoder.img_size), mode='trilinear', align_corners=False)

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=[self.original_size[0], self.original_size[1]],
            original_size=[self.original_size[0], self.original_size[1]],
        )

        #TODO: add here the 3d loss function?

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
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def get_points(self, prev_masks, ground_truth, projection, mask_threshold= 0.4, depth_threshold=0.6):
        batch_points = []
        batch_labels = []

        index = 0 if projection == "axial" else 1 if projection == "coronal" else 2

        #Create the depth map out of the previous prediction
        pred_masks = torch.zeros_like(ground_truth[0,0,:,:])
        for i in reversed(range(prev_masks.shape[index])):
            slice_ = prev_masks[i]
            depth_value = 1 - i / (prev_masks.shape[index] - 1)

            # The value of the depth map of prev_masks is equal to the ground_truth at best and smaller than the ground_truth at worst
            nonzero_indexes = torch.nonzero(slice_ > mask_threshold)
            pred_masks[nonzero_indexes] = depth_value * slice_[nonzero_indexes]
        pred_masks = pred_masks.flip(0)

        #TODO: temporal, fix in dataloader nifti images
        if projection == "sagittal":
            pred_masks = pred_masks.flip(1)

        pred_masks = (pred_masks > depth_threshold)
        true_masks = (ground_truth > depth_threshold)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fn_masks_loc = torch.argwhere(fn_masks)
        point_input = fn_masks_loc[torch.randint(len(fn_masks_loc))]
        label_input = 1 # Because it is a false negative

        #TODO: Not ready for batch > 1

        return point_input, label_input


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class MixConv(nn. Module):
    def __init__(self, inp, oup):
        super(MixConv, self).__init__()

        self.groups = oup // 4
        in_channel = inp // 4
        out_channel = oup // 4

        self.dwconv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.dwconv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2)
        self.dwconv3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        self.dwconv4 = nn.Conv2d(in_channel, out_channel, 9, padding=4)

        self.pwconv = nn.Sequential(
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, 1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a, b, c, d = torch.split(x, self.groups, dim=1)
        a = self.dwconv1(a)
        b = self.dwconv1(b)
        c = self.dwconv1(c)
        d = self.dwconv1(d)

        out = torch.cat([a, b, c, d], dim=1)
        out = self.pwconv(out)

        return out

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
    
class doubelconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(doubelconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

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
