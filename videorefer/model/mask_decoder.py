# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for GROVE: Grounded Video Caption Generation
from typing import List, Tuple, Type
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer
from einops import rearrange

# Reuse LayerNorm2d from timm (VideoRefer already uses timm)
try:
    from timm.models.layers import LayerNorm2d
except Exception:
    # Fallback: simple LayerNorm2d if timm not available
    class LayerNorm2d(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm = nn.LayerNorm(channels)

        def forward(self, x):
            # expect [b, c, h, w] -> normalize over channel dim
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            return x.permute(0, 3, 1, 2).contiguous()


class MaskDecoder(nn.Module):
    """
    Decoder that predicts either segmentation masks (original SAM mode) 
    or bounding boxes (GROVE query mode) given image and prompt embeddings.
    
    Key innovation from GROVE paper:
    - Uses detection token embeddings as queries (from LLM)
    - Visual features as keys/values
    - Frame-wise cross-attention for temporal consistency
    """

    def __init__(
        self,
        *,
        transformer_dim: int = 256,  # Channel dimension of transformer
        transformer: nn.Module = None,  # SAM's TwoWayTransformer (can be None initially)
        num_multimask_outputs: int = 3,  # Number of mask outputs (SAM legacy)
        activation: Type[nn.Module] = nn.GELU,  # Activation for upscaling
        iou_head_depth: int = 3,  # Depth of IoU prediction MLP
        iou_head_hidden_dim: int = 256,  # Hidden dim for IoU head
        decoding_type: str = "query",  # "query" = bbox mode, "mask" = SAM mode
        use_temp_objectness: bool = True,  # Enable temporal objectness prediction
    ) -> None:
        super().__init__()

        # Store core configuration
        self.transformer_dim = transformer_dim
        self.transformer = transformer  # Will be set later if None
        self.num_multimask_outputs = num_multimask_outputs

        # ============================================================
        # LEARNED TOKENS (from original SAM architecture)
        # ============================================================
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # ============================================================
        # MASK PREDICTION COMPONENTS (original SAM, not used in GROVE)
        # ============================================================
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # ============================================================
        # BOUNDING BOX PREDICTION COMPONENTS (GROVE additions)
        # ============================================================
        self.decoding_type = decoding_type

        if decoding_type == "query":
            # Bounding box head: 2-layer MLP predicting normalized coordinates
            self.bbox_prediction_head = nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, 4)  # (x_min, y_min, x_max, y_max)
            )

            if use_temp_objectness:
                # Temporal objectness head: predicts visibility per frame
                # Removed device='cpu' - respect module device
                self.temporal_objectness_head = nn.Linear(transformer_dim, 1)
                self.use_temp_objectness = True
            else:
                self.use_temp_objectness = False

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [T, C, H, W]
        image_pe: torch.Tensor,  # [T, C, H, W]
        sparse_prompt_embeddings: torch.Tensor,  # [N_d_total, 1, D]
        dense_prompt_embeddings: torch.Tensor,  # [T, C, H, W]
        multimask_output: bool,
        reps: List[int],  # [N_d^0, N_d^1, ..., N_d^(T-1)]
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        """
        Forward pass for bounding box prediction.
        
        Args:
            image_embeddings: Visual features from encoder [T, C, H, W]
            image_pe: Positional encodings [T, C, H, W]
            sparse_prompt_embeddings: Detection token embeddings [N_d_total, 1, D]
            dense_prompt_embeddings: Dense prompts (usually zeros) [T, C, H, W]
            multimask_output: Not used in query mode
            reps: Number of detection tokens per frame
        
        Returns:
            bbox_pred: Bounding box predictions [sum(reps), 4]
            temp_objectness_logits: Visibility scores [sum(reps)]
        """


        if self.transformer is None:
            raise RuntimeError(
                "MaskDecoder.transformer is None. "
                "Set self.transformer to a TwoWayTransformer instance before forward()."
            )

        # Concatenate output tokens (SAM legacy)
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        # Combine with detection token embeddings
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Frame-wise batching strategy
        # Create indices to replicate frames according to number of objects per frame
        indices = torch.repeat_interleave(
            torch.arange(image_embeddings.size(0)), torch.tensor(reps)
        )
        indices = indices.to(image_embeddings.device)

        # Expand image embeddings, dense prompts, and positional encodings using same indices
        src = torch.index_select(image_embeddings, 0, indices)
        dense_prompts_expanded = torch.index_select(dense_prompt_embeddings, 0, indices)
        src = src + dense_prompts_expanded
        
        # FIX: Also expand positional encodings using same indices
        pos_src = torch.index_select(image_pe, 0, indices)

        b, c, h, w = src.shape

        # Run SAM's two-way transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # GROVE query mode: predict bounding boxes
        if self.decoding_type == "query":
            # Extract detection token outputs
            query_out = hs[:, (1 + self.num_mask_tokens) :, :]
            
            # Predict bounding boxes
            bbox_pred = torch.sigmoid(self.bbox_prediction_head(query_out))
            bbox_pred = bbox_pred.squeeze(1)
            
            if self.use_temp_objectness:
                # Predict temporal objectness
                temp_objectness_logits = self.temporal_objectness_head(query_out)
                temp_objectness_logits = temp_objectness_logits.squeeze((1, 2))
                return bbox_pred, temp_objectness_logits
            else:
                return bbox_pred
        
        # Original SAM mask prediction mode
        else:
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(src)
            
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            hyper_in = torch.stack(hyper_in_list, dim=1)
            
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
                b, self.num_mask_tokens, h, w
            )
            
            iou_pred = self.iou_prediction_head(iou_token_out)
            return masks, iou_pred

class MLP(nn.Module):

    """
    Simple multi-layer perceptron with ReLU activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
