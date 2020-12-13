import torch
from torch import nn


# from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
# from second.pytorch.models.middle import register_middle
# from torchplus.nn import Empty
# from torchplus.tools import change_default_args
# import numpy as np

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape=[1, 1, 400, 400, 64],
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features):
        coords = torch.randn(8500, 4, device='cuda')
        batch_size = 1

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas


model = PointPillarsScatter()

vf = torch.randn(8500, 64, device='cuda')
cd = torch.randn(8500, 4, device='cuda')

cn = model(voxel_features=vf)
#
# file_name = "checkpoint.pth"
# torch.save(model.state_dict(), file_name)

state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

# torch.onnx.export(model, vf, "mayank.onnx")
