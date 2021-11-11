from torch import nn


class ProjectionPostprocessor(nn.Module):
    '''Postprocessing to use a linear layer to project PerceiverDecoder output's last dimension, i.e. shape (O, E) -> (O, proj_dim).'''

    def __init__(self, output_shape: tuple, proj_dim: int):
        super().__init__()
        output_o, output_e = output_shape  # PerceiverDecoder output shape (O, E)
        self.proj_dim = proj_dim
        self.linear = nn.Linear(output_e, self.proj_dim)

    def forward(self, output):
        return self.linear(output)


class ClassificationPostprocessor(ProjectionPostprocessor):
    '''Classification postprocessing for Perceiver. This is simply projecting decoder output to classes, i.e. shape (O, E) -> (num_classes)'''

    def __init__(self, output_shape: tuple, num_classes: int):
        assert output_shape[0] == 1, 'output_shape[0] must be 1 for classification'
        super().__init__(output_shape, num_classes)

    def forward(self, output):
        return super().forward(output).squeeze(dim=1)  # output logits

# TODO embedding postprocessor
# TODO multimodal postprocessor by partitioning output_shape[0] to each postprocessor
