# from torcharc.module.merge import MergeConcat, MergeFiLM
# import pytest
# import torch


# @pytest.mark.parametrize('xs,out_shape', [
#     (
#         {'x0': torch.rand(4, 8), 'x1': torch.rand(4, 10)},
#         [4, 18],
#     ), (
#         {'x0': torch.rand(4, 8), 'x1': torch.rand(4, 10), 'x2': torch.rand(4, 12)},
#         [4, 30],
#     ),
# ])
# def test_concat_merge(xs, out_shape):
#     merge = ConcatMerge()
#     y = merge(xs)
#     assert y.shape == torch.Size(out_shape)


# @pytest.mark.parametrize('feature', [
#     torch.ones([1, 3]),  # vector
#     torch.ones([1, 3, 20]),  # time series
#     torch.ones([1, 3, 20, 20]),  # image
# ])
# def test_film_affine_transform(feature):
#     batch_size, n_feat, *_ = feature.shape
#     feature_dim = len(feature.shape)
#     conditioner_scale = torch.tensor([[1.0, 0.0, 1.0]])
#     conditioner_shift = torch.zeros([batch_size, n_feat])
#     y = MergeFiLM.affine_transform(feature, conditioner_scale, conditioner_shift)
#     if feature_dim > 2:
#         mean_y = y.mean(list(range(feature_dim))[2:])
#     else:
#         mean_y = y
#     assert torch.equal(mean_y, conditioner_scale)


# @pytest.mark.parametrize('names,shapes,xs', [
#     (  # vector
#         {'feature': 'vector_0', 'conditioner': 'vector_1'},
#         {'vector_0': [3], 'vector_1': [8]},
#         {'vector_0': torch.rand(4, 3), 'vector_1': torch.rand(4, 8)},
#     ), (  # time series
#         {'feature': 'ts', 'conditioner': 'vector'},
#         {'ts': [3, 20], 'vector': [8]},
#         {'ts': torch.rand(4, 3, 20), 'vector': torch.rand(4, 8)},
#     ), (  # image
#         {'feature': 'image', 'conditioner': 'vector'},
#         {'image': [3, 20, 20], 'vector': [8]},
#         {'image': torch.rand(4, 3, 20, 20), 'vector': torch.rand(4, 8)},
#     )
# ])
# def test_film_merge(names, shapes, xs):
#     merge = MergeFiLM(names, shapes)
#     y = merge(xs)
#     assert y.shape == xs[names['feature']].shape
