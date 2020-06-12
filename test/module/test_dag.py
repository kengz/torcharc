from torcharc import arc_ref, net_util
from torcharc.module import dag
import pydash as ps
import torch


def test_dag_forward():
    arc = arc_ref.REF_ARCS['forward']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)


def test_dag_concatmerge():
    arc = arc_ref.REF_ARCS['ConcatMerge']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)


def test_dag_filmmerge():
    arc = arc_ref.REF_ARCS['FiLMMerge']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)


def test_dag_reusefork():
    arc = arc_ref.REF_ARCS['ReuseFork']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_splitfork():
    arc = arc_ref.REF_ARCS['SplitFork']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_merge_fork():
    arc = arc_ref.REF_ARCS['merge_fork']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs._asdict())  # test dict input for tracing
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_fork_merge():
    arc = arc_ref.REF_ARCS['fork_merge']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)


def test_dag_reuse_fork_forward():
    arc = arc_ref.REF_ARCS['reuse_fork_forward']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_split_fork_forward():
    arc = arc_ref.REF_ARCS['split_fork_forward']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_merge_forward_split():
    arc = arc_ref.REF_ARCS['merge_forward_split']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs._asdict())  # test dict input for tracing
    ys = model(xs)
    assert ps.is_tuple(ys)


def test_dag_hydra():
    arc = arc_ref.REF_ARCS['hydra']
    in_shapes = arc['dag_in_shape']
    xs = net_util.get_rand_tensor(in_shapes)
    model = dag.DAGNet(arc)
    ys = model(xs._asdict())  # test dict input for tracing
    ys = model(xs)
    assert ps.is_tuple(ys)
