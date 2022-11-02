from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.testing import dump_ast
import tvm.script
import _gradient
from utils import LowerToTensorIRPass


@tvm.script.ir_module
class MultiLayerPerceptron:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32"),
             label: Tensor((1,10), "float32")):

        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.nn.matmul(x, w0)
            lv1 = relax.add(lv0, b0)
            # relu0
            lv2 = relax.nn.relu(lv1)
            # linear1
            lv3 = relax.nn.matmul(lv2, w1)
            out = relax.add(lv3, b1)
            loss = relax.nn.softmax_cross_entropy(out, label)
            R.output(loss)
        return loss
    

    @R.function
    def main_pred(x: Tensor((1, 784), "float32"),
             w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32")):

        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.nn.matmul(x, w0)
            lv1 = relax.add(lv0, b0)
            # relu0
            lv2 = relax.nn.relu(lv1)
            # linear1
            lv3 = relax.nn.matmul(lv2, w1)
            out = relax.add(lv3, b1)
            R.output(out)
        return out


AutoDiffMLP = relax.transform.SimpleAD(MultiLayerPerceptron.get_global_var("main"), require_grads=[1, 2, 3, 4])(MultiLayerPerceptron)
TIRMod = LowerToTensorIRPass()(AutoDiffMLP)
ex = relax.vm.build(TIRMod, target="llvm")
ex.mod.export_library("mlp_mod.so")