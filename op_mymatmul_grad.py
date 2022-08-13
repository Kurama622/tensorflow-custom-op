import tensorflow as tf
from tensorflow.python.framework import ops
m = tf.load_op_library('./op_mymatmul_grad.so')

@ops.RegisterGradient("Mymatmul")

def mymatmul_grad_cc(op, grad):
    return m.mymatmul_grad(grad, op.inputs[0], op.inputs[1])
