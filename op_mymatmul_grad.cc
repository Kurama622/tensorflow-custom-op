#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("MymatmulGrad")
  .Attr("T: {float, int32, int64, double}")
  .Input("grad: T")
  .Input("input1: T")
  .Input("input2: T")
  .Output("grad_input1: T")
  .Output("grad_input2: T");


template<typename T>
class MymatmulGradOp : public OpKernel {
public:
  explicit MymatmulGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // create input tensor
    const Tensor& grad_tensor   = context->input(0);
    const Tensor& input_tensor1 = context->input(1);
    const Tensor& input_tensor2 = context->input(2);
    const TensorShape& grad_shape   = grad.shape();
    const TensorShape& input_shape1 = input_tensor1.shape();
    const TensorShape& input_shape2 = input_tensor2.shape();

    // create output tensor
    Tensor* grad_input_tensor1 = NULL;
    Tensor* grad_input_tensor2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape1, &grad_input_tensor1));
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape2, &grad_input_tensor2));

    auto grad   = grad_tensor.matrix<T>();
    auto input1 = input_tensor1.matrix<T>();
    auto input2 = input_tensor2.matrix<T>();
    auto grad_input1 = grad_input_tensor1->matrix<T>();
    auto grad_input2 = grad_input_tensor2->matrix<T>();

    int K = input_shape1.dim_size(1);
    int N = input_shape1.dim_size(0);
    int M = input_shape2.dim_size(1);

    // init
    for(int j = 0; j < M; j++) {
      for(int i = 0; i < K; i++) {
        grad_input1(i, j) = 0.0;
      }
    }

    for(int j = 0; j < K; j++) {
      for(int i = 0; i < N; i++) {
        grad_input2(i, j) = 0.0;
      }
    }

    // matmul
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        for(int k = 0; k < K; k++) {
          grad_input1(i, k) += input2(k, j) * grad(i, j);
          grad_input2(k, j) += input1(i, k) * grad(i, j);
        }
      }
    }
  }
};

#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                        \
    Name("MymatmulGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),\
    MymatmulGradOp<type>)

  REGISTER_KERNEL(int32)
  REGISTER_KERNEL(int64)
  REGISTER_KERNEL(float)
  REGISTER_KERNEL(double)
