#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("Mymatmul")
  .Attr("T: {float, int32, int64, double}")
  .Input("matrix1: T")
  .Input("matrix2: T")
  .Output("matmuled: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
                auto N = c->Dim(c->input(0), 0);
                auto M = c->Dim(c->input(1), 1);
                c->set_output(0, c->MakeShape({N,M}));
                return Status::OK();
              });

template<typename T>
class MymatmulOp : public OpKernel {
public:
  explicit MymatmulOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // create input tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const TensorShape& input1_shape = input_tensor1.shape();
    const TensorShape& input2_shape = input_tensor2.shape();

    // create output tensor
    TensorShape output_shape;
    const int N = input1_shape.dim_size(0);
    const int M = input2_shape.dim_size(1);
    output_shape.AddDim(N);
    output_shape.AddDim(M);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));


    auto input1 = input_tensor1.matrix<T>();
    auto input2 = input_tensor2.matrix<T>();
    auto output = output_tensor->template matrix<T>();

    // matmul
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        output(i,j) = 0;
        for(int k = 0; k < input1_shape.dim_size(1); k++) {
          output(i,j) += input1(i, k) * input2(k, j);
        }
      }
    }
  }
};

#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                        \
    Name("Mymatmul").Device(DEVICE_CPU).TypeConstraint<type>("T"),\
    MymatmulOp<type>)

  REGISTER_KERNEL(int32)
  REGISTER_KERNEL(int64)
  REGISTER_KERNEL(float)
  REGISTER_KERNEL(double)
