#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <tuple>
#include <limits>

namespace at {
namespace native {
namespace {

template <typename scalar_t, 
         int kKnownKernelT, int kKnownKernelH, int kKnownKernelW,
         int kKnownDilationT, int kKnownDilationH, int kKnownDilationW>
__global__ void conv_depthwise3d_cuda_kernel(
    const PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    const PackedTensorAccessor64<scalar_t, 5> kernel,
    const scalar_t* bias,
    int dT, int dH, int dW,
    int pT, int pH, int pW,
    int dilationT_, int dilationH_, int dilationW_)
{
  const int kT = kKnownKernelT > 0 ? kKnownKernelT : (int)kernel.size(2);
  const int kH = kKnownKernelH > 0 ? kKnownKernelH : (int)kernel.size(3);
  const int kW = kKnownKernelW > 0 ? kKnownKernelW : (int)kernel.size(4);
  const int oC = output.size(1);
  const int oT = output.size(2);
  const int oH = output.size(3);
  const int oW = output.size(4);
  const int iC = input.size(1);
  const int iT = input.size(2);
  const int iH = input.size(3);
  const int iW = input.size(4);
  const int channel_multiplier = oC / iC;
  const int dilationT = kKnownDilationT > 0 ? kKnownDilationT : dilationT_;
  const int dilationH = kKnownDilationH > 0 ? kKnownDilationH : dilationH_;
  const int dilationW = kKnownDilationW > 0 ? kKnownDilationW : dilationW_;
  const int64_t num_output = (int64_t)output.size(0) * oC * oT * oH * oW;

  CUDA_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int batch = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;
    
    const int in_col_start = out_col * dW - pW;
    const int in_row_start = out_row * dH - pH;
    const int in_frame_start = out_frame * dT - pT;

    scalar_t sum = (scalar_t)0;
    const scalar_t *kernel_ptr = kernel[out_channel].data();
    const scalar_t *input_ptr = 
      &input[batch][in_channel][in_frame_start][in_row_start][in_col_start];
    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const int in_col = in_col_start + k_col * dilationW;

          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            sum += __ldg(input_ptr) * __ldg(kernel_ptr);
          }
          kernel_ptr ++;
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW;
      }
      input_ptr += (iH * dilationT - kH) * iW;
    }
    if (bias != NULL) {
      sum += bias[out_channel];
    }

    output[batch][out_channel][out_frame][out_row][out_col] = sum;
  }
}

template <typename scalar_t, 
         int kKnownKernelT, int kKnownKernelH, int kKnownKernelW,
         int kKnownDilationT, int kKnownDilationH, int kKnownDilationW,
         int kKnownStrideT, int kKnownStrideH, int kKnownStrideW,
         int kKnownChannelMultiplier>
__global__ void conv_depthwise3d_cuda_backward_input_kernel(
    const PackedTensorAccessor64<scalar_t, 5> grad_output,
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    const PackedTensorAccessor64<scalar_t, 5> kernel,
    int dT_, int dH_, int dW_,
    int pT, int pH, int pW, 
    int dilationT_, int dilationH_, int dilationW_) {
  const int kT = kKnownKernelT > 0 ? kKnownKernelT : kernel.size(2);
  const int kH = kKnownKernelH > 0 ? kKnownKernelH : kernel.size(3);
  const int kW = kKnownKernelW > 0 ? kKnownKernelW : kernel.size(4);
  const int oC = grad_output.size(1);
  const int oT = grad_output.size(2);
  const int oH = grad_output.size(3);
  const int oW = grad_output.size(4);
  const int iC = grad_input.size(1);
  const int iT = grad_input.size(2);
  const int iH = grad_input.size(3);
  const int iW = grad_input.size(4);
  const int channel_multiplier = oC / iC;
  const int dilationT = kKnownDilationT > 0 ? kKnownDilationT : dilationT_;
  const int dilationH = kKnownDilationH > 0 ? kKnownDilationH : dilationH_;
  const int dilationW = kKnownDilationW > 0 ? kKnownDilationW : dilationW_;
  const int dT = kKnownStrideT > 0 ? kKnownStrideT : dT_;
  const int dH = kKnownStrideH > 0 ? kKnownStrideH : dH_;
  const int dW = kKnownStrideW > 0 ? kKnownStrideW : dW_;
  const int64_t num_input = grad_input.size(0) * grad_input.stride(0);

  CUDA_KERNEL_LOOP(index, num_input) {
    const int in_col = index % iW;
    const int in_row = (index / iW) % iH;
    const int in_frame = (index / iW / iH) % iT;
    const int in_channel = (index / iW / iH / iT) % iC;
    const int batch = index / iW / iH / iT / iC;

    const int out_col_end = in_col + pW;
    const int out_row_end = in_row + pH;
    const int out_frame_end = in_frame + pT;

    const scalar_t* kernel_ptr = kernel[in_channel * channel_multiplier].data();
    scalar_t sum = (scalar_t)0;

    for (int k_chn = in_channel * channel_multiplier;
        k_chn < (in_channel + 1) * channel_multiplier;
        ++k_chn) {
      for (int k_frame = 0; k_frame < kT; ++k_frame) {
        const int out_frame = out_frame_end - k_frame * dilationT;
        for (int k_row = 0; k_row < kH; ++k_row) {
          const int out_row = out_row_end - k_row * dilationH;
          for (int k_col = 0; k_col < kW; ++k_col) {
            const int out_col = out_col_end - k_col * dilationW;
            if (out_frame >= 0 && out_row >= 0 && out_col >= 0 &&
                out_frame < oT * dT && out_row < oH * dH && out_col < oW * dW &&
                out_frame % dT == 0 && out_row % dH == 0 && out_col % dW == 0) {
              sum += __ldg(kernel_ptr) * 
                __ldg(&grad_output[batch][k_chn]
                    [out_frame / dT][out_row / dH][out_col / dW]);
            }
            ++kernel_ptr;
          }
        }
      }
    }

    grad_input[batch][in_channel][in_frame][in_row][in_col] = sum;
  }
}

template <typename scalar_t>
__global__ void conv_depthwise3d_cuda_backward_weight_kernel(
    const PackedTensorAccessor64<scalar_t, 5> grad_output,
    const PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> grad_kernel,
    int dT, int dH, int dW,
    int pT, int pH, int pW, 
    int dilationT, int dilationH, int dilationW) {
  const int kC = grad_kernel.size(0);
  const int kT = grad_kernel.size(2);
  const int kH = grad_kernel.size(3);
  const int kW = grad_kernel.size(4);

  const int k_col = blockIdx.x % kW;
  const int k_row = (blockIdx.x / kW) % kH;
  const int k_frame = (blockIdx.x / kW / kH) % kT;
  const int k_channel = blockIdx.x / kW / kH / kT;

  const int oT = grad_output.size(2);
  const int oH = grad_output.size(3);
  const int oW = grad_output.size(4);
  const int64_t oTHW = grad_output.stride(1);
  const int iT = input.size(2);
  const int iH = input.size(3);
  const int iW = input.size(4);
  const int channel_multiplier = grad_output.size(1) / input.size(1);
  const int in_channel = k_channel / channel_multiplier;

  extern __shared__ int sdata_raw[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);

  if (k_channel >= kC) {
    return;
  }

  const int laneid = threadIdx.x % C10_WARP_SIZE;
  const int warpid = threadIdx.x / C10_WARP_SIZE;
  const int nwarps = blockDim.x / C10_WARP_SIZE;

  scalar_t grad = (scalar_t)0;
  for (int batch = warpid; batch < input.size(0); batch += nwarps) {
    const scalar_t* gout_ptr = grad_output[batch][k_channel].data() + laneid;

    for (int64_t pos = laneid; pos < oTHW; pos += C10_WARP_SIZE) {
      const int gout_col = pos % oW;
      const int gout_row = (pos / oW) % oH;
      const int gout_frame = pos / oW / oH;

      const int in_col = (gout_col * dW) + (k_col * dilationW) - pW;
      const int in_row = (gout_row * dH) + (k_row * dilationH) - pH;
      const int in_frame = (gout_frame * dT) + (k_frame * dilationT) - pT;
      
      if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
          in_frame < iT && in_row < iH && in_col < iW) {
        grad += __ldg(gout_ptr) * 
          __ldg(&input[batch][in_channel][in_frame][in_row][in_col]);
      }
      gout_ptr += C10_WARP_SIZE;
    }
  }
  
  sdata[threadIdx.x] = grad;
  __syncthreads();
 
  assert(__popc(blockDim.x) == 1);
#pragma unroll
  for (int i = blockDim.x / 2; i >= 1; i >>= 1) {
    if (threadIdx.x < i) {
      sdata[threadIdx.x] += sdata[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    grad_kernel[k_channel][0][k_frame][k_row][k_col] = sdata[0];
  }
}

template <int dim>
std::vector<int64_t> get_output_size(
    const Tensor &input,
    const Tensor &weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  std::vector<int64_t> output_size;
  
  if (input.dim() == dim + 2 /* is_batch */) {
    output_size.push_back(input.size(0));
  }
  output_size.push_back(weight.size(0));
  for (int i = 0; i < dim; ++i) {
    int64_t dim_ksize = weight.size(i + 2);
    int64_t dim_isize = input.size(-dim + i);
    int64_t dim_stride = stride[i];
    int64_t dim_padding = padding[i];
    int64_t dim_dilation = dilation[i];

    int64_t dim_kspan = (dim_ksize - 1) * dim_dilation + 1;
    output_size.push_back(
        (dim_isize + dim_padding * 2 - dim_kspan + 1 - 1) / dim_stride + 1);
  }

  return output_size;
}

template <int dim>
void conv_depthwise_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(kernel_size.size() == dim,
      "kernel size length should be ", dim, ", but got ", kernel_size.size());
  TORCH_CHECK(stride.size() == dim,
      "stride length should be ", dim, ", but got ", stride.size());
  TORCH_CHECK(padding.size() == dim,
      "padding length should be ", dim, ", but got ", padding.size());
  TORCH_CHECK(dilation.size() == dim,
      "dilation length should be ", dim, ", but got ", dilation.size());

  TORCH_CHECK(input.defined(),
      "Input must be defined.");
  TORCH_CHECK(weight.defined(),
      "Weight must be defined.");
  TORCH_CHECK(input.dim() == dim + 1 || input.dim() == dim + 2,
      "Input dimension should be ", 
      dim + 1, "D or ", dim + 2, "D, got ",
      input.dim(), "D");
  TORCH_CHECK(weight.dim() == dim + 2,
      "Weight dimension should be ", dim + 2, "D, got ", weight.dim(), "D");
  TORCH_CHECK(weight.size(1) == 1,
      "Depthwise weight should have in_channels=1, got ", weight.size(1));
  TORCH_CHECK(weight.size(0) % input.size(-dim - 1) == 0,
      "Depthwise out channels should be a multiple of in channels, got ",
      weight.size(0), " and ", input.size(-dim - 1));
  for (int i = 0; i < dim; ++i) {
    TORCH_CHECK(weight.size(i + 2) == kernel_size[i],
        "kernel size and weight size mismatch, got ",
        kernel_size, " and ", weight.sizes());
    TORCH_CHECK(stride[i] >= 1, 
        "stride should be at least 1, got ", stride);
    TORCH_CHECK(padding[i] >= 0,
        "padding should be non-negative, got ", padding);
    TORCH_CHECK(dilation[i] >= 1,
        "dilation should be at least 1, got ", dilation);
  }

  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1,
        "Bias should be 1D tensor, got ", bias.dim(), "D");
    TORCH_CHECK(bias.size(0) == weight.size(0),
        "Bias length should be equal to out_channels, got ",
        bias.size(0), " and ", weight.size(0));
  }

  if (grad_output.defined()) {
    auto expected_output_size = get_output_size<dim>(input, weight,
        stride, padding, dilation);
    TORCH_CHECK(grad_output.dim() == expected_output_size.size(),
        "Expect grad_output to be ",
        expected_output_size.size(), "D, got ",
        grad_output.dim(), "D.");
    for (int i = 0; i < grad_output.dim(); ++i) {
      TORCH_CHECK(grad_output.size(i) == expected_output_size[i],
          "Expect grad_output to be of same shape as output, got ",
          grad_output.size(i), " and ", expected_output_size[i],
          " at dimension ", i);
    }
  }

  // We also want to check every single dimension is within (0, 2^31) because
  // our kernels use 32-bit signed for indices.
#define CHECK_TENSOR_DIM_WITHIN_INT32_RANGE(x) \
  for (int i = 0; i < (x).dim(); ++i) {                                       \
    TORCH_CHECK(                                                              \
        (x).size(i) <= std::numeric_limits<int32_t>::max() && (x).size(i) > 0,\
        #x, " dimension ", i, " should be no greater than max integer, got ", \
        (x).size(i)); \
  }
  CHECK_TENSOR_DIM_WITHIN_INT32_RANGE(input);
  CHECK_TENSOR_DIM_WITHIN_INT32_RANGE(weight);
  if (bias.defined()) {
    CHECK_TENSOR_DIM_WITHIN_INT32_RANGE(bias);
  }
  if (grad_output.defined()) {
    CHECK_TENSOR_DIM_WITHIN_INT32_RANGE(grad_output);
  }
}

}

#define DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(kt, kh, kw, dilt, dilh, dilw) \
  if (kernel_size[0] == (kt) && dilation[0] == (dilt) &&        \
    kernel_size[1] == (kh) && dilation[1] == (dilh) &&          \
    kernel_size[2] == (kw) && dilation[2] == (dilw)) {          \
    conv_depthwise3d_cuda_kernel                                \
    <scalar_t, (kt), (kh), (kw), (dilt), (dilh), (dilw)>        \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(   \
        input_.packed_accessor64<scalar_t, 5>(),                \
        output_.packed_accessor64<scalar_t, 5>(),               \
        weight_.packed_accessor64<scalar_t, 5>(),               \
        bias_ptr,                                               \
        stride[0], stride[1], stride[2],                        \
        padding[0], padding[1], padding[2],                     \
        dilation[0], dilation[1], dilation[2]);                 \
  } else

#define DWCONV3D_FORWARD_DISPATCH_OTHERS \
  {                                                             \
    conv_depthwise3d_cuda_kernel                                \
    <scalar_t, -1, -1, -1, -1, -1, -1>                          \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(   \
        input_.packed_accessor64<scalar_t, 5>(),                \
        output_.packed_accessor64<scalar_t, 5>(),               \
        weight_.packed_accessor64<scalar_t, 5>(),               \
        bias_ptr,                                               \
        stride[0], stride[1], stride[2],                        \
        padding[0], padding[1], padding[2],                     \
        dilation[0], dilation[1], dilation[2]);                 \
  }

Tensor conv_depthwise3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride, 
    IntArrayRef padding,
    IntArrayRef dilation) { 
  conv_depthwise_shape_check<3>(input, weight, bias, Tensor() /* undefined */, 
      kernel_size, stride, padding, dilation);
  
  auto output_size = get_output_size<3>(input, weight, 
      stride, padding, dilation);
  for (size_t i = 0; i < output_size.size(); ++i) {
    TORCH_CHECK(output_size[i] > 0,
        "Output size should be positive, got ", output_size[i], " at dim ", i);
  }
  Tensor output = at::empty(output_size, input.options());
  
  Tensor input_ = input;
  Tensor output_ = output;
  if (input.dim() == 4 /* no batch */) {
    input_ = input.unsqueeze(0);
    output_ = output.unsqueeze(0);
  }
  Tensor weight_ = weight.contiguous();
  Tensor bias_ = bias.defined() ? bias.contiguous() : bias;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "conv_depthwise3d",
      [&]{
        int64_t num_outputs = output_.numel();
        int64_t block = 256;
        int64_t grid = std::min((num_outputs - 1) / block + 1, (int64_t)65536);
        const scalar_t* bias_ptr = 
          bias_.defined() ? bias_.data_ptr<scalar_t>() : NULL;

        DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(3, 3, 3, 1, 1, 1)
        DWCONV3D_FORWARD_DISPATCH_OTHERS
      }
  );

  return output;
}

#undef DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION
#undef DWCONV3D_FORWARD_DISPATCH_OTHERS

#define DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION( \
    kt, kh, kw, dilt, dilh, dilw, dt, dh, dw, channel_multiplier) \
  if (kernel_size[0] == (kt) && dilation[0] == (dilt) && stride[0] == (dt) && \
      kernel_size[1] == (kh) && dilation[1] == (dilh) && stride[1] == (dh) && \
      kernel_size[2] == (kw) && dilation[2] == (dilw) && stride[2] == (dw) && \
      (channel_multiplier) == grad_output_.size(1) / input_.size(1)) {        \
    conv_depthwise3d_cuda_backward_input_kernel                               \
    <scalar_t, (kt), (kh), (kw), (dilt), (dilh), (dilw), (dt), (dh), (dw),    \
     (channel_multiplier)>                                                    \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(                 \
        grad_output_.packed_accessor64<scalar_t, 5>(),                        \
        grad_input_.packed_accessor64<scalar_t, 5>(),                         \
        weight_.packed_accessor64<scalar_t, 5>(),                             \
        stride[0], stride[1], stride[2],                                      \
        padding[0], padding[1], padding[2],                                   \
        dilation[0], dilation[1], dilation[2]);                               \
  } else

#define DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS \
  {                                                                         \
    conv_depthwise3d_cuda_backward_input_kernel                             \
    <scalar_t, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1>                      \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(               \
        grad_output_.packed_accessor64<scalar_t, 5>(),                      \
        grad_input_.packed_accessor64<scalar_t, 5>(),                       \
        weight_.packed_accessor64<scalar_t, 5>(),                           \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
  }

std::tuple<Tensor, Tensor, Tensor> conv_depthwise3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, 
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask) {
  conv_depthwise_shape_check<3>(
      input, weight, Tensor() /* undefined */, grad_output,
      kernel_size, stride, padding, dilation);

  bool is_batch = input.dim() == 5;
  auto options = grad_output.options();

  const Tensor grad_output_ = 
    (is_batch ? grad_output.contiguous() 
              : grad_output.contiguous().unsqueeze(0));
  const Tensor input_ =
    (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();

  Tensor grad_input = 
    (output_mask[0] ? at::empty(input.sizes(), options) : Tensor());
  Tensor grad_weight = 
    (output_mask[1] ? at::empty(weight.sizes(), options) : Tensor());
  Tensor grad_bias; /* undefined temporarily */

  Tensor grad_input_ = 
    (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                    : Tensor());

  if (output_mask[0]) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_output.scalar_type(),
        "conv_depthwise3d",
        [&] {
          int64_t num_inputs = grad_input_.numel();
          int64_t block = 256;
          int64_t grid = std::min((num_inputs - 1) / block + 1, (int64_t)65536);
  
          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
            3, 3, 3, 1, 1, 1, 1, 1, 1, 1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
            3, 3, 3, 1, 1, 1, 2, 2, 2, 1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS
        }
    );
  }

  if (output_mask[1]) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_output.scalar_type(),
        "conv_depthwise3d",
        [&] {
          int64_t grid = grad_weight.numel();
          int64_t block = 256;
          int64_t smem = sizeof(scalar_t) * block;
          conv_depthwise3d_cuda_backward_weight_kernel<scalar_t>
            <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
              grad_output_.packed_accessor64<scalar_t, 5>(),
              input_.packed_accessor64<scalar_t, 5>(),
              grad_weight.packed_accessor64<scalar_t, 5>(),
              stride[0], stride[1], stride[2],
              padding[0], padding[1], padding[2],
              dilation[0], dilation[1], dilation[2]);
        }
    );
  }

  if (output_mask[2]) {
    grad_bias = grad_output.sum({0, 2, 3, 4});
  }

  return std::tie(grad_input, grad_weight, grad_bias);
}

#undef DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION
#undef DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS

}
}
