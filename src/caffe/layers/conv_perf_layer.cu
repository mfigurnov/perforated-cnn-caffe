#include <algorithm>
#include <vector>

#include "caffe/common.cuh"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void interpolate_gpu_kernel(const int n,
      const Dtype* input, const unsigned int* indices,
      const int input_size, const int output_size, const int channels,
      Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    const int i = index % output_size;
    const int c = index / output_size;
    output[index] = input[c * input_size + indices[i]];
  }
}

template <typename Dtype>
__global__ void deinterpolate_gpu_kernel(const int n,
      const Dtype* input, const unsigned int* indices,
      const int input_size, const int output_size, const int channels,
      Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    const int i = index % input_size;
    const int c = index / input_size;
    atomicAdd(&output[c * output_size + indices[i]], input[index]);
  }
}

template<typename Dtype>
__global__ void transpose01_gpu_kernel(const int n,
      const Dtype* input, const int d0, const int d1,
      const int d2, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    int k = index;
    int j = index / d2;
    int i = j / d1;
    k %= d2;
    j %= d1;

    output[(j * d0 + i) * d2 + k] = input[(i * d1 + j) * d2 + k];
  }
}

/// @brief refer to gpu forward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    Dtype* col_buff = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const unsigned int* non_perf_indices = non_perforated_indices_.gpu_data();
    const unsigned int* interp_indices = interpolation_indices_.gpu_data();
    Dtype* top_buff = top_buffer_.mutable_gpu_data();
    Dtype* top_buff_2 = top_buffer_2_.mutable_gpu_data();
    for (int n = 0; n < num_; n += micro_batch_size_) {
      int num_images = std::min(micro_batch_size_, num_ - n);
      int current_N = non_perforated_out_ * num_images;
      int weight_offset = M_ * K_;  // number of filter parameters in a group
      // number of values in an input region / column
      int col_offset = K_ * current_N;
      // number of values in an output region / column
      int top_offset = M_ * current_N;
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_perf_gpu(bottom_data + bottom[i]->offset(n),
          num_images, channels_, height_,
          width_, non_perforated_out_, kernel_h_, kernel_w_, pad_h_, pad_w_,
          stride_h_, stride_w_, non_perf_indices, col_buff);
      // Take inner products for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, current_N, K_,
          (Dtype)1., weight + weight_offset * g, col_buff + col_offset * g,
          (Dtype)0., top_buff + top_offset * g);
      }
      // Add bias.
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            current_N, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            bias_multiplier_.gpu_data(),
            (Dtype)1., top_buff);
      }
      const Dtype* result_buff;
      if (num_images > 1) {
        int num_kernels = num_output_ * num_images * non_perforated_out_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        transpose01_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>
          (num_kernels, top_buff, num_output_, num_images, non_perforated_out_,
          top_buff_2);
        result_buff = top_buff_2;
      } else {
        result_buff = top_buff;
      }
      int num_kernels = height_out_ * width_out_ * num_output_ * num_images;
      // NOLINT_NEXT_LINE(whitespace/operators)
      interpolate_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                      CAFFE_CUDA_NUM_THREADS>>>
        (num_kernels, result_buff, interp_indices, non_perforated_out_,
        height_out_*width_out_, num_output_*num_images,
        top_data + top[i]->offset(n));
    }
  }
}

/// @brief refer to gpu backward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  Dtype* top_buff = top_buffer_.mutable_gpu_data();
  Dtype* top_buff_2 = top_buffer_2_.mutable_gpu_data();
  const unsigned int* non_perf_indices = non_perforated_indices_.gpu_data();
  const unsigned int* interp_indices = interpolation_indices_.gpu_data();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    for (int n = 0; n < num_; n += micro_batch_size_) {
      int num_images = std::min(micro_batch_size_, num_ - n);
      int current_N = non_perforated_out_ * num_images;
      int weight_offset = M_ * K_;  // number of filter parameters in a group
      // number of values in an input region / column
      int col_offset = K_ * current_N;
      // number of values in an output region / column
      int top_offset = M_ * current_N;
      int num_kernels = height_out_ * width_out_ * num_output_ * num_images;
      caffe_gpu_set(non_perforated_out_*num_output_*num_images, Dtype(0),
          top_buff);
      // NOLINT_NEXT_LINE(whitespace/operators)
      deinterpolate_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>
          (num_kernels, top_diff + top[i]->offset(n), interp_indices,
          height_out_*width_out_, non_perforated_out_,
          num_output_*num_images, top_buff);
      const Dtype* result_buff;
      if (num_images > 1) {
        int num_kernels = non_perforated_out_ * num_output_ * num_images;
        // NOLINT_NEXT_LINE(whitespace/operators)
        transpose01_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                        CAFFE_CUDA_NUM_THREADS>>>
          (num_kernels, top_buff, num_images, num_output_, non_perforated_out_,
          top_buff_2);
        result_buff = top_buff_2;
      } else {
        result_buff = top_buff;
      }
      // Bias gradient, if necessary.
      if (bias_term_ && this->param_propagate_down_[1]) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, current_N,
            1., result_buff,
            bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        Dtype* col_buff = col_buffer_.mutable_gpu_data();
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_perf_gpu(bottom_data + bottom[i]->offset(n),
            num_images, channels_,
            height_, width_, non_perforated_out_, kernel_h_, kernel_w_,
            pad_h_, pad_w_, stride_h_, stride_w_, non_perf_indices, col_buff);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, current_N,
                (Dtype)1., result_buff + top_offset * g,
                col_buff + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, current_N, M_,
                (Dtype)1., weight + weight_offset * g,
                result_buff + top_offset * g,
                (Dtype)0., col_buff + col_offset * g);
          }
          // col2im back to the data
          col2im_perf_gpu(col_buff, num_images, channels_, height_, width_,
              non_perforated_out_,
              kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_, non_perf_indices,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionPerforatedLayer);

}  // namespace caffe
