#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == 2)
        << "kernel_size must be specified once, or once per spatial dimension ";
    kernel_h_ = conv_param.kernel_size(0);
    kernel_w_ = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : 1);
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  // Setup stride dimensions (stride_).
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == 2)
        << "stride must be specified once, or once per spatial dimension ";
    const int kDefaultStride = 1;
    stride_h_ = (num_stride_dims == 0) ? kDefaultStride :
        conv_param.stride(0);
    stride_w_ = (num_stride_dims == 0) ? kDefaultStride :
        conv_param.stride((num_stride_dims == 1) ? 0 : 1);
  }
  CHECK_GT(stride_h_, 0) << "Stride dimensions must be nonzero.";
  CHECK_GT(stride_w_, 0) << "Stride dimensions must be nonzero.";
  // Setup pad dimensions (pad_).
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == 2)
        << "pad must be specified once, or once per spatial dimension ";
    const int kDefaultPad = 0;
    pad_h_ = (num_pad_dims == 0) ? kDefaultPad :
        conv_param.pad(0);
    pad_w_ = (num_pad_dims == 0) ? kDefaultPad :
        conv_param.pad((num_pad_dims == 1) ? 0 : 1);
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(4);
  weight_shape[0] = num_output_;
  weight_shape[1] = channels_ / group_;
  weight_shape[2] = kernel_h_;
  weight_shape[3] = kernel_w_;
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  height_out_ =
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // DEBUG code. Should be equivalent to non-perforated convolution.
  // non_perforated_indices_.Reshape(1, 1, 1, height_out_*width_out_);
  // int *perf_indices = non_perforated_indices_.mutable_cpu_data();
  //
  // interpolation_indices_.Reshape(1, 1, height_out_, width_out_);
  // int *interp_indices = interpolation_indices_.mutable_cpu_data();
  //
  // for (int i = 0; i < height_out_*width_out_; ++i) {
  //   perf_indices[i] = i;
  //   interp_indices[i] = i;
  // }
  PerforationParameter perf_param = this->layer_param_.perf_param();
  non_perforated_out_ = perf_param.non_perforated_indices_size();
  CHECK_LE(non_perforated_out_, height_out_*width_out_);
  non_perforated_indices_.Reshape(1, 1, 1, non_perforated_out_);
  unsigned int* non_perf_data = non_perforated_indices_.mutable_cpu_data();
  for (int i = 0; i < non_perforated_out_; ++i) {
    non_perf_data[i] = perf_param.non_perforated_indices(i);
  }
  int interp_data_size = perf_param.interpolation_indices_size();
  CHECK_EQ(interp_data_size, height_out_*width_out_);
  interpolation_indices_.Reshape(1, 1, height_out_, width_out_);
  unsigned int* interp_data = interpolation_indices_.mutable_cpu_data();
  for (int i = 0; i < interp_data_size; ++i) {
    interp_data[i] = perf_param.interpolation_indices(i);
  }
  micro_batch_size_ = perf_param.micro_batch_size();
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = non_perforated_out_ * micro_batch_size_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_, 1, N_);
  // Set up two buffers for storing non-perforated pixels
  // of micro_batch_size_ images
  top_buffer_.Reshape(1, num_output_, 1, N_);
  top_buffer_2_.Reshape(1, num_output_, 1, N_);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype>
void interpolate_cpu(const Dtype* input, const unsigned int* indices,
    const int input_size, const int output_size, const int channels,
    Dtype* output) {
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < output_size; ++i) {
      output[i] = input[indices[i]];
    }
    input += input_size;
    output += output_size;
  }
}

template<typename Dtype>
void deinterpolate_cpu(const Dtype* input, const unsigned int* indices,
    const int input_size, const int output_size, const int channels,
    Dtype* output) {
  caffe_set(output_size*channels, Dtype(0), output);
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < input_size; ++i) {
      output[indices[i]] += input[i];
    }
    input += input_size;
    output += output_size;
  }
}

template<typename Dtype>
void transpose01_cpu(const Dtype* input, const int d0, const int d1,
  const int d2, Dtype* output) {
  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(&output[(j*d0 + i)*d2], &input[(i*d1 + j)*d2], d2*sizeof(Dtype));
    }
  }
}

template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    Dtype* col_buff = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const unsigned int* non_perf_indices = non_perforated_indices_.cpu_data();
    const unsigned int* interp_indices = interpolation_indices_.cpu_data();
    Dtype* top_buff = top_buffer_.mutable_cpu_data();
    Dtype* top_buff_2 = top_buffer_2_.mutable_cpu_data();
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
      im2col_perf_cpu(bottom_data + bottom[i]->offset(n),
          num_images, channels_, height_,
          width_, non_perforated_out_, kernel_h_, kernel_w_, pad_h_, pad_w_,
          stride_h_, stride_w_, non_perf_indices, col_buff);
      // Take inner products for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, current_N, K_,
          (Dtype)1., weight + weight_offset * g, col_buff + col_offset * g,
          (Dtype)0., top_buff + top_offset * g);
      }
      // Add bias.
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            current_N, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
            bias_multiplier_.cpu_data(),
            (Dtype)1., top_buff);
      }
      const Dtype* result_buff;
      if (num_images > 1) {
        transpose01_cpu(top_buff, num_output_, num_images, non_perforated_out_,
          top_buff_2);
        result_buff = top_buff_2;
      } else {
        result_buff = top_buff;
      }
      interpolate_cpu(result_buff, interp_indices, non_perforated_out_,
          height_out_*width_out_, num_output_*num_images,
          top_data + top[i]->offset(n));
    }
  }
}

template <typename Dtype>
void ConvolutionPerforatedLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
  }
  Dtype* top_buff = top_buffer_.mutable_cpu_data();
  Dtype* top_buff_2 = top_buffer_2_.mutable_cpu_data();
  const unsigned int* non_perf_indices = non_perforated_indices_.cpu_data();
  const unsigned int* interp_indices = interpolation_indices_.cpu_data();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    for (int n = 0; n < num_; n += micro_batch_size_) {
      int num_images = std::min(micro_batch_size_, num_ - n);
      int current_N = non_perforated_out_ * num_images;
      int weight_offset = M_ * K_;  // number of filter parameters in a group
      // number of values in an input region / column
      int col_offset = K_ * current_N;
      // number of values in an output region / column
      int top_offset = M_ * current_N;
      deinterpolate_cpu(top_diff + top[i]->offset(n), interp_indices,
          height_out_*width_out_, non_perforated_out_,
          num_output_*num_images, top_buff);
      const Dtype* result_buff;
      if (num_images > 1) {
        transpose01_cpu(top_buff, num_images, num_output_, non_perforated_out_,
          top_buff_2);
        result_buff = top_buff_2;
      } else {
        result_buff = top_buff;
      }
      // Bias gradient, if necessary.
      if (bias_term_ && this->param_propagate_down_[1]) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, current_N,
            1., result_buff,
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        Dtype* col_buff = col_buffer_.mutable_cpu_data();
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_perf_cpu(bottom_data + bottom[i]->offset(n),
            num_images, channels_,
            height_, width_, non_perforated_out_, kernel_h_, kernel_w_,
            pad_h_, pad_w_, stride_h_, stride_w_, non_perf_indices, col_buff);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, current_N,
                (Dtype)1., result_buff + top_offset * g,
                col_buff + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, current_N, M_,
                (Dtype)1., weight + weight_offset * g,
                result_buff + top_offset * g,
                (Dtype)0., col_buff + col_offset * g);
          }
          // col2im back to the data
          col2im_perf_cpu(col_buff, num_images, channels_, height_, width_,
              non_perforated_out_,
              kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_, non_perf_indices,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionPerforatedLayer);
#endif

INSTANTIATE_CLASS(ConvolutionPerforatedLayer);
REGISTER_LAYER_CLASS(ConvolutionPerforated);
}  // namespace caffe
