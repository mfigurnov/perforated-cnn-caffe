#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    PerforationParameter* perf_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->kernel_size_size()) {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (conv_param->pad_size()) {
    pad_h = pad_w = conv_param->pad(0);
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (conv_param->stride_size()) {
    stride_h = stride_w = conv_param->stride(0);
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  CHECK_LE(perf_param->non_perforated_indices_size(),
      out->width()*out->height());
  shared_ptr<Blob<Dtype> > temp(new Blob<Dtype>(out->num(), out->channels(),
      1, perf_param->non_perforated_indices_size()));
  // Perforated convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* temp_data = temp->mutable_cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int i = 0; i < temp->width(); i++) {
            int index = perf_param->non_perforated_indices(i);
            CHECK_LE(index, out->width() * out->height());
            int y = index / out->width();
            int x = index % out->width();
            for (int p = 0; p < kernel_h; p++) {
              for (int q = 0; q < kernel_w; q++) {
                int in_y = y * stride_h - pad_h + p;
                int in_x = x * stride_w - pad_w + q;
                if (in_y >= 0 && in_y < in->height()
                  && in_x >= 0 && in_x < in->width()) {
                  temp_data[temp->offset(n, o + o_head, 0, i)] +=
                      in_data[in->offset(n, k + k_head, in_y, in_x)]
                      * weight_data[weights[0]->offset(o + o_head, k, p, q)];
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int i = 0; i < temp->width(); i++) {
          temp_data[temp->offset(n, o, 0, i)] += bias_data[o];
        }
      }
    }
  }
  // Interpolation
  CHECK_EQ(perf_param->interpolation_indices_size(),
      out->width() * out->height());
  for (int n = 0; n < out->num(); n++) {
    for (int o = 0; o < out->channels(); o++) {
      for (int y = 0; y < out->height(); ++y) {
        for (int x = 0; x < out->width(); ++x) {
          int source_index =
            perf_param->interpolation_indices(y * out->width() + x);
          CHECK_LE(source_index, temp->width());
          out_data[out->offset(n, o, y, x)] =
              temp_data[temp->offset(n, o, 0, source_index)];
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param, PerforationParameter* perf_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param, PerforationParameter* perf_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ConvolutionPerforatedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionPerforatedLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionPerforatedLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionPerforatedLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionPerforatedLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 2; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 2; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestSimpleConvolutionPermuted) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_non_perforated_indices(1);
  perf_param->add_interpolation_indices(1);
  perf_param->add_interpolation_indices(0);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestSimpleConvolutionPerforated) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->add_interpolation_indices(0);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest,
           TestSimpleConvolutionPerforatedBatched) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->set_micro_batch_size(3);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 6*4; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, Test1x1ConvolutionPerforated) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 6; ++i) {
    perf_param->add_non_perforated_indices(i*4);
    for (int j = 0; j < 4; ++j) {
      perf_param->add_interpolation_indices(i);
    }
  }
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestSimpleConvolutionGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 2; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionPerforatedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, perf_param,
      layer->blobs(), this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 2; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestGradientPermuted) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_non_perforated_indices(1);
  perf_param->add_interpolation_indices(1);
  perf_param->add_interpolation_indices(0);
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestGradientPerforated) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->add_interpolation_indices(0);
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestGradientPerforatedBatched) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  perf_param->add_non_perforated_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->add_interpolation_indices(0);
  perf_param->set_micro_batch_size(3);
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, Test1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 6*4; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, Test1x1GradientPerforated) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 6; ++i) {
    perf_param->add_non_perforated_indices(i*4);
    for (int j = 0; j < 4; ++j) {
      perf_param->add_interpolation_indices(i);
    }
  }
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionPerforatedLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  PerforationParameter* perf_param = layer_param.mutable_perf_param();
  for (int i = 0; i < 2; ++i) {
    perf_param->add_non_perforated_indices(i);
    perf_param->add_interpolation_indices(i);
  }
  ConvolutionPerforatedLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
