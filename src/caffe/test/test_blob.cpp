#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
      : blob_(new TBlob<Dtype>()),
        blob_preshaped_(new TBlob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
  TBlob<Dtype>* const blob_;
  TBlob<Dtype>* const blob_preshaped_;
};

TYPED_TEST_CASE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num_axes(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

TYPED_TEST(BlobSimpleTest, TestLegacyBlobProtoShapeEquals) {
  BlobProto blob_proto;

  // Reshape to (3 x 2).
  vector<int> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  this->blob_->Reshape(shape);

  // (3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (0 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(0);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (3 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(3);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (1 x 3 x 2).
  shape.insert(shape.begin(), 1);
  this->blob_->Reshape(shape);

  // (1 x 3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (2 x 3 x 2).
  shape[0] = 2;
  this->blob_->Reshape(shape);

  // (2 x 3 x 2) blob != (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));
}

template <typename TypeParam>
class BlobMathTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BlobMathTest()
      : blob_(new TBlob<Dtype>(2, 3, 4, 5)),
        epsilon_(tol<Dtype>(1e-6, 3e-3)) {}

  virtual ~BlobMathTest() { delete blob_; }
  TBlob<Dtype>* const blob_;
  float epsilon_;
};

TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

TYPED_TEST(BlobMathTest, TestSumOfSquares) {
  typedef typename TypeParam::Dtype Dtype;
  // Uninitialized TBlob should have sum of squares == 0.
  EXPECT_FLOAT_EQ(0.F, this->blob_->sumsq_data());
  EXPECT_FLOAT_EQ(0.F, this->blob_->sumsq_diff());

  for (int i = 0; i < 2; ++i) {
    FillerParameter filler_param;
    filler_param.set_min(-3000);
    filler_param.set_max(3000);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_);
    float expected_sumsq = 0, psum = 0;
    const Dtype *data = this->blob_->cpu_data();
    for (int i = 0; i < this->blob_->count(); ++i) {
      psum += data[i] * data[i];
      if (i > 0 && i % 10 == 0) {
        expected_sumsq += psum;
        psum = 0;
      }
    }
    expected_sumsq += psum;

    EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
        this->epsilon_ * expected_sumsq);
    EXPECT_FLOAT_EQ(0.F, this->blob_->sumsq_diff());

    // Check sumsq_diff too.
    const Dtype kDiffScaleFactor = 7;
    caffe_cpu_scale<Dtype>(this->blob_->count(), kDiffScaleFactor, data,
        this->blob_->mutable_cpu_diff());

    EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
        this->epsilon_ * expected_sumsq);
    const Dtype expected_sumsq_diff =
        expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
    EXPECT_NEAR(expected_sumsq_diff, this->blob_->sumsq_diff(),
        this->epsilon_ * expected_sumsq_diff);

    this->blob_->Reshape(4, 4, 4, 4);  // pow(2) case
  }
}

TYPED_TEST(BlobMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;
  // Uninitialized TBlob should have asum == 0.
  EXPECT_FLOAT_EQ(0.F, this->blob_->asum_data());
  EXPECT_FLOAT_EQ(0.F, this->blob_->asum_diff());
  for (int i = 0; i < 2; ++i) {
    FillerParameter filler_param;
    filler_param.set_min(-30);
    filler_param.set_max(30);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_);
    Dtype expected_asum = 0, psum = 0;
    const Dtype* data = this->blob_->cpu_data();
    for (int i = 0; i < this->blob_->count(); ++i) {
      psum += std::fabs(data[i]);
      if (i > 0 && i % 10 == 0) {
        expected_asum += psum;
        psum = 0;
      }
    }
    expected_asum += psum;
    EXPECT_NEAR(expected_asum, this->blob_->asum_data(),
                this->epsilon_ * expected_asum);
    EXPECT_FLOAT_EQ(0.F, this->blob_->asum_diff());

    // Check asum_diff too.
    const Dtype kDiffScaleFactor = 7;
    caffe_cpu_scale<Dtype>(this->blob_->count(), kDiffScaleFactor, data,
                    this->blob_->mutable_cpu_diff());
    EXPECT_NEAR(expected_asum, this->blob_->asum_data(), this->epsilon_ * expected_asum);
    const Dtype expected_diff_asum = expected_asum * kDiffScaleFactor;
    EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
        this->epsilon_ * expected_diff_asum);
    this->blob_->Reshape(4, 4, 4, 4);  // pow(2) case
  }
}

TYPED_TEST(BlobMathTest, TestAmax) {
  typedef typename TypeParam::Dtype Dtype;
  // Uninitialized TBlob should have amax == 0.
  EXPECT_FLOAT_EQ(0.F, this->blob_->amax_data());
  EXPECT_FLOAT_EQ(0.F, this->blob_->amax_diff());
  for (int i = 0; i < 2; ++i) {
    FillerParameter filler_param;
    filler_param.set_min(-300);
    filler_param.set_max(300);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_);
    Dtype expected_amax = 0, pmax = 0;
    const Dtype* data = this->blob_->cpu_data();
    for (int i = 0; i < this->blob_->count(); ++i) {
      pmax = std::fabs(data[i]);
      if (expected_amax < pmax) {
        expected_amax = pmax;
      }
    }
    EXPECT_NEAR(expected_amax, this->blob_->amax_data(), this->epsilon_ * expected_amax) << i;

    EXPECT_FLOAT_EQ(0.F, this->blob_->amax_diff());
    // Check amax_diff too.
    const Dtype kDiffScaleFactor = 7;
    caffe_cpu_scale<Dtype>(this->blob_->count(), kDiffScaleFactor, data,
        this->blob_->mutable_cpu_diff());
    EXPECT_NEAR(expected_amax, this->blob_->amax_data(), this->epsilon_ * expected_amax) << i;
    const Dtype expected_diff_amax = expected_amax * kDiffScaleFactor;
    EXPECT_NEAR(expected_diff_amax, this->blob_->amax_diff(), this->epsilon_ * expected_diff_amax);
    this->blob_->Reshape(4, 4, 4, 4);  // pow(2) case
  }
}

TYPED_TEST(BlobMathTest, TestScaleData) {
  typedef typename TypeParam::Dtype Dtype;
  EXPECT_FLOAT_EQ(0.F, this->blob_->asum_data());
  EXPECT_FLOAT_EQ(0.F, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-30);
  filler_param.set_max(30);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  const Dtype asum_before_scale = this->blob_->asum_data();

  const Dtype kDataScaleFactor = 3;
  this->blob_->scale_data(kDataScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->blob_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  EXPECT_NEAR(0, this->blob_->asum_diff(), tol<Dtype>(1.e-6, 1.e-4));

  // Check scale_diff too.
  const Dtype kDataToDiffScaleFactor = 7;
  const Dtype* data = this->blob_->cpu_data();
  caffe_cpu_scale<Dtype>(this->blob_->count(), kDataToDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  const Dtype expected_asum_before_scale = asum_before_scale * kDataScaleFactor;
  EXPECT_NEAR(expected_asum_before_scale, this->blob_->asum_data(),
      this->epsilon_ * expected_asum_before_scale);
  const Dtype expected_diff_asum_before_scale =
      asum_before_scale * kDataScaleFactor * kDataToDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum_before_scale, this->blob_->asum_diff(),
      this->epsilon_ * expected_diff_asum_before_scale);

  const Dtype kDiffScaleFactor = 3;
  this->blob_->scale_diff(kDiffScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->blob_->asum_data(),
      this->epsilon_ * asum_before_scale * kDataScaleFactor);
  Dtype expected_diff_asum =
      expected_diff_asum_before_scale * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
      this->epsilon_ * expected_diff_asum);
}

template <typename Dtype>
class BlobSerializationTest : public ::testing::Test {
 protected:
  BlobSerializationTest()
      : blob_(2, 3, 4, 5) {
    Dtype* pdata = blob_.mutable_cpu_data();
    Dtype* pdiff = blob_.mutable_cpu_diff();
    for (int i = 0; i < blob_.count(); ++i) {
      pdata[i] = Dtype(i+1);
      pdiff[i] = Dtype(-i-1);
    }
  }
  TBlob<Dtype> blob_;
};

TYPED_TEST_CASE(BlobSerializationTest, TestDtypesNoFP16);

TYPED_TEST(BlobSerializationTest, TestSerialization) {
  BlobProto proto;
  TBlob<TypeParam> blob;

  bool store_in_old_format = false;
  this->blob_.ToProto(&proto, store_in_old_format, true);
  blob.FromProto(proto, true);
  EXPECT_TRUE(this->blob_.shape() == blob.shape());
  const TypeParam* psrc_data = this->blob_.cpu_data();
  const TypeParam* pdst_data = blob.cpu_data();
  const TypeParam* psrc_diff = this->blob_.cpu_diff();
  const TypeParam* pdst_diff = blob.cpu_diff();
  for (int i = 0; i < this->blob_.count(); ++i) {
    EXPECT_FLOAT_EQ(float(psrc_data[i]), float(pdst_data[i]));
    EXPECT_FLOAT_EQ(float(psrc_diff[i]), float(pdst_diff[i]));
  }

  store_in_old_format = true;
  this->blob_.ToProto(&proto, store_in_old_format, true);
  blob.FromProto(proto, true);
  EXPECT_TRUE(this->blob_.shape() == blob.shape());
  for (int i = 0; i < this->blob_.count(); ++i) {
    EXPECT_FLOAT_EQ(float(psrc_data[i]), float(pdst_data[i]));
    EXPECT_FLOAT_EQ(float(psrc_diff[i]), float(pdst_diff[i]));
  }

  std::string str = blob.to_string();
  EXPECT_GT(str.find("ASUM: 7260"), 0UL);
}

}  // namespace caffe
