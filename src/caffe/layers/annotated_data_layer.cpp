#include <opencv2/core/core.hpp>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
AnnotatedDataLayer<Ftype, Btype>::AnnotatedDataLayer(const LayerParameter& param,
    size_t solver_rank)
  : DataLayer<Ftype, Btype>(param, solver_rank) {}

template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::DataLayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const LayerParameter& param = this->layer_param();
  const AnnotatedDataParameter& anno_data_param = param.annotated_data_param();
  const int batch_size = param.data_param().batch_size();
  const bool cache = this->cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && this->shuffle_ && this->phase_ == TRAIN;
  TBlob<Ftype> transformed_datum;

  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }

  if (this->auto_mode()) {
    if (!sample_areader_) {
      sample_areader_ = std::make_shared<DataReader<AnnotatedDatum>>(param,
          Caffe::solver_count(),
          this->rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          true,
          false,
          cache,
          shuffle,
          false);
    } else if (!areader_) {
      areader_ = std::make_shared<DataReader<AnnotatedDatum>>(param,
          Caffe::solver_count(),
          this->rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          false,
          true,
          cache,
          shuffle,
          this->phase_ == TRAIN);
    }
  } else if (!areader_) {
    areader_ = std::make_shared<DataReader<AnnotatedDatum>>(param,
        Caffe::solver_count(),
        this->rank_,
        this->parsers_num_,
        this->threads_num(),
        batch_size,
        false,
        false,
        cache,
        shuffle,
        this->phase_ == TRAIN);
    start_reading();
  }

  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  shared_ptr<AnnotatedDatum> sample_datum =
      this->sample_only_ ? this->sample_areader_->sample() : this->areader_->sample();
  AnnotatedDatum& anno_datum = *sample_datum;
  this->ResizeQueues();
  this->init_offsets();

  // Calculate the variable sized transformed datum shape.
  vector<int> sample_datum_shape = this->bdt(0)->InferDatumShape(sample_datum->datum());
  // Reshape top[0] and prefetch_data according to the batch_size.
  // Note: all these reshapings here in load_batch are needed only in case of
  // different datum shapes coming from database.
  vector<int> top_shape = this->bdt(0)->InferBlobShape(sample_datum_shape);
  transformed_datum.Reshape(top_shape);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(4, 1);
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
  }
  this->batch_transformer_->reshape(top_shape, label_shape, this->is_gpu_transform());

  LOG(INFO) << this->print_current_device() << " Output data size: "
      << top[0]->num() << ", "
      << top[0]->channels() << ", "
      << top[0]->height() << ", "
      << top[0]->width();
}

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t queue_id) {
  const bool sample_only = this->sample_only_.load();
  TBlob<Ftype> transformed_datum;

  //const bool use_gpu_transform = false;//this->is_gpu_transform();
  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();

  const size_t qid = sample_only ? 0UL : queue_id;
  DataReader<AnnotatedDatum>* reader = sample_only ? sample_areader_.get() : areader_.get();
  shared_ptr<AnnotatedDatum> init_datum = reader->full_peek(qid);
  CHECK(init_datum);

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->bdt(thread_id)->InferBlobShape(init_datum->datum());
  transformed_datum.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_->Reshape(top_shape);

  Ftype* top_data = batch->data_->mutable_cpu_data<Ftype>();
  Ftype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_->mutable_cpu_data<Ftype>();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  size_t current_batch_id = 0UL;
  for (size_t entry = 0; entry < batch_size; ++entry) {
    // get an anno_datum
    shared_ptr<AnnotatedDatum> anno_datum = reader->full_pop(qid, "Waiting for data");
    size_t item_id = anno_datum->record_id() % batch_size;
    if (item_id == 0UL) {
      current_batch_id = anno_datum->record_id() / batch_size;
    }
    AnnotatedDatum distort_datum;
    AnnotatedDatum expand_datum;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(*anno_datum);
      this->bdt(thread_id)->DistortImage(anno_datum->datum(), distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        this->bdt(thread_id)->ExpandImage(distort_datum, &expand_datum);
      } else {
        expand_datum = distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        this->bdt(thread_id)->ExpandImage(*anno_datum, &expand_datum);
      } else {
        expand_datum = *anno_datum;
      }
    }
    AnnotatedDatum sampled_datum;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        this->bdt(thread_id)->CropImage(expand_datum, sampled_bboxes[rand_idx], &sampled_datum);
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    vector<int> shape =
        this->bdt(thread_id)->InferBlobShape(sampled_datum.datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        transformed_datum.Reshape(shape);
        batch->data_->Reshape(shape);
        top_data = batch->data_->mutable_cpu_data<Ftype>();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_->offset(item_id);
    transformed_datum.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum.has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum.set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum.type()) << "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->fdt(thread_id)->Transform(sampled_datum, &transformed_datum, &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->fdt(thread_id)->Transform(sampled_datum.datum(), &(transformed_datum));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum.datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum.datum().label();
      }
    } else {
      this->fdt(thread_id)->Transform(sampled_datum.datum(), &transformed_datum);
    }

    reader->free_push(queue_id, anno_datum);
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_->Reshape(label_shape);
        caffe_set<Ftype>(8, -1, batch->label_->mutable_cpu_data<Ftype>());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_->Reshape(label_shape);
        top_label = batch->label_->mutable_cpu_data<Ftype>();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
//    batch->set_data_packing(packing); todo
  batch->set_id(current_batch_id);
  this->sample_only_.store(false);
}

INSTANTIATE_CLASS_FB(AnnotatedDataLayer);
REGISTER_LAYER_CLASS_R(AnnotatedData);

}  // namespace caffe
