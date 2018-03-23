#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

#ifdef USE_NCCL
#include "caffe/util/nccl.hpp"
#endif

namespace caffe {

// Scores Holder
template<typename Dtype>
struct SharedScores {
  explicit SharedScores(size_t nranks) : memory_(nranks) {
    for (size_t i = 0; i < nranks; ++i) {
      memory_[i].resize(MAX_SCORES);
    }
  }
  vector<Dtype>& rank_scores(size_t rank) {
    CHECK_LT(rank, memory_.size());
    return memory_[rank];
  }

 private:
  vector<vector<Dtype>> memory_;
  static constexpr size_t MAX_SCORES = 1000*10;
};

template<typename Dtype>
constexpr size_t SharedScores<Dtype>::MAX_SCORES;

class P2PSync;

class P2PManager {
 public:
  P2PManager(shared_ptr<Solver> root_solver, int nranks, const SolverParameter& param);

  void Run(const vector<int>& gpus);
  void EarlyCancel(P2PSync* killed);

  static void dl_bar_wait() {
    dl_bar->wait();
  }
  static void bar_wait() {
    bar->wait();
  }
  static void rbar_wait(int type_id) {
    if (type_id == 0) {
      rbar0->wait();
    } else {
      rbar1->wait();
    }
  }

 protected:
  const size_t nranks_;
  vector<unique_ptr<P2PSync>> syncs_;
  shared_ptr<SharedScores<float>> shared_;
  shared_ptr<Solver> root_solver_;
#ifdef USE_NCCL
  ncclUniqueId nccl_id_;
#endif

  static unique_ptr<boost::barrier> dl_bar;  // DataLayer sync helper
  static unique_ptr<boost::barrier> bar;
  static unique_ptr<boost::barrier> rbar0;
  static unique_ptr<boost::barrier> rbar1;
};

// Synchronous data parallelism using map-reduce between local GPUs.
class P2PSync : public Solver::Callback, public InternalThread {
  friend class P2PManager;
 public:
  P2PSync(P2PManager* mgr, shared_ptr<Solver> root_solver,
      int rank, int nranks, const SolverParameter& param);
  virtual ~P2PSync();

  // Divide the batch size by the number of solvers
  static unsigned int divide_batch_size(NetParameter* net);

  void allreduce(int type_id, int param_id) override;
  void allreduce_bucket(int type_id, size_t count, void* bucket, Type type) override;
  void soft_barrier() override;
  void reduce_barrier(int type_id) override;
  void saveTestResults(float loss, const vector<float>& scores) override;
  void aggregateTestResults(float* loss, vector<float>* scores) override;

  cudaStream_t comm_stream(int type_id) {
    return comm_stream_[type_id]->get();
  }

 protected:
  void on_start(const vector<shared_ptr<Blob>>& net) override;
#ifdef USE_NCCL
  ncclComm_t nccl_comm_;
#endif
  void InternalThreadEntry() override;

  P2PManager* mgr_;
  const int rank_;
  const size_t nranks_;
  const int initial_iter_;
  shared_ptr<Solver> solver_, root_solver_;
  SolverParameter solver_param_;
  shared_ptr<CudaStream> comm_stream_[2];

  // memory shared between threads
  shared_ptr<SharedScores<float>> shared_;
};

}  // namespace caffe

#endif
