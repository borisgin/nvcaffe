#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

inline void PyErrReport() {
  PyErr_Print();
  std::cerr << std::endl;
  LOG(FATAL) << "Python error";
}

class PyInit {
  PyThreadState *state_;
  PyGILState_STATE gil_state_;
 public:
  PyInit() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      LOG(INFO) << "Python initialized";
    }
    state_ = PyEval_SaveThread();
    gil_state_ = PyGILState_Ensure();
    LOG(INFO) << "Python state initialized";
  }
  ~PyInit() {
//    PyGILState_Release(gil_state_);
//    PyEval_RestoreThread(state_);
//    Py_Finalize();
  }
  DISABLE_COPY_MOVE_AND_ASSIGN(PyInit);
};

template <typename Ftype, typename Btype>
class PythonLayer : public Layer<Ftype, Btype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Ftype, Btype>(param), self_(bp::handle<>(bp::borrowed(self))) {}

  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("param_str") = bp::str(this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
  }

  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("reshape")(bottom, top);
  }

  inline bool ShareInParallel() const override {
    return this->layer_param_.python_param().share_in_parallel();
  }

  inline const char* type() const override { return "Python"; }

  static std::mutex& mutex() {
    return mutex_;
  }

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("forward")(bottom, top);
  }

  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("backward")(top, propagate_down, bottom);
  }

 private:
  bp::object self_;
  static std::mutex mutex_;
};

template <typename Ftype, typename Btype> std::mutex PythonLayer<Ftype, Btype>::mutex_;

}  // namespace caffe

#endif
