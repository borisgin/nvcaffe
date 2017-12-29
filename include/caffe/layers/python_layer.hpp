#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

class PyGILAquire {
  PyGILState_STATE state_;
 public:
  PyGILAquire() : state_(PyGILState_Ensure()) {}

  ~PyGILAquire() {
    PyGILState_Release(state_);
  }
DISABLE_COPY_MOVE_AND_ASSIGN(PyGILAquire);
};

class PyGILRelease {
  PyThreadState *state_;
 public:
  PyGILRelease() {
    state_ = PyEval_SaveThread();
  }
  ~PyGILRelease() {
    PyEval_RestoreThread(state_);
  }
  DISABLE_COPY_MOVE_AND_ASSIGN(PyGILRelease);
};

namespace caffe {

inline void PyErrFatal() {
  PyErr_Print();
  abort();
}
void PyErrReportAndForward();

template <typename Ftype, typename Btype>
class PythonLayer : public Layer<Ftype, Btype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Ftype, Btype>(param), self_(bp::handle<>(bp::borrowed(self))) {}

  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      PyGILAquire pgil;
      self_.attr("param_str") = bp::str(this->layer_param_.python_param().param_str());
      self_.attr("setup")(bottom, top);
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      PyGILAquire pgil;
      self_.attr("reshape")(bottom, top);
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  inline bool ShareInParallel() const override {
    return this->layer_param_.python_param().share_in_parallel();
  }

  inline const char* type() const override { return "Python"; }

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      PyGILAquire pgil;
      self_.attr("forward")(bottom, top);
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override {
    try {
      PyGILAquire pgil;
      self_.attr("backward")(top, propagate_down, bottom);
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif
