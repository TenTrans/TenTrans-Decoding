#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Logging.h"

namespace TenTrans {

// This code draws on Marian open source project 
class HUShape {
public:	
  std::vector<int> shape_;

public:
  //constructor
  HUShape() : shape_{1} {}

  //constructor with lists. 
  //Example: {2,3,4}
  HUShape(std::initializer_list<int> il) : HUShape() {
    shape_.resize(il.size());
    std::copy(il.begin(), il.end(), begin());
  }

  // resize dimension of shape
  void resize(size_t n) { shape_.resize(n, 1); }

  const int* data() const { return shape_.data(); }

  int* data() { return shape_.data(); }

  // constructor with other shape 
  HUShape(const HUShape& shape) : HUShape() {
    shape_.resize(shape.size());
    std::copy(shape.begin(), shape.end(), begin());
  }

  inline void set(int i, int val) { dim(i) = val; }

  inline int& dim(int i) {
    if(i >= 0) {
      ABORT_IF(i >= size(),
               "Index {} is out of bounds, shape has {} dimension",
               i,
               size());
      return shape_[i];
    } else {
      ABORT_IF((int)size() + i < 0,
               "Negative index {} is out of bounds, shape has {} dimension",
               i,
               size());
      return shape_[size() + i];
    }
  }

  inline const int& dim(int i) const {
    return const_cast<HUShape&>(*this).dim(i);
  }

  inline int operator[](int i) { return dim(i); }

  inline int operator[](int i) const { return dim(i); }

  inline int& back() { return shape_.back(); }

  // if shape is {2,3,4}, stride is {3,4,1}
  inline int stride(int i) const {
    std::vector<int> stride(shape_.size(), 1);
    for(int j = shape_.size() - 2; j >= 0; --j)
      stride[j] = stride[j + 1] * shape_[j + 1];

    if(i >= 0)
      return stride[i];
    else
      return stride[size() + i];
  }

  // dimension of shape
  inline size_t size() const { return shape_.size(); }

  inline int elements() const {
    int el = 1;
    for(auto s : shape_)
      el *= s;
    return el;
  }

  // Find corresponding dimension of element in the tensor by index
  inline void dims(int i, std::vector<int>& d) const {
    d.resize(shape_.size());

    std::vector<int> stride(shape_.size(), 1);
    for(int j = shape_.size() - 2; j >= 0; --j)
      stride[j] = stride[j + 1] * shape_[j + 1];

    for(int j = 0; j < d.size(); ++j)
      d[j] = (i / stride[j]) % shape_[j];
  }

  auto begin() -> decltype(shape_.begin()) { return shape_.begin(); }
  auto begin() const -> decltype(shape_.begin()) { return shape_.begin(); }

  auto end() -> decltype(shape_.end()) { return shape_.end(); }
  auto end() const -> decltype(shape_.end()) { return shape_.end(); }

  auto rbegin() -> decltype(shape_.rbegin()) { return shape_.rbegin(); }
  auto rbegin() const -> decltype(shape_.rbegin()) { return shape_.rbegin(); }

  auto rend() -> decltype(shape_.rend()) { return shape_.rend(); }
  auto rend() const -> decltype(shape_.rend()) { return shape_.rend(); }

  bool operator==(const HUShape& other) const {
    return size() == other.size() && std::equal(begin(), end(), other.begin());
  }

  bool operator!=(const HUShape& other) const { return !(*this == other); }

  std::string toString() const {
    std::stringstream strm;
    strm << "shape=" << (*this)[0];
    for(int i = 1; i < size(); ++i)
      strm << "x" << (*this)[i];
    strm << " size=" << elements() << " (" << elements() * sizeof(float)
         << "B)";
    return strm.str();
  }

  friend std::ostream& operator<<(std::ostream& strm, const HUShape& shape) {
    strm << shape.toString();
    return strm;
  }

  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  int axis(int ax) {
    if(ax < 0)
      return size() + ax;
    else
      return ax;
  }

  // shape1 : {2,3,4}
  // shape2 : {2,3}
  // ==> return {2,3,4}
  static HUShape broadcast(const std::vector<HUShape>& shapes) {
    int maxDims = 0;
    for(auto& s : shapes)
      if(s.size() > maxDims)
        maxDims = s.size();

    HUShape shape;
    shape.resize(maxDims);

    for(auto& s : shapes) {
      for(int i = 0; i < s.size(); ++i) {
        ABORT_IF(shape[-i] != s[-i] && shape[-i] != 1 && s[-i] != 1,
                 "HUShapes {} and {} cannot be broadcasted",
                 (std::string)shape,
                 (std::string)s);
        shape.set(-i, std::max(shape[-i], s[-i]));
      }
    }
    return shape;
  }

  template <typename T>
  static HUShape broadcast(const std::initializer_list<T>& il) {
    return broadcast(std::vector<T>(il));
  }

  template <typename T>
  static HUShape broadcast(const std::vector<T>& nodes) {
    int maxDims = 0;
    for(auto& n : nodes)
      if(n->shape().size() > maxDims)
        maxDims = n->shape().size();

    HUShape shape;
    shape.resize(maxDims);

    for(auto& node : nodes) {
      const HUShape& shapen = node->shape();
      for(int i = 1; i <= shapen.size(); ++i) {
        ABORT_IF(shape[-i] != shapen[-i] && shape[-i] != 1 && shapen[-i] != 1,
                 "HUShapes {} and {} cannot be broadcasted",
                 (std::string)shape,
                 (std::string)shapen);
        shape.set(-i, std::max(shape[-i], shapen[-i]));
      }
    }
    return shape;
  }
};
}

