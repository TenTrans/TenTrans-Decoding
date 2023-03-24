#pragma once
#include "HUGlobal.h"
#include "Logging.h"
#include "HUDevice.h"
#include <stdlib.h>
#include <iostream>
#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace TenTrans {
class AllocationException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Memory re-allocation attempted";
  }
};

// Memory Piece in use
class HUMemoryPiece {
private:
  	uint8_t* data_;
  	size_t size_;

public:
  	HUMemoryPiece(uint8_t* data, size_t size) : data_(data), size_(size) {}

  	uint8_t* data() const { return data_; }
  	uint8_t* data() { return data_; }

  	template <typename T>
  	T* data() const {
    	return (T*)data_;
  	}

  	template <typename T>
  	T* data() {
    	return (T*)data_;
  	}

  	size_t size() const { return size_; }

  	void set(uint8_t* data, size_t size) {
    	data_ = data;
    	size_ = size;
  	}

  	void setPtr(uint8_t* data) { data_ = data; }

  	friend std::ostream& operator<<(std::ostream& out, const HUMemoryPiece mp) {
    	out << "MemoryPiece - ptr: " << std::hex << (size_t)mp.data() << std::dec
        	<< " size: " << mp.size();
    	return out;
  	}
};


// Memory Block class
class HUMemoryBlock{
private:
	uint8_t* data_;
	size_t size_;

public:
	HUMemoryBlock(uint8_t* data, size_t size) : data_(data), size_(size) {}
	uint8_t* data() const { return data_; }
	uint8_t* data() { return data_; }

  	size_t size() const { return size_; }

  	bool operator<(const HUMemoryBlock& mp) const {
    	return (size_ < mp.size()) || (size_ == mp.size() && data_ < mp.data());
  	}

  	bool operator==(const HUMemoryBlock& mp) const {
    	return data_ == mp.data() && size_ == mp.size();
  	}

  	bool adjacent(const HUMemoryBlock& mp) const {
    	return data_ + size_ == mp.data() || mp.data() + mp.size() == data_;
  	}

  	friend HUMemoryBlock operator+(const HUMemoryBlock& mp1, const HUMemoryBlock& mp2) {
    	return HUMemoryBlock(mp1.data(), mp1.size() + mp2.size());
  	}

  	HUMemoryBlock combine(const HUMemoryBlock& mp) const {
    	if(mp.data() < this->data())
      		return mp + *this;
    	else
      		return *this + mp;
  	}

  	HUMemoryBlock rest(size_t offset) const { return HUMemoryBlock(data_ + offset, size_ - offset); }
};


// This code draws on memory management part of Marian project
class HUMemPool{
private:
	// device memory manager 
	HUPtr<HUDevice> device_;

	// Available memory space. If available memory space is smaller than needed size, you should call alloc() to allocate new memory space.
  	size_t available_{0};

	// Expanded block size every step
  	size_t step_{128 * Mb};
	
	// Memory alignment 
  	size_t alignment_{256};
  	bool throw_{false};

	// Currently allocated free memory blocks in memory pool
	std::set<HUMemoryBlock> blocks_;

	// Currently memory piece maps in use
	std::unordered_map<uint8_t*, HUPtr<HUMemoryPiece>> allocated_;

	size_t align(size_t size) {
		return ceil(size / (float)alignment_) * alignment_;
	}

	// If all the current memory block size are smaller than required size, you should call grow() to expand free memory block size in memory pool
	// Step 1. call reserve() to allocate new memory space in device, and get new bigger memory piece to use.
	// Step 2. correct old pointer of memory block and call insertBlock() to insert this new bigger memory block.
	// Step 3. correct old allocated maps
	void grow(size_t add) {
		add = align(add);
	    uint8_t* oldData = device_->data();
	    size_t oldSize = device_->size();

	    device_->reserve(oldSize + add);

	    std::set<HUMemoryBlock> oldBlocks;
	    blocks_.swap(oldBlocks);

	    for(auto block : oldBlocks)
			blocks_.insert(HUMemoryBlock(device_->data() + std::distance(oldData, block.data()), block.size()));
	    insertBlock(HUMemoryBlock(device_->data() + oldSize, add));

	    std::unordered_map<uint8_t*, HUPtr<HUMemoryPiece>> oldAllocated;
	    allocated_.swap(oldAllocated);
	    for(auto it : oldAllocated) {
	      uint8_t* newPtr = device_->data() + std::distance(oldData, it.first);
	      allocated_[newPtr] = oldAllocated[it.first];
	      allocated_[newPtr]->setPtr(newPtr);
		}
	}

	// Get suitable memory block
	HUMemoryBlock getBlock(size_t size) {
		size = align(size);
	    auto it = std::lower_bound(blocks_.begin(), blocks_.end(), HUMemoryBlock(nullptr, size));

	    if(throw_ && it == blocks_.end()) {
	      throw AllocationException();
	    }

		while(it == blocks_.end()) {
			grow(step_);
			it = std::lower_bound(blocks_.begin(), blocks_.end(), HUMemoryBlock(nullptr, size));
		}

		available_ -= it->size();
		return *it;
	}	
	
	// If two memory blocks are adjacent, combine these two memory blocks
    // Example: There are A and C memory blocks in memory pool. B is a new memory block that will be inseted, and A and B are adjacent. So we can get new memory pool like AB, C after insertion.  
	void insertBlock(HUMemoryBlock block, bool consolidate = true) {
		available_ += block.size();
		if(consolidate) {
			auto it = blocks_.begin();
			std::vector<decltype(it)> adjacent;
			while(it != blocks_.end()) {
				if(block.adjacent(*it)) {
					block = block.combine(*it);
					adjacent.push_back(it);
				}
				it++;
			}
			for(auto&& a : adjacent)
				blocks_.erase(a);
		}
		blocks_.insert(block);
	}

public:
	// Constructor
	HUMemPool(HUPtr<HUDevice> device, size_t bytes, size_t step, size_t alignment = 256)
		: device_(device),
		  step_(step),
	      available_(0),
		  alignment_(alignment) {
		Reserve(bytes);
	}

	void throwAtReallocation(bool throwRealloc) { throw_ = throwRealloc; }

	// Allocate a corresponding amount of memory space on CPU memory or GPU memory
	void Reserve(size_t bytes) {
		bytes = align(bytes);
		if(bytes > 0)
			device_->reserve(bytes);
		clear();
	}

	template <typename T>
	size_t capacity(size_t num) {
		return align(num * sizeof(T));
	}

	template <typename T>
	HUPtr<HUMemoryPiece> alloc(size_t num) {
		return alloc(capacity<T>(num));
	}

    HUPtr<HUMemoryPiece> alloc(size_t num, TENSOR_DATA_TYPE dataType) {
        if (dataType == TENSOR_DATA_TYPE::TT_INT32) {
            return alloc(capacity<int>(num));
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_FLOAT32) {
            return alloc(capacity<float>(num));
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_DOUBLE) {
            return alloc(capacity<double>(num));
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_FLOAT16) {
            return alloc(capacity<half>(num));
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_INT8) {
            return alloc(capacity<bool>(num));
        }
        else {
            return alloc(capacity<float>(num));
        }  
    }
	
	// Allocate a piece of memory from memory pool as needed
	// Step 1. Get the corresponding memory block according to the needed size, and erase this block from block lists
	// Step 2. If obtained memory block space greater than the size of demand, recycling the extra parts
	// Step 3. Using map to manage allocated memory space
	HUPtr<HUMemoryPiece> alloc(size_t bytes) {
        // std::cout << "[Debug - Alloc]: " << bytes << std::endl;
		bytes = align(bytes);
		HUMemoryBlock block = getBlock(bytes);

		blocks_.erase(block);
		if(block.size() > bytes) {
			insertBlock(block.rest(bytes), false);
		}

		auto ptr = block.data();
		auto mp = HUNew<HUMemoryPiece>(ptr, bytes);
		allocated_[ptr] = mp;
		return mp;
	}

	//free a piece of memory 
	bool free(uint8_t* ptr, size_t bytes) {
        // std::cout << "[Debug - Free]: " << bytes << std::endl;
		bytes = align(bytes);

		ABORT_IF(ptr == 0, "[TenTrans] Double free?");

		if(!ptr)
			return false;

		auto it = allocated_.find(ptr);
	    if(it != allocated_.end()) {
			allocated_.erase(ptr);
			insertBlock(HUMemoryBlock(ptr, bytes), true);
			return true;
		}
		return false;
	}

	//free a piece of memory
	bool free(HUPtr<HUMemoryPiece> mp) {
		if(free(mp->data(), mp->size())) {
			mp->set(nullptr, 0);
			return true;
		}
		return false;
	}

	// clear all free memory block and allocated map record, but not free allocated memory! 
	void clear() {
		available_ = 0;
		blocks_.clear();
		allocated_.clear();
		insertBlock({device_->data(), device_->size()}, false);
	}

	// Get allocated memory piece
	HUPtr<HUMemoryPiece> memory() {
		return HUNew<HUMemoryPiece>(device_->data(), device_->size());
	}

	// Get size of allocated memory piece
	size_t size() { return device_->size(); }

	size_t available() { return available_; }

	// Get device information 
	DeviceId getDevice() { return device_->getDeviceId(); }	
};

}
