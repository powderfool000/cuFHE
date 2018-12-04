/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "allocator.h"
#include <map>

namespace cufhe {

// A customized memory method for each GPU device.
class DeviceAllocator {
public:
	DeviceAllocator();
	~DeviceAllocator();
	// find empty block in map;
	// if not available, cudaMalloc() and add a block to map
	char* allocate(std::ptrdiff_t size);
	// remove ptr from map, set block to empty
	void deallocate(char* ptr);
	// cudaFree() all blocks in map
	void freeAll();
private:
	typedef std::multimap<std::ptrdiff_t, char*> FreeBlocks;
	typedef std::map<char*, std::ptrdiff_t> AllocatedBlocks;
	FreeBlocks freeBlocks;
	AllocatedBlocks allocatedBlocks;
};

// Return a pointer on current device.
void* deviceMalloc(size_t size);

// Free a pointer on current device.
void deviceFree(void* ptr);

// Check if DeviceAllocator is enabled.
bool deviceAllocatorIsOn();

// Create a deviceAllocator for each GPU;
//	slice currect available GPU memory into blocks with a large "size";
//	(now has problem when size is too small map is too large).
void bootDeviceAllocator(size_t size, unsigned long int num = 0);

// Free all allocated GPU memory and deviceAllocators.
void haltDeviceAllocator();

// A customized memory method for each GPU device.
class HostAllocator {
public:
	HostAllocator();
	~HostAllocator();
	// find empty block in map;
	// if not available, cudaMalloc() and add a block to map
	char* allocate(std::ptrdiff_t size);
	// remove ptr from map, set block to empty
	void deallocate(char* ptr);
	// cudaFree() all blocks in map
	void freeAll();
private:
	typedef std::multimap<std::ptrdiff_t, char*> FreeBlocks;
	typedef std::map<char*, std::ptrdiff_t> AllocatedBlocks;
	FreeBlocks freeBlocks;
	AllocatedBlocks allocatedBlocks;
};

// Return a pointer on current device.
void* hostMalloc(size_t size);

// Free a pointer on current device.
void hostFree(void* ptr);

// Check if DeviceAllocator is enabled.
bool hostAllocatorIsOn();

// Create a deviceAllocator for each GPU;
//	slice currect available GPU memory into blocks with a large "size";
//	(now has problem when size is too small map is too large).
void bootHostAllocator(size_t size, unsigned long int num = 0);

// Free all allocated GPU memory and deviceAllocators.
void haltHostAllocator();

class AllocatorCPU: public Allocator {
public:
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);
  static void Delete(void* ptr);
  MemoryDeleter GetDeleter();
};

class AllocatorBoth: public Allocator {
public:
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);
  static void Delete(void* ptr);
  MemoryDeleter GetDeleter();
};

class AllocatorGPU: public Allocator {
public:
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);
  static void Delete(void* ptr);
  MemoryDeleter GetDeleter();
};

} // namespace cufhe
