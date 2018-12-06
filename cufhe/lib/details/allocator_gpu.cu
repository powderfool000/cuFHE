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

#include <include/details/allocator_gpu.cuh>
#include <include/details/error_gpu.cuh>

namespace cufhe {


// Functions to Call
static bool deviceAllocatorIsOn_ = false;
static DeviceAllocator *deviceAllocator;
void bootDeviceAllocator(size_t size, unsigned long int num) {
	deviceAllocatorIsOn_ = true;
	deviceAllocator = new DeviceAllocator;
	size_t empty, total;
	CuSafeCall(cudaSetDevice(0));
	CuSafeCall(cudaMemGetInfo(&empty, &total));
	int cnt;
	if (num == 0)
		cnt = empty/size;
	else
		cnt = empty/size >= num ? num : empty/size;
	char** ptr = new char* [cnt];
	for (int i=0; i<cnt; i++)
		ptr[i] = deviceAllocator[0].allocate(size);
	for (int i=0; i<cnt; i++)
		deviceAllocator[0].deallocate(ptr[i]);
	delete [] ptr;
}
void haltDeviceAllocator() {
	deviceAllocatorIsOn_ = false;
	delete [] deviceAllocator;
}
bool deviceAllocatorIsOn() {
	return deviceAllocatorIsOn_;
}
// return the id of current selected device
static __inline__ int nowDevice() {
	int ret;
	cudaGetDevice(&ret);
	return ret;
}
void* deviceMalloc(size_t size) {
	return (void *)deviceAllocator[nowDevice()].allocate(size);
}
void deviceFree(void* ptr) {
	deviceAllocator[nowDevice()].deallocate((char* )ptr);
}

// @class DeviceAllocator
DeviceAllocator::DeviceAllocator() {
}
DeviceAllocator::~DeviceAllocator() {
	freeAll();
}
char* DeviceAllocator::allocate(std::ptrdiff_t size) {
	char* result = 0;
	FreeBlocks::iterator freeBlock = freeBlocks.find(size);
	if (freeBlock != freeBlocks.end()) { // found a virtual block
		result = freeBlock->second;
		freeBlocks.erase(freeBlock);
	}
	else { // not available in map
		try { // cudaMalloc
			void* temp;
			CuSafeCall(cudaMalloc(&temp, size));
			result = (char*)temp;
		}
		catch(std::runtime_error &e) { // not enough memory
			return NULL;
		}
	}
	allocatedBlocks.insert(std::make_pair(result, size));
	return result;
}
void DeviceAllocator::deallocate(char* ptr) {
	AllocatedBlocks::iterator iter = allocatedBlocks.find(ptr);
	std::ptrdiff_t size = iter->second;
	allocatedBlocks.erase(iter);
	freeBlocks.insert(std::make_pair(size, ptr));
}
void DeviceAllocator::freeAll() {
	for (FreeBlocks::iterator i = freeBlocks.begin();
		i != freeBlocks.end(); i++) {
		CuSafeCall(cudaFree(i->second));
	}
	for (AllocatedBlocks::iterator i = allocatedBlocks.begin();
		i != allocatedBlocks.end(); i++) {
		CuSafeCall(cudaFree(i->first));
    }
}

// Functions to Call
static bool hostAllocatorIsOn_ = false;
static HostAllocator *hostAllocator;
void bootHostAllocator(size_t size, unsigned long int num) {
	hostAllocatorIsOn_ = true;
	hostAllocator = new HostAllocator;
	size_t empty, total;
	CuSafeCall(cudaSetDevice(0));
	CuSafeCall(cudaMemGetInfo(&empty, &total));
	int cnt;
	if (num == 0)
		cnt = empty/size;
	else
		cnt = empty/size >= num ? num : empty/size;
	char** ptr = new char* [cnt];
	for (int i=0; i<cnt; i++)
		ptr[i] = hostAllocator[0].allocate(size);
	for (int i=0; i<cnt; i++)
		hostAllocator[0].deallocate(ptr[i]);
	delete [] ptr;
}
void haltHostAllocator() {
	hostAllocatorIsOn_ = false;
	delete [] hostAllocator;
}
bool hostAllocatorIsOn() {
	return hostAllocatorIsOn_;
}
void* hostMalloc(size_t size) {
	return (void *)hostAllocator[nowDevice()].allocate(size);
}
void hostFree(void* ptr) {
	hostAllocator[nowDevice()].deallocate((char* )ptr);
}

// @class DeviceAllocator
HostAllocator::HostAllocator() {
}
HostAllocator::~HostAllocator() {
	freeAll();
}
char* HostAllocator::allocate(std::ptrdiff_t size) {
	char* result = 0;
	FreeBlocks::iterator freeBlock = freeBlocks.find(size);
	if (freeBlock != freeBlocks.end()) { // found a virtual block
		result = freeBlock->second;
		freeBlocks.erase(freeBlock);
	}
	else { // not available in map
		try { // cudaMalloc
			void* temp;
			CuSafeCall(cudaMallocHost(&temp, size));
			result = (char*)temp;
		}
		catch(std::runtime_error &e) { // not enough memory
			return NULL;
		}
	}
	allocatedBlocks.insert(std::make_pair(result, size));
	return result;
}
void HostAllocator::deallocate(char* ptr) {
	AllocatedBlocks::iterator iter = allocatedBlocks.find(ptr);
	std::ptrdiff_t size = iter->second;
	allocatedBlocks.erase(iter);
	freeBlocks.insert(std::make_pair(size, ptr));
}
void HostAllocator::freeAll() {
	for (FreeBlocks::iterator i = freeBlocks.begin();
		i != freeBlocks.end(); i++) {
		CuSafeCall(cudaFreeHost(i->second));
	}
	for (AllocatedBlocks::iterator i = allocatedBlocks.begin();
		i != allocatedBlocks.end(); i++) {
		CuSafeCall(cudaFreeHost(i->first));
    }
}

std::pair<void*, MemoryDeleter> AllocatorCPU::New(size_t nbytes) {
  void* ptr = nullptr;
  if (hostAllocatorIsOn()) {
  	// ptr = hostMalloc(nbytes);
  } else {
  	CuSafeCall(cudaMallocHost(&ptr, nbytes));
  }
  return {ptr, Delete};
}

void AllocatorCPU::Delete(void* ptr) { if (deviceAllocatorIsOn()) { /*hostFree(ptr);*/ } else { CuSafeCall(cudaFreeHost(ptr)); } }

MemoryDeleter AllocatorCPU::GetDeleter() { return Delete; }


std::pair<void*, MemoryDeleter> AllocatorBoth::New(size_t nbytes) {
  void* ptr = nullptr;
  CuSafeCall(cudaMallocManaged(&ptr, nbytes));
  return {ptr, Delete};
}

void AllocatorBoth::Delete(void* ptr) { CuSafeCall(cudaFree(ptr)); }

MemoryDeleter AllocatorBoth::GetDeleter() { return Delete; }


std::pair<void*, MemoryDeleter> AllocatorGPU::New(size_t nbytes) {
  void* ptr = nullptr;
  if (deviceAllocatorIsOn()) {
  	ptr = deviceMalloc(nbytes);
  } else {
  	CuSafeCall(cudaMalloc(&ptr, nbytes));
  }
  return {ptr, Delete};
}

void AllocatorGPU::Delete(void* ptr) { if (deviceAllocatorIsOn()) { deviceFree(ptr); } else { CuSafeCall(cudaFree(ptr)); } }

MemoryDeleter AllocatorGPU::GetDeleter() { return Delete; }

} // namespace cufhe
