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

// Include these two files for CPU computing.
#include <include/cufhe_gpu.cuh>
using namespace cufhe;

#include <iostream>
using namespace std;

#include <cuda_profiler_api.h>

Ctxt cufhe::ct_zero;
Ctxt cufhe::ct_one;

// Initialize a plaintext array
void init_ptxt(Ptxt* p, int8_t x, uint8_t n) {
  for (int i = 0; i < n; i++) {
    p[i].message_ = x & 0x1;
    x >>= 1;
  }
}

int8_t dump_ptxt(Ptxt* p, uint8_t n) {
  int8_t out = 0;

  for (int i = n-1; i >= 0; i--) {
    cout<<p[i].message_;
    out |= p[i].message_ << i;
  }

  cout<<endl;

  return out;
}

int main() {
  uint8_t N = 8;

  SetSeed();  // set random seed

  // plaintext
  Ptxt* pta = new Ptxt[N]; // input a
  Ptxt* ptb = new Ptxt[N]; // input b
  Ptxt* ptz = new Ptxt[N]; // output
  Ptxt* pts = new Ptxt;

  Ctxt* cta = new Ctxt[N]; // input a
  Ctxt* ctb = new Ctxt[N]; // input b
  Ctxt* ctz = new Ctxt[N]; // output
  Ctxt* ctc = new Ctxt[N]; // carry
  Ctxt* ctt = new Ctxt[10*N+1];
  Ctxt* cts = new Ctxt;

  char filename[11];

  /*float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);*/

  cout<< "------ Open Key ------" <<endl;
  PriKey pri_key;
  ReadPriKeyFromFile(pri_key, "pri_key.txt");
 
  Encrypt(*cts, *pts, pri_key);

  Ptxt* pt_one = new Ptxt;
  Ptxt* pt_zero = new Ptxt;
  init_ptxt(pt_zero, 0, 1);
  init_ptxt(pt_one, 1, 1);

  Encrypt(ct_zero, *pt_zero, pri_key);
  Encrypt(ct_one, *pt_one, pri_key);

  // Calculate
  /*cout<< "Calculating..."<<endl;

  cudaProfilerStart();

  cudaEventRecord(start, 0);

  Add(ctz, ctc, cta, ctb, ctt, st, N);

  Synchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  */

  // Decrypt

  for (int i = N-1; i >= 0; i--) {
    
    snprintf(filename, 11, "answer%d.txt", i);
    ReadCtxtFromFile(ctz[i],filename);
  }

  cout<< "Decrypting"<<endl;
  for (int i = N-1; i >= 0; i--) {
    Decrypt(ptz[i], ctz[i], pri_key);
  }

  cout<<"A + B = "<<int(dump_ptxt(ptz, N))<<endl;

  /*cudaEventElapsedTime(&et, start, stop);
  cout<<"Elapsed: "<<et<<" ms"<<endl;*/

  Decrypt(pta[0], ctc[N-1], pri_key);

  cout<<"carry out: "<<pta[0].message_<<endl;

  delete [] pta;
  delete [] ptb;
  delete [] ptz;
  delete [] cta;
  delete [] ctb;
  delete [] ctz;
  delete [] ctc;
  return 0;
}
