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

Ctxt cufhe::ct_zero;
Ctxt cufhe::ct_one;

// Initialize a plaintext array
void init_ptxt(Ptxt* p, uint x, uint8_t n) {
  for (int i = 0; i < n; i++) {
    p[i].message_ = x & 0x1;
    x >>= 1;
  }
}

void init_fp(Ptxt* p, float x) {
  unsigned int* px = (unsigned int*) &x;
  unsigned int s, e, m, f;

  s = (*px >> 31) & 0x1;
  e = ((*px >> 23) - (127 - 15)) & 0x1F;
  m = (*px >> 13) & 0x3FF;

  f = (s << 15) | (e << 10) | m;

  printf("%x\t%x\t%x\n", s, e, m);
  printf("%x\n", f);

  init_ptxt(p, f, 16);
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
  uint8_t N = 16;

  SetSeed();  // set random seed

  // plaintext
  Ptxt* pta = new Ptxt[N]; // input a
  Ptxt* ptb = new Ptxt[N]; // input b
  Ptxt* ptz = new Ptxt[N]; // output
  Ptxt* pts = new Ptxt;

  init_fp(pta, 0.3);
  init_fp(ptb, 2.2);

  cout<<"A: "<<int(dump_ptxt(pta, N))<<endl;
  cout<<"B: "<<int(dump_ptxt(ptb, N))<<endl;

  Ctxt* cta = new Ctxt[N]; // input a
  Ctxt* ctb = new Ctxt[N]; // input b
  Ctxt* ctz = new Ctxt[N]; // output
  Ctxt* ctc = new Ctxt[N]; // carry
  Ctxt* cts = new Ctxt;

  cout<< "------ Key Generation ------" <<endl;
  PriKey pri_key;
  PubKey pub_key;
  KeyGen(pub_key, pri_key);

  Initialize(pub_key);

  Synchronize();

  // Create CUDA streams for parallel gates.
  Stream* st = new Stream[N];
  for (int i = 0; i < N; i ++)
    st[i].Create();

  cout<< "------ Adder Test ------" <<endl;

  // init_ptxt(pts, 1, 1);
  // init_ptxt(pta, 8, N);
  // init_ptxt(ptb, 3, N);

  // Encrypt
  cout<< "Encrypting..."<<endl;
  for (int i = N-1; i >= 0; i--) {
    Encrypt(cta[i], pta[i], pri_key);
    Encrypt(ctb[i], ptb[i], pri_key);
  }

  Encrypt(*cts, *pts, pri_key);

  Ptxt* pt_one = new Ptxt;
  Ptxt* pt_zero = new Ptxt;
  init_ptxt(pt_zero, 0, 1);
  init_ptxt(pt_one, 1, 1);

  Encrypt(ct_zero, *pt_zero, pri_key);
  Encrypt(ct_one, *pt_one, pri_key);

  // Calculate
  cout<< "Calculating..."<<endl;

  // add_n(ctz, ctc, cta, ctb, pub_key, N);

  // Add(ctz, ctc, cta, ctb, pub_key, N);
  // Add(ctz, ctc, cta, ctb, cts, pub_key, N);
  // Mux(ctz, cta, ctb, cts, pub_key, N);
  // Sub(ctz, ctc, cta, ctb, pub_key, N);
  // Div(ctz, cta, ctb, st, N);
  FpAdd(ctz, cta, ctb, st);

  // Ctxt* p0 = new Ctxt[8];
  // Ctxt* p1 = new Ctxt[8];
  // Ctxt* is = new Ctxt;

  // Not(*is, *cts);

  // for (uint8_t i = 0; i < N; i++) {
  //   And(p0[i], cta[i], *is, pub_key);
  //   And(p1[i], ctb[i], *cts, pub_key);
  // }

  // for (uint8_t i = 0; i < N; i++) {
  //   Or(ctz[i], p0[i], p1[i], pub_key);
  // }

  // Decrypt
  cout<< "Decrypting"<<endl;
  for (int i = N-1; i >= 0; i--) {
    Decrypt(ptz[i], ctz[i], pri_key);
  }

  cout<<"A + B = "<<int(dump_ptxt(ptz, N))<<endl;

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
