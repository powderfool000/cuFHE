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
#include <include/cufhe_cpu.h>
using namespace cufhe;

#include <iostream>
using namespace std;

Ctxt cufhe::ct_zero;
Ctxt cufhe::ct_one;

// Full adder without carry in
void full_adder(Ctxt& z, Ctxt& co, Ctxt& a, Ctxt& b, PubKey& pub_key) {
  And(co, a, b, pub_key);
  Xor(z, a, b, pub_key);
}

// Full adder with carry in
void full_adder(Ctxt& z, Ctxt& co, Ctxt& a, Ctxt& b, Ctxt& ci, PubKey& pub_key) {
  Or(co, a, b, pub_key);
  And(co, ci, co, pub_key);
  And(z, a, b, pub_key);
  Or(co, co, z, pub_key);

  Xor(z, a, b, pub_key);
  Xor(z, ci, z, pub_key);
}

// N bit adder with overflow
void add_n(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  full_adder(z[0], c[0], a[0], b[0], pub_key);

  for (int i = 1; i < n; i++) {
    full_adder(z[i], c[i], a[i], b[i], c[i-1], pub_key);
  }
}

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
  Ctxt* cts = new Ctxt;

  cout<< "------ Key Generation ------" <<endl;
  PriKey pri_key;
  PubKey pub_key;
  KeyGen(pub_key, pri_key);

  cout<< "------ Adder Test ------" <<endl;

  init_ptxt(pts, 1, 1);
  init_ptxt(pta, 8, N);
  init_ptxt(ptb, 3, N);

  cout<<"A: "<<int(dump_ptxt(pta, N))<<endl;
  cout<<"B: "<<int(dump_ptxt(ptb, N))<<endl;

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
  Div(ctz, cta, ctb, pub_key, N);

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
