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
void init_ptxt(Ptxt* p, int8_t n) {
  for (int i = 0; i < 8; i++) {
    p[i].message_ = n & 0x1;
    n >>= 1;
  }
}

int8_t dump_ptxt(Ptxt* p) {
  int8_t out = 0;

  for (int i = 0; i < 8; i++) {
    out |= p[i].message_ << i;
  }

  return out;
}

int main() {
  uint8_t N = 8;

  SetSeed();  // set random seed

  // plaintext
  Ptxt* pta = new Ptxt[N]; // input a
  Ptxt* ptb = new Ptxt[N]; // input b
  Ptxt* ptz = new Ptxt[N]; // output

  Ctxt* cta = new Ctxt[N]; // input a
  Ctxt* ctb = new Ctxt[N]; // input b
  Ctxt* ctz = new Ctxt[N]; // output
  Ctxt* ctc = new Ctxt[N]; // carry

  cout<< "------ Key Generation ------" <<endl;
  PriKey pri_key;
  PubKey pub_key;
  KeyGen(pub_key, pri_key);

  cout<< "------ Adder Test ------" <<endl;

  init_ptxt(pta, -1);
  init_ptxt(ptb, -1);

  cout<<"A: "<<int(dump_ptxt(pta))<<endl;
  cout<<"B: "<<int(dump_ptxt(ptb))<<endl;

  // Encrypt
  cout<< "Encrypting..."<<endl;
  for (int i = N-1; i >= 0; i--) {
    Encrypt(cta[i], pta[i], pri_key);
    Encrypt(ctb[i], ptb[i], pri_key);
  }

  // Calculate
  cout<< "Calculating..."<<endl;

  add_n(ctz, ctc, cta, ctb, pub_key, 8);

  // Decrypt
  cout<< "Decrypting"<<endl;
  for (int i = N-1; i >= 0; i--) {
    Decrypt(ptz[i], ctz[i], pri_key);
  }

  cout<<"A + B = "<<int(dump_ptxt(ptz))<<endl;

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
