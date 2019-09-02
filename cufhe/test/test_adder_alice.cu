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

//#include <cuda_profiler_api.h>

Ctxt cufhe::ct_zero;
Ctxt cufhe::ct_one;

// Initialize a plaintext array
void init_ptxt(Ptxt* p, int32_t x, uint32_t n) {
  for (int i = 0; i < n; i++) {
    p[i].message_ = x & 0x1;
    x >>= 1;
  }
}

//Dump the interger intake to binary
int32_t dump_ptxt(Ptxt* p, uint32_t n) {
  int32_t out = 0;

  for (int i = n-1; i >= 0; i--) {
    cout<<p[i].message_;
    out |= p[i].message_ << i;
  }

  cout<<endl;

  return out;
}

int main() {
  uint32_t N = 32;

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


  Ptxt* ptx = new Ptxt[N]; // input a
  Ptxt* pty = new Ptxt[N]; // input b
  Ctxt* ctx = new Ctxt[N]; // input b
  Ctxt* cty = new Ctxt[N]; // input b

  char f1lename[13];
  char f2lename[13];

  cout<< "------ ALU Test ------" <<endl;

  init_ptxt(pts, 1, 1);
  init_ptxt(pta, 8, N);
  init_ptxt(ptb, 3, N);

  cout<<"A: "<<dump_ptxt(pta, N)<<endl;
  cout<<"A: "<<int(dump_ptxt(pta, N))<<endl;
  cout<<"B: "<<int(dump_ptxt(ptb, N))<<endl;
  cout<<"s: "<<int(dump_ptxt(pts, 1))<<endl;

  cout<< "------ Read key files ------" <<endl;
  PriKey pri_key; 
  ReadPriKeyFromFile(pri_key, "pri_key.txt");
  //ReadPubKeyFromFile(pub_key, "pub_key.txt");

  // Encrypt
  cout<< "Encrypting..."<<endl;
  for (int i = N-1; i >= 0; i--) {
    Encrypt(cta[i], pta[i], pri_key);
    snprintf(f1lename, 12, "cipherA%d.txt", i);
    WriteCtxtToFile(cta[i], f1lename);

    Encrypt(ctb[i], ptb[i], pri_key);
    snprintf(f2lename, 12, "cipherB%d.txt", i);
    WriteCtxtToFile(ctb[i], f2lename);
  }

  /*cout<< "Encrypting done..."<<endl;
  WriteCtxtToFile(cta, "cipher1.txt");
  cout<< "encrypt 1st plaintext"<<endl;

  WriteCtxtToFile(ctb, "cipher2.txt");
  cout<< "encrypt 2nd plaintext"<<endl;*/

 /* for (int i = N-1; i >= 0; i--) {

    //snprintf(f1lename, 12, "cipherA%d.txt", i);
    ReadCtxtFromFile(ctx[i],f1lename);

    //snprintf(f2lename, 12, "cipherB%d.txt", i);
    ReadCtxtFromFile(cty[i],f2lename);
    //cout<< "i = "<< i <<endl;
  }
  */
  /*ReadCtxtFromFile(ctx,"cipher1.txt");
  ReadCtxtFromFile(cty,"cipher2.txt");*/
/*
  cout<< "Decrypting"<<endl;
  for (int i = N-1; i >= 0; i--) {
    Decrypt(ptx[i], ctx[i], pri_key);
    Decrypt(pty[i], cty[i], pri_key);
    //cout<< "i = "<< i << dump_ptxt(pta[i], N) <<endl;
  }

  cout<<"A = "<<int(dump_ptxt(ptx, N))<<endl;
  cout<<"B = "<<int(dump_ptxt(pty, N))<<endl;
*/
  //from this point on no need check

  cout<< "Extra done..."<<endl;
  Encrypt(*cts, *pts, pri_key);

  Ptxt* pt_one = new Ptxt;
  Ptxt* pt_zero = new Ptxt;
  //init_ptxt(pt_zero, 0, 1);
  //init_ptxt(pt_one, 1, 1);

  Encrypt(ct_zero, *pt_zero, pri_key);
  Encrypt(ct_one, *pt_one, pri_key);
  
  delete [] pta;
  delete [] ptb;
  delete [] ptz;
  delete [] cta;
  delete [] ctb;
  delete [] ctz;
  delete [] ctc;
  return 0;
}
