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

#include <include/cufhe.h>
#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>

namespace cufhe {

void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns);
void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream* st, uint8_t n, uint8_t ns);

void Initialize(const PubKey& pub_key) {
  BootstrappingKeyToNTT(pub_key.bk_);
  KeySwitchingKeyToDevice(pub_key.ksk_);
}

void CleanUp() {
  DeleteBootstrappingKeyNTT();
  DeleteKeySwitchingKey();
}

inline void CtxtCopyH2D(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_device_->data(),
                  c.lwe_sample_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice,
                  st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_->data(),
                  c.lwe_sample_device_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost,
                  st.st());
}

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Not(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = -in.lwe_sample_->data()[i];
}

void Copy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
}

void Ha(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, Stream& st) {
  Xor(z, a, b, st);
  And(co, a, b, st);
}

void Fa(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, const Ctxt& ci, Stream& st) {
  Ctxt t0, t1, t2;

  Xor(t0, a, b, st);
  And(t1, a, b, st);
  And(t2, ci, t0, st);
  Xor(z, ci, t0, st);
  Or(co, t1, t2, st);
}

void Rca(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream& st, uint8_t n) {
  Ha(z[0], c[0], a[0], b[0], st);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], c[i], a[i], b[i], c[i-1], st);
  }
}

void Rca(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream& st, uint8_t n) {
  Fa(z[0], co[0], a[0], b[0], *ci, st);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], co[i], a[i], b[i], co[i-1], st);
  }
}

void Mux(Ctxt* z, Ctxt* in0, Ctxt* in1, Ctxt* s, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt p0[n];
  Ctxt p1[n];
  Ctxt is;

  Not(is, *s, st[0]);

  Synchronize();

  for (uint8_t i = 0; i < n; i++) {
    And(p0[i], in0[i], is, st[i]);
    And(p1[i], in1[i], *s, st[(2*i)%ns]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], p0[i], p1[i], st[i]);
  }
}

void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  if (n >= 4 && 2*ns >= 3*n) {
    Csa(z, c, a, b, st, n/2, ns/3);

    Csa(t0, c0, a+n/2, b+n/2, st+ns/3, (n+1)/2, ns/3);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, st+2*ns/3, (n+1)/2, ns/3);
  } else {
    Rca(z, c, a, b, st[0], n/2);

    Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
    Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);
  }

  Synchronize();

  Mux(z+n/2, t0, t1, c+n/2-1, st, (n+1)/2, ns/2);
  Mux(c+n/2, c0, c1, c+n/2-1, st+ns/2, (n+1)/2, ns/2);
}

void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  if (n >= 4 && 2*ns >= 3*n) {
    Csa(z, co, a, b, ci, st, n/2, ns/3);

    Csa(t0, c0, a+n/2, b+n/2, st+ns/3, (n+1)/2, ns/3);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, st+2*ns/3, (n+1)/2, ns/3);
  } else {
    Rca(z, co, a, b, ci, st[0], n/2);

    Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
    Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);
  }

  Synchronize();

  Mux(z+n/2, t0, t1, co+n/2-1, st, (n+1)/2, ns/2);
  Mux(co+n/2, c0, c1, co+n/2-1, st+ns/2, (n+1)/2, ns/2);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns) {
  Csa(z, c, a, b, st, n, ns);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* s, Stream* st, uint8_t n, uint8_t ns) {
  Csa(z, c, a, b, s, st, n, ns);
}

void Sub(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt t[n];

  for (uint8_t i = 0; i < n; i++) {
    Not(t[i], b[i], st[i]);
  }

  Synchronize();

  Add(z, c, a, t, &ct_one, st, n, ns);
}

void Mul(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
}

// a / b = z
void Div(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt r[2*n];      // non-restoring reg
  Ctxt* s = r+n;      // 'working' index
  Ctxt t0[n], t1[n];    // temp
  Ctxt c[n];    // carry
  Ctxt bi[n];   // bi = -b

  Synchronize();

  // initialize
  for (int i = 0; i < n; i++) {
    Not(bi[i], b[i], st[i]);
    Copy(s[i], ct_zero, st[i]);
    Copy(r[i], a[i], st[i]);
  }

  Synchronize();

  Add(bi, c, bi, s, &ct_one, st, n, ns);

  Synchronize();

  // first iteration is always subtract (add bi)
  s--;
  Add(t0, c, s, bi, st, n, ns);

  Synchronize();

  for (int i = 0; i < n; i++) {
    Copy(s[i], t0[i], st[i]);
  }

  Synchronize();

  Not(z[s-r], s[n-1], st[0]);

  Synchronize();

  while (s > r) {
    s--;
    Add(t0, c, s, bi, st, n, ns);
    Add(t1, c, s, b, st, n, ns);

    Synchronize();

    Mux(s, t0, t1, s+n, st, n, ns);

    Synchronize();

    Not(z[s-r], s[n-1]);

    Synchronize();
  }
}
//--------------------------------------------------------------------------------------------------------------------------
void halffloatAdd(Ctxt* z, Ctxt* in1, Ctxt* in2, Stream* st) {
  int ns = 10;
  //Part 1
  Ctxt smallZero;
  Ctxt one; // a simple holder for one
  Ctxt expsum[5];  //for the exponent subtraction
  Ctxt negcheck;   //for checking which exponent is larger
  Ctxt tempexpo[5];  //tentative exponent  
  Ctxt in1exp [5]; //exponent of in1
  Ctxt in2exp [5]; //exponent of in2
  //Ctxt zero;     //a zero for shifting
  Ctxt smallIn[16]; //holder for the smaller input
  Ctxt bigIn[16];  //holder for the larger input
  Ctxt smallInman[13]; //holder for smaller input mantisa
  Ctxt bigInman[13];   //holder for the bigger input mantisa
  //Part 2
  Ctxt in1mantisaR[13];  //Used for holding the round bits, and adding
  Ctxt in2mantisaR[13];  //Used for holding the round bits, and adding
  Ctxt smallOut[13][10]; //smaller output 

  //Part 3
  Ctxt mantisaSum[13]; //sum of the two mantisas
  Ctxt* co; //carryout

  //Part 4
  Ctxt mantisaSumcarryo[13]; //mantisaSum if there is a carry out in the addition
  Ctxt tempexpoCo[5];   //the exponent if there is a carry  
  Ctxt expoOne[5]; // a one to add to the exponents for rounding

  //PArt 5
  Ctxt mantisaSumround[13];
  Ctxt finalSum[19];
  Ctxt fullOne[19];
  Ctxt roundhold;
  Ctxt finalSumRound[19];

//--------------------Zero and One Inililiations----
  initZero(in1mantisaR, 13, st); //need to initialize to 0 for the round bits to be 0
  initZero(in2mantisaR, 13, st); //need to initialize to 0 for the round bits to be 0
  //initZero(&zero, 13, st);   //make =0
  initZero(&smallZero, 1, st);
  initOne(&negcheck, 1, st);     //make =1
  for(int i = 0; i <13; i++){
    initZero(smallOut[i], 5, st);
  }
  //part4 
  initOne(&one, 1, st);
  initZero(expoOne, 5, st);
  And(expoOne[0], expoOne[0], &one, st[0]);
  //part5
  initZero(fullOne, 19, st);
  And(fullOne[0], fullOne[0], &one, st[0]);

//-------------------getting temporary arrays of the exponents and mantisas --------------
  for(int i = 0; i < 5; i++){
    in1exp[i] = in1[9+i];
    in2exp[i] = in2[9+i];
  }
  for(int i = 0; i< 10; i++){             
    in1mantisaR[i+3] = in1[i];  //leae last 3 bits for the round bits
    in2mantisaR[i+3] = in2[i];
  }

//--------------------------PART 1-----------------------------

  Sub(expsum, &smallZero, in1exp, in2exp, st, 5); // subtract the first exponentes

  //check if negative to determine which exponent is larger

  And(negcheck, &negcheck, expsum[4], st[1]);

//if "negcheck" is positive, then input2 is larger, else input1 larger. 
  Mux(tempexpo, in1exp, in2exp, &negcheck, st, 5);       //make tempexpo into whichever exponent is higher 
  Mux(smallIn, in2, in1,  &negcheck, st, 16);            //chosing which input is the "smaller" one , aka the one with smaller input
  Mux(bigIn, in1, in2, &negcheck, st, 16);            // which input is bigger
  Mux(smallInman, in2mantisaR, in1mantisaR, &negcheck, st, 13);  //chose the smaller mantisa
  Mux(bigInman, in1mantisaR, in2mantisaR,  &negcheck, st, 13);    //chose the larger mantisa

//-------------------------PART 2-------------------------------
//-----------------------------------
//  Xor(expsum[4], expsum [4], negcheck, st[2]); //make sure it is positive
///----------------------------- IDK if this is needed, basicly its checking if the value is negative, meaning we could have to shift differently? probs not

  for(int i=0; i <= 10; i++){
    if(i < 3){                    //cases for when nothing shifted out past the sticky
      Shift(smallOut[i], smallInman, &smallZero, 10, i, st);
    }
    else{                     //itterating 0-10 shifts, assigning rounding bits along the way
      Shift(smallOut[i], smallInman, &smallZero, 10, i, st);
      Copy(smallOut[2][i], smallInman[i-1], st[7]);  //guard
      Copy(smallOut[1][i], smallInman[i-2], st[8]);  //round
      Copy(smallOut[0][i], smallInman[i-3], st[9]);  //stickey
//ISSUES-------------------------------------
      Or(smallOut[0][i], smallOut[0][i], smallOut[0][i-1], st[0]); //ORing the sticky together
    }
  }
//-----------------------THIS IS THE 10 BIT MUX RIGHT HERE-----------------------------
//  bitMUX(smallInman, smallOut[0], smallOut[1], smallOut[2], smallOut[3], ...ect, expsum, pubkey_, n);
//-----------------------STILL IN PROCESS------------------------------------------

//-----------------------Part 3 ------------------------------
  Add(mantisaSum, co, smallInman, bigInman, &smallZero, st, 13); //adding the mantisas together
//----------------------Part 4 -------------------------------
  //Normalizing for Carry Out
  roundNormalize(finalSum, tempexpoCo, mantisaSumcarryo, co, mantisaSum, expoOne, tempexpo, &smallZero, st);
//----------------------Part 5 -------------------------------------
  Or(roundhold, finalSum[2], finalSum[0], st[3]);
  And(roundhold, &roundhold, finalSum[1], st[3]);  //if all are 1, then make round temp 1, so you will round

  Add(finalSumRound, co, finalSum, fullOne, &smallZero, st, 19); //adds one to the answer
  Mux(finalSum, finalSumRound, finalSum, co, st, 19); //select the correct finalSum using co as select

  //cut off round bits
  removeRound(z, finalSum, 19, st);
  
  //*insert clause for when mantisa is all ones and need to round up 

}



void roundNormalize(Ctxt* finalSum, Ctxt* tempexpoCo, Ctxt* mantisaCosum, Ctxt* co, Ctxt* mantisaSum, Ctxt* expoOne, Ctxt* tempexpo, Ctxt* smallZero, Stream* st){
  Shift(mantisaCosum, mantisaSum, smallZero, 10, 1, st);   // if carry out, shift mantisa right by 1    ***note, may be interpretting "product" wrong in the algorithm description***
  Add(tempexpoCo, co, tempexpo, expoOne, smallZero, st, 5);  // expoOne is just a 5 bit number 00001 to add to tempexpo

  Mux(tempexpo, tempexpoCo, tempexpo, co, st, 5); //chose to use the added exponent or not
  Mux(mantisaSum, mantisaCosum, mantisaSum, co, st, 13); //chosing the correct mantisa
  for(int i=0; i<19; i++){
    if(i<13){
      finalSum[i] = mantisaSum[i];
    }
    else{
      finalSum[i] = tempexpo[i+13];
    }
  }
}

void normShift(Ctxt* smallOut, Ctxt* smallIn, Ctxt* smallZero, uint8_t n, int nshift, Stream* st){
  int temp = (n-nshift);
  for(int i = 0; i<(temp); i++){
    Copy(smallOut[i], smallIn[nshift+i], st[i]);
  }
  for(int i= temp; i < n; i++){
    Copy(smallOut[i+3], smallZero, st[i]);
  }
}

void Shift(Ctxt* smallOut, Ctxt* smallIn, Ctxt* smallZero, uint8_t n, int nshift, Stream* st){
  int temp = (n-nshift);
  for(int i = 0; i<(temp); i++){
    Copy(smallOut[i], smallIn[nshift+i], st[i]);
  }
  for(int i=(temp); i< 13; i++){
    Copy(smallOut[i+3], smallZero, st[i]);
  }
}

void removeRound(Ctxt* withoutround, Ctxt* withround, uint8_t n, Stream* st){
  for(int i = 3; i < n; i++){
    withoutround[i-3] = withround[i];
  }
}

void initZero(Ctxt* ct, const int ln, Stream* st) {
  Ctxt* ct1 = new Ctxt[ln];

  for(int i = 0; i < ln; i++) {
    Not(ct1[i], ct[i], st[i]);
  }

  for(int i = 0; i < ln; i++) {
    And(ct[i], ct[i], ct1[i], st[i]);
  }
}

void initOne(Ctxt* ct, const int ln, Stream* st) {
  Ctxt* ct1 = new Ctxt[ln];

  for(int i = 0; i < ln; i++) {
    Not(ct1[i], ct[i], st[i]);
  }

  for(int i = 0; i < ln; i++) {
    Nand(ct[i], ct[i], ct1[i], st[i]);
  }
}
} // namespace cufhe