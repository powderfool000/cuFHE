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
void halffloatAdd(z, Ctxt in1*, Ctxt in2*, pubkey_){ 
  //Part 1
  Ctxt one; // a simple holder for one
  Ctxt* expsum[5];  //for the exponent subtraction
  Ctxt* negcheck;   //for checking which exponent is larger
  Ctxt* tempexpo[5];  //tentative exponent  
  Ctxt* in1exp [5]; //exponent of in1
  Ctxt* in2exp [5]; //exponent of in2
  Ctxt* zero;     //a zero for shifting
  Ctxt* smalIn[16]; //holder for the smaller input
  Ctxt* bigIn[16];  //holder for the larger input
  Ctxt* smallInman[13]; //holder for smaller input mantisa
  Ctxt* bigInman[13];   //holder for the bigger input mantisa

  //Part 2
  Ctxt* in1mantisaR[13];  //Used for holding the round bits, and adding
  Ctxt* in2mantisaR[13];  //Used for holding the round bits, and adding
  Ctxt* smallOut[13][10]; //smaller output 

  //Part 3
  Ctxt* mantisaSum[13]; //sum of the two mantisas
  Ctxt* co; //carryout

  //Part 4
  Ctxt* mantisaSumcarryo[13]; //mantisaSum if there is a carry out in the addition
  Ctxt* tempexpocarry[5];   //the exponent if there is a carry  
  Ctxt* expoOne[5];

  //PArt 5
  Ctxt* mantisaSumround[13];
  Ctxt* finalSum[19];
  Ctxt* fullOne[19];

//--------------------Zero and One Inililiations----
  initZero(in1mantisaR, 13, pubkey_); //need to initialize to 0 for the round bits to be 0
  initZero(in2mantisaR, 13, pubkey_); //need to initialize to 0 for the round bits to be 0
  initZero(Ctxt* zero, n, pubkey_);   //make =0
  initone(negcheck, pubkey_);     //make =1
  //part4 
  initOne(one, pubkey);
  initZero(expoOne, pubkey);
  And(expoOne, expoOne, one, pubkey_);
  //part5
  initZero(fullOne, pubkey_);
  And(fullOne, fullOne, one, pubkey_);

//-------------------getting temporary arrays of the exponents and mantisas --------------
  for(i = 0; i < 5; i++){
  in1exp*[i] = in1*[9+i];
  in2exp*[i] = in2*[9+i];
  }
  for(i = 0; i< 10; i++){             
    in1mantisaR[i] = in1[i+3];  //leae last 3 bits for the round bits
    in2mantisaR[i] = in2[i+3];
  }

//--------------------------PART 1-----------------------------

  Sub(expsum*, in1exp*, in2exp*, 5, pubkey_); // subtract the first exponentes

  //check if negative to determine which exponent is larger

  AND(negcheck, negcheck, expsum[4], pubkey_);

//if "negcheck" is positive, then input2 is larger, else input1 larger. 
  MUX(tempexpo*, in1exp*, in2exp*, negcheck, pubkey_, 5);       //make tempexpo into whichever exponent is higher 
  MUX(smalIn, in2, in1,  negcheck, pubkey_, 16);            //chosing which input is the "smaller" one , aka the one with smaller input
  MUX(bigIn, in1, in2, negcheck, pubkey_, 16);            // which input is bigger
  MUX(smalInman, in2mantisaR, in1mantisaR, negcheck, pubkey_, 13);  //chose the smaller mantisa
  MUX(bigInman, in1mantisaR, in2mantisaR,  negcheck, pubkey_, 13);    //chose the larger mantisa

//-------------------------PART 2-------------------------------
//-----------------------------------
  XOR(expsum*[4], negcheck, pubkey_); //make sure it is positive
///----------------------------- IDK if this is needed, basicly its checking if the value is negative, meaning we could have to shift differently? probs not

  for(i=0; i =< 10; i++){
    if(i < 3){                    //cases for when nothing shifted out past the sticky
      Shift(smallOut[i], smallInman, 10, i);
    }
    else{                     //itterating 0-10 shifts, assigning rounding bits along the way
      Shift(smallOut[i], smallInman, 10, i);
      Copy(smallOut[2][i], smallInman[i-1], pubkey);  //guard
      Copy(smallOut[1][i], smallInman[i-2], pubkey);  //round
      Copy(smallOut[0][i], smallInman[i-3], pubkey);  //stickey

      OR(Ctxt* smallOut[0][i], Ctxt* smallOut[0][i], Ctxt* smallOut[0][i-1], pubkey); //ORing the sticky together
    }
  }
//-----------------------THIS IS THE 10 BIT MUX RIGHT HERE-----------------------------
//  bitMUX(smallInman, smallOut[0], smallOut[1], smallOut[2], smallOut[3], ...ect, expsum, pubkey_, n);
//-----------------------STILL IN PROCESS------------------------------------------

//-----------------------Part 3 ------------------------------
  Add(mantisaSum, co, smallInman, bigInman, 13, pubkey_); //adding the mantisas together
//----------------------Part 4 -------------------------------
  //Normalizing for Carry Out
  roundNormalize(finalSum, tempexpoCo, co, mantisaSumcarryo, mantisaSum, expoOne, tempexpo, pubkey_);
//----------------------Part 5 -------------------------------------
  OR(roundhold, finalSum[2], finalSum[0], pubkey_);
  AND(roundhold, roundhold, finalSum[1], pubkey_);  //if all are 1, then make round temp 1, so you will round

  Add(finalSumRound, co, finalSum, fullOne, pubkey_); //adds one to the answer
  MUX(finalSum, finalSumRound, finalSum, pubkey_); //select the correct finalSum

  //cut off round bits
  removeRound(z, finalSum, 19, pubkey_);
  
  //*insert clause for when mantisa is all ones and need to round up 
}




void roundNormalize(Ctxt* tempexpoCo, Ctxt* mantisaCosum, Ctxt co, Ctxt* mantisaSum, Ctxt* exopoOne, Ctxt* tempexpo, , pubkey_){
  Shift(mantisaCosum, mantisaSum, 10, 1);   // if carry out, shift mantisa right by 1    ***note, may be interpretting "product" wrong in the algorithm description***
  Add(tempexpoCo, tempexpo, expoOne, co, 5, pubkey);  // expoOne is just a 5 bit number 00001 to add to tempexpo

  MUX(tempexpo, tempexpocarry, tempexpo, co, pubkey_, 5); //chose to use the added exponent or not
  MUX(mantisaSum, mantisaCosum, mantisaSum, co, pubkey_, 13); //chosing the correct mantisa
  for(i=0; i<19; i++){
    if(i<13)
      finalSum[i] = mantisaSum[i];
    else
      finalSum[i] = tempexpoCo[i+13];
  }
}

void normShift(Ctxt* smallOut, ctxt* smallIn, n, nshift){

  for(i = 0; i<(n-nshift); i++){
    Copy(smallOut[i], in[nshift+i]);
  }
  for(i=(n-nshift); i<n; i++;){
    Copy(smallOut[i], Ctxt* zero);
  }
}

void Shift(Ctxt* smallOut, ctxt* smallIn, n, nshift){

  for(i = 0; i<(n-nshift); i++){
    Copy(smallOut[i+3], in[nshift+i]);
  }
  for(i=(n-nshift); i<n; i++;){
    Copy(smallOut[i+3], Ctxt* zero);
  }
}

void removeRound(Ctxt* withoutround, Ctxt* withround, n, pubkey_){
  for(i = 3; i < n; i++){
    withoutround[i-3] = withround[i];
  }
}
} // namespace cufhe