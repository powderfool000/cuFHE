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

  Synchronize();

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], c[i], a[i], b[i], c[i-1], st);
  }

  Synchronize();
}

void Rca(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream& st, uint8_t n) {
  Fa(z[0], co[0], a[0], b[0], *ci, st);

  Synchronize();

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], co[i], a[i], b[i], co[i-1], st);
  }

  Synchronize();
}

void Mux(Ctxt* z, Ctxt* in0, Ctxt* in1, Ctxt* s, Stream* st, uint8_t n) {
  Ctxt p0[n];
  Ctxt p1[n];
  Ctxt is;

  Synchronize();

  Not(is, *s, st[0]);

  Synchronize();

  for (uint8_t i = 0; i < n; i++) {
    And(p0[i], in0[i], is, st[i]);
    And(p1[i], in1[i], *s, st[i]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], p0[i], p1[i], st[i]);
  }

  Synchronize();
}

void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Synchronize();

  Rca(z, c, a, b, st[0], n/2);

  Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);

  Synchronize();

  Mux(z+n/2, t0, t1, c+n/2-1, st, (n+1)/2);
  Mux(c+n/2, c0, c1, c+n/2-1, st, (n+1)/2);

  Synchronize();
}

void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream* st, uint8_t n) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Synchronize();

  Rca(z, co, a, b, ci, st[0], n/2);

  Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);

  Synchronize();

  Mux(z+n/2, t0, t1, co+n/2-1, st, (n+1)/2);
  Mux(co+n/2, c0, c1, co+n/2-1, st, (n+1)/2);

  Synchronize();
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
  Csa(z, c, a, b, st, n);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* s, Stream* st, uint8_t n) {
  Csa(z, c, a, b, s, st, n);
}

void Sub(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
  Ctxt t[n];

  Synchronize();

  for (uint8_t i = 0; i < n; i++) {
    Not(t[i], b[i], st[i]);
  }

  Synchronize();

  Add(z, c, a, t, &ct_one, st, n);

  Synchronize();
}

void Mul(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
}

// a / b = z
void Div(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
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

  Add(bi, c, bi, s, &ct_one, st, n);

  Synchronize();

  // first iteration is always subtract (add bi)
  s--;
  Add(t0, c, s, bi, st, n);

  Synchronize();

  for (int i = 0; i < n; i++) {
    Copy(s[i], t0[i], st[i]);
  }

  Synchronize();

  Not(z[s-r], s[n-1], st[0]);

  Synchronize();

  while (s > r) {
    s--;
    Add(t0, c, s, bi, st, n);
    Add(t1, c, s, b, st, n);

    Synchronize();

    Mux(s, t0, t1, s+n, st, n);

    Synchronize();

    Not(z[s-r], s[n-1]);

    Synchronize();
  }
}

void FpAdd(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st) {
  Ctxt es[5], ec[5];
  Ctxt lgm[14], smm[14];

  Sub(es, ec, a+10, b+10, st, 5); // Get difference of exponents

  Mux(lgm+3, a, b, es, st, 10); // Select mantissa of larger number
  Mux(smm+3, b, a, es, st, 10); // Select mantissa of smaller number

  for (int i = 0; i < 3; i++) { // Zero out guard round and sticky
    Copy(lgm[i], ct_zero);
    Copy(smm[i], ct_zero);
  }

  Copy(lgm[13], ct_one); // Assume normalized for now
  Copy(smm[13], ct_one);

  // Shift smaller mantissa

  // Add mantissas

  // Normalize
}

} // namespace cufhe
