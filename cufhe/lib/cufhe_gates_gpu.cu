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
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)c.lock_), 0);
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
  cudaEventRecord(*((cudaEvent_t*)c.lock_), st.st());
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

// Requires 3 temporary ctxts
void Fa(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, const Ctxt& ci, Ctxt* t, Stream& st) {
  Xor(t[0], a, b, st);
  And(t[1], a, b, st);
  And(t[2], ci, t[0], st);
  Xor(z, ci, t[0], st);
  Or(co, t[1], t[2], st);
}

// Requires 3 temporary ctxts
void Rca(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream& st, uint8_t n) {
  Ha(z[0], c[0], a[0], b[0], st);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], c[i], a[i], b[i], c[i-1], t, st);
  }
}

// Requires 3 temporary ctxts
void Rca(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Ctxt* t, Stream& st, uint8_t n) {
  Fa(z[0], co[0], a[0], b[0], *ci, t, st);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], co[i], a[i], b[i], co[i-1], t, st);
  }
}

// Requires 2n+1 temporary ctxts
void Mux(Ctxt* z, Ctxt* in0, Ctxt* in1, Ctxt* s, Ctxt* t, Stream* st, uint8_t n) {
  Not(t[0], *s, st[0]);

  for (uint8_t i = 0; i < n; i++) {
    And(t[i+1], in0[i], t[0], st[i]);
    And(t[n+i+1], in1[i], *s, st[i]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], t[i+1], t[n+i+1], st[i]);
  }
}

// Requires 4n+1 temporary ctxts
void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n) {
  // Ctxt t0[(n+1)/2], t1[(n+1)/2];
  // Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Ctxt* t0 = t;
  Ctxt* t1 = t+(n+1)/2;
  Ctxt* c0 = t+n;
  Ctxt* c1 = c0+(n+1)/2;
  Ctxt* rcat = c0+n;

  Rca(z, c, a, b, rcat, st[0], n/2);

  Rca(t0, c0, a+n/2, b+n/2, rcat+3, st[1], (n+1)/2);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, rcat+6, st[2], (n+1)/2);

  Mux(z+n/2, t0, t1, c+n/2-1, rcat, st, (n+1)/2);
  Mux(c+n/2, c0, c1, c+n/2-1, rcat, st, (n+1)/2);
}

// Requires 4n+1 temporary ctxts
void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Ctxt* t, Stream* st, uint8_t n) {
  // Ctxt t0[(n+1)/2], t1[(n+1)/2];
  // Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Ctxt* t0 = t;
  Ctxt* t1 = t+(n+1)/2;
  Ctxt* c0 = t+n;
  Ctxt* c1 = c0+(n+1)/2;
  Ctxt* rcat = c0+n;

  Rca(z, co, a, b, ci, rcat, st[0], n/2);

  Rca(t0, c0, a+n/2, b+n/2, rcat+3, st[1], (n+1)/2);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, rcat+6, st[2], (n+1)/2);

  Mux(z+n/2, t0, t1, co+n/2-1, rcat, st, (n+1)/2);
  Mux(co+n/2, c0, c1, co+n/2-1, rcat, st, (n+1)/2);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n) {
  Csa(z, c, a, b, t, st, n);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* s, Ctxt* t, Stream* st, uint8_t n) {
  Csa(z, c, a, b, s, t, st, n);
}

// Requires 5n+1
void Sub(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n) {
  for (uint8_t i = 0; i < n; i++) {
    Not(t[i], b[i], st[i]);
  }

  Add(z, c, a, t, &ct_one, t+n, st, n);
}

void Mul(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n) {
}

// a / b = z
// Requires
void Div(Ctxt* z, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n) {
  Ctxt* r = t+4*n+1;      // non-restoring reg
  Ctxt* s = r+n;      // 'working' index
  Ctxt* t0 = r+2*n;
  Ctxt* t1 = t0 + n;    // temp
  Ctxt* c = t1 + n;    // carry
  Ctxt* bi = c + n;   // bi = -b

  // initialize
  for (int i = 0; i < n; i++) {
    Not(bi[i], b[i], st[i]);
    Copy(s[i], ct_zero, st[i]);
    Copy(r[i], a[i], st[i]);
  }

  Add(bi, c, bi, s, &ct_one, t, st, n);

  // first iteration is always subtract (add bi)
  s--;
  Add(t0, c, s, bi, t, st, n);

  Synchronize();

  for (int i = 0; i < n; i++) {
    Copy(s[i], t0[i], st[i]);
  }

  Not(z[s-r], s[n-1], st[0]);

  Synchronize();

  while (s > r) {
    s--;
    Add(t0, c, s, bi, t, st, n);
    Add(t1, c, s, b, t, st, n);

    Mux(s, t0, t1, s+n, t, st, n);

    Not(z[s-r], s[n-1]);
  }
}

} // namespace cufhe
