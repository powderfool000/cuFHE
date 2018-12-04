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

  for (uint8_t i = 0; i < n; i++) {
    And(p0[i], in0[i], is, st[i%ns]);
    And(p1[i], in1[i], *s, st[i%ns]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], p0[i], p1[i], st[i%ns]);
  }
}

void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  if (n >= 4 && ns >= 9) {
    Csa(z, c, a, b, st, n/2, ns/3);

    Csa(t0, c0, a+n/2, b+n/2, st+ns/3, (n+1)/2, ns/3);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, st+2*ns/3, (n+1)/2, ns/3);
  } else {
    Rca(z, c, a, b, st[0], n/2);

    Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
    Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);
  }

  Mux(z+n/2, t0, t1, c+n/2-1, st, (n+1)/2, ns);
  Mux(c+n/2, c0, c1, c+n/2-1, st, (n+1)/2, ns);
}

void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Stream* st, uint8_t n, uint8_t ns) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  if (n >= 4 && ns >= 9) {
    Csa(z, co, a, b, ci, st, n/2, ns/3);

    Csa(t0, c0, a+n/2, b+n/2, st+ns/3, (n+1)/2, ns/3);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, st+2*ns/3, (n+1)/2, ns/3);
  } else {
    Rca(z, co, a, b, ci, st[0], n/2);

    Rca(t0, c0, a+n/2, b+n/2, st[1], (n+1)/2);
    Rca(t1, c1, a+n/2, b+n/2, &ct_one, st[2], (n+1)/2);
  }

  Mux(z+n/2, t0, t1, co+n/2-1, st, (n+1)/2, ns);
  Mux(co+n/2, c0, c1, co+n/2-1, st, (n+1)/2, ns);
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

} // namespace cufhe
