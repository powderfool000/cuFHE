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
  // cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)c.lock_), 0);
  // if (!*(c.on_device_)) {
    // *c.on_device_ = true;
    cudaMemcpyAsync(c.lwe_sample_device_->data(),
                    c.lwe_sample_->data(),
                    c.lwe_sample_->SizeData(),
                    cudaMemcpyHostToDevice,
                    st.st());
  // }
  // cudaEventRecord(*((cudaEvent_t*)c.lock_), st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st) {
  // cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)c.lock_), 0);
  cudaMemcpyAsync(c.lwe_sample_->data(),
                    c.lwe_sample_device_->data(),
                    c.lwe_sample_->SizeData(),
                    cudaMemcpyDeviceToHost,
                    st.st());
  // cudaEventRecord(*((cudaEvent_t*)c.lock_), st.st());
}

inline void CtxtCopyD2D(Ctxt& out,
                        const Ctxt& in,
                        Stream st) {
  // cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in.lock_), 0);
  // cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.lock_), 0);
  cudaMemcpyAsync(out.lwe_sample_device_->data(),
                  in.lwe_sample_device_->data(),
                  in.lwe_sample_device_->SizeData(),
                  cudaMemcpyDeviceToDevice,
                  st.st());
  // *out.on_device_ = true;
  // cudaEventRecord(*((cudaEvent_t*)out.lock_), st.st());
}

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  // CtxtCopyH2D(in0, st);
  // CtxtCopyH2D(in1, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in0.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in1.wlock_), 0);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in0.rlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in1.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

__global__
void __NotBootstrap__(Torus* out, Torus* in) {
  #pragma unroll
  for (int i = 0; i <= 500; i++) {
    out[i] = -in[i];
  }
  __syncthreads();
}

void Not(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  // CtxtCopyH2D(in, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in.wlock_), 0);
  __NotBootstrap__<<<1, 1, 0, st.st()>>>(out.lwe_sample_device_->data(), in.lwe_sample_device_->data());
  // *out.on_device_ = true;
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Copy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  // CtxtCopyH2D(in, st);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.wlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)out.rlock_), 0);
  cudaStreamWaitEvent(st.st(), *((cudaEvent_t*)in.wlock_), 0);
  CtxtCopyD2D(out, in, st);
  // *out.on_device_ = true;
  cudaEventRecord(*((cudaEvent_t*)out.wlock_), st.st());
  cudaEventRecord(*((cudaEvent_t*)in.rlock_), st.st());
  // CtxtCopyD2H(out, st);
}

void Ha(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, Stream& st, bool memcpy = true) {
  Xor(z, a, b, st);
  And(co, a, b, st);
}

// Requires 3 temporary ctxts
void Fa(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, const Ctxt& ci, Ctxt* t, Stream& st, bool memcpy = true) {
  Xor(t[0], a, b, st);
  And(t[1], a, b, st);
  And(t[2], ci, t[0], st);
  Xor(z, ci, t[0], st);
  Or(co, t[1], t[2], st);
}

// Requires 3 temporary ctxts
void Rca(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy = true) {
  // Ha(z[0], c[0], a[0], b[0], st);

  // for (uint8_t i = 1; i < n; i++) {
  //   Fa(z[i], c[i], a[i], b[i], c[i-1], t, st);
  // }
  uint stn = 2;

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }
  }

  Xor(z[0], a[0], b[0], st[0]);
  And(c[0], a[0], b[0], st[1%ns]);

  for (uint8_t i = 1; i < n; i++) {
    Xor(t[0], a[i], b[i], st[(stn++)%ns]);
    And(t[1], a[i], b[i], st[(stn++)%ns]);
    And(t[2], c[i-1], t[0], st[(stn++)%ns]);
    Xor(z[i], c[i-1], t[0], st[(stn++)%ns]);
    Or(c[i], t[1], t[2], st[(stn++)%ns]);
  }

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(c[i], st[(i+n)%ns]);
    }
  }
}

// Requires 3 temporary ctxts
void Rca(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy = true) {
  // Fa(z[0], co[0], a[0], b[0], *ci, t, st);

  // for (uint8_t i = 1; i < n; i++) {
  //   Fa(z[i], co[i], a[i], b[i], co[i-1], t, st);
  // }
  uint stn = 5;

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(*ci, st[0]);
  }

  Xor(t[0], a[0], b[0], st[0]);
  And(t[1], a[0], b[0], st[1%ns]);
  And(t[2], *ci, t[0], st[2%ns]);
  Xor(z[0], *ci, t[0], st[3%ns]);
  Or(co[0], t[1], t[2], st[4%ns]);

  // And(z[0], a[0], b[0], st[0]);
  // Xor(co[0], a[0], b[0], st[1%ns]);

  for (uint8_t i = 1; i < n; i++) {
    Xor(t[0], a[i], b[i], st[(stn++)%ns]);
    And(t[1], a[i], b[i], st[stn++%ns]);
    And(t[2], co[i-1], t[0], st[(stn++)%ns]);
    Xor(z[i], co[i-1], t[0], st[(stn++)%ns]);
    Or(co[i], t[1], t[2], st[(stn++)%ns]);
  }

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(co[i], st[(i+n)%ns]);
    }
  }
}

// Requires 2n+1 temporary ctxts
void Mux(Ctxt* z, Ctxt* in0, Ctxt* in1, Ctxt* s, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(in0[i], st[i%ns]);
      CtxtCopyH2D(in1[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(*s, st[0]);
  }

  Not(t[0], *s, st[0]);

  for (uint8_t i = 0; i < n; i++) {
    And(t[i+1], in0[i], t[0], st[i%ns]);
    And(t[n+i+1], in1[i], *s, st[(i+n)%ns]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], t[i+1], t[n+i+1], st[i%ns]);
  }

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
    }
  }
}

// Requires 2n + max(9, 4n+2) temporary ctxts
void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy = true) {
  // Ctxt t0[(n+1)/2], t1[(n+1)/2];
  // Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Ctxt* t0 = t;
  Ctxt* t1 = t+(n+1)/2;
  Ctxt* c0 = t+n;
  Ctxt* c1 = c0+(n+1)/2;
  Ctxt* rcat = c0+n;

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_one, st[0]);
  }

  Rca(z, c, a, b, rcat, st, n/2, 2, false);

  Rca(t0, c0, a+n/2, b+n/2, rcat+3, st+2, (n+1)/2, 2, false);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, rcat+6, st+4, (n+1)/2, 2, false);

  // Synchronize();

  Mux(z+n/2, t0, t1, c+n/2-1, rcat, st, (n+1)/2, ns, false);
  Mux(c+n/2, c0, c1, c+n/2-1, rcat, st, (n+1)/2, ns, false);

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(c[i], st[(i+n)%ns]);
    }
  }
}

// Requires 2n + max(9, 2n+1) temporary ctxts
void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy = true) {
  // Ctxt t0[(n+1)/2], t1[(n+1)/2];
  // Ctxt c0[(n+1)/2], c1[(n+1)/2];

  Ctxt* t0 = t;
  Ctxt* t1 = t+(n+1)/2;
  Ctxt* c0 = t+n;
  Ctxt* c1 = c0+(n+1)/2;
  Ctxt* rcat = c0+n;

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_one, st[0]);
    CtxtCopyH2D(*ci, st[1]);
  }

  Rca(z, co, a, b, ci, rcat, st, n/2, 2, false);

  Rca(t0, c0, a+n/2, b+n/2, rcat+3, st+2, (n+1)/2, 2, false);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, rcat+6, st+4, (n+1)/2, 2, false);

  Mux(z+n/2, t0, t1, co+n/2-1, rcat, st, (n+1)/2, ns, false);
  Mux(co+n/2, c0, c1, co+n/2-1, rcat, st, (n+1)/2, ns, false);

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(co[i], st[(i+n)%ns]);
    }
  }
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_one, st[0]);
  }

  // Rca(z, c, a, b, t, st, n, ns);
  Csa(z, c, a, b, t, st, n, ns, false);

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(c[i], st[(i+n)%ns]);
    }
  }
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* s, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_one, st[0]);
    CtxtCopyH2D(*s, st[1]);
  }

  // Rca(z, c, a, b, s, t, st, n, ns);
  Csa(z, c, a, b, s, t, st, n, ns, false);

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(c[i], st[(i+n)%ns]);
    }
  }
}

// Requires 5n+1
void Sub(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_one, st[0]);
  }

  for (uint8_t i = 0; i < n; i++) {
    Not(t[i], b[i], st[i%ns]);
  }

  Add(z, c, a, t, &ct_one, t+n, st, n, ns, false);

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
      CtxtCopyD2H(c[i], st[(i+n)%ns]);
    }
  }
}

void Mul(Ctxt* z, Ctxt* a, Ctxt* b, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
}

// a / b = z
// Requires
void Div(Ctxt* z, Ctxt* a, Ctxt* b, Ctxt* t, Stream* st, uint8_t n, uint8_t ns, bool memcpy) {
  Ctxt* r = t+4*n+1;      // non-restoring reg
  Ctxt* s = r+n;      // 'working' index
  Ctxt* t0 = r+2*n;
  Ctxt* t1 = t0 + n;    // temp
  Ctxt* c = t1 + n;    // carry
  Ctxt* bi = c + n;   // bi = -b

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyH2D(a[i], st[i%ns]);
      CtxtCopyH2D(b[i], st[(i+n)%ns]);
    }

    CtxtCopyH2D(ct_zero, st[0]);
    CtxtCopyH2D(ct_one, st[1%ns]);
  }

  Synchronize();

  // initialize
  for (int i = 0; i < n; i++) {
    Not(bi[i], b[i], st[i%ns]);
    Copy(s[i], ct_zero, st[(i+n)%ns]);
    Copy(r[i], a[i], st[(i+2*n)%ns]);
  }

  // Synchronize();

  Add(bi, c, bi, s, &ct_one, t, st, n, ns, false);

  // Synchronize();

  // first iteration is always subtract (add bi)
  s--;
  Add(t0, c, s, bi, t, st, n, ns, false);

  // Synchronize();

  for (int i = 0; i < n; i++) {
    Copy(s[i], t0[i], st[i%ns]);
  }

  // Synchronize();

  Not(z[s-r], s[n-1], st[0]);

  // Synchronize();

  while (s > r) {
    s--;
    Add(t0, c, s, bi, t, st, n, ns, false);
    // Synchronize();
    Add(t1, c, s, b, t, st, n, ns, false);
    // Synchronize();
    Mux(s, t0, t1, s+n, t, st, n, ns, false);
    // Synchronize();
    Not(z[s-r], s[n-1], st[0]);
    // Synchronize();
  }

  Synchronize();

  if (memcpy) {
    for (int i = 0; i < n; i++) {
      CtxtCopyD2H(z[i], st[i%ns]);
    }
  }
}

} // namespace cufhe
