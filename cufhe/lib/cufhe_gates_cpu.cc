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
#include <include/cufhe_cpu.h>
#include <include/bootstrap_cpu.h>

namespace cufhe {

//void Initialize(PubKey pub_key);
//void And (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
//void Or  (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
//void Xor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n);
void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, PubKey& pub_key, uint8_t n);

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - in0.lwe_sample_->data()[i]
                                   - in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + in0.lwe_sample_->data()[i]
                                   + in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + in0.lwe_sample_->data()[i]
                                   + in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - in0.lwe_sample_->data()[i]
                                   - in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + 2 * in0.lwe_sample_->data()[i]
                                   + 2 * in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - 2 * in0.lwe_sample_->data()[i]
                                   - 2 * in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, pub_key.bk_, pub_key.ksk_);
}

void Not(Ctxt& out,
         const Ctxt& in) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = -in.lwe_sample_->data()[i];
}

void Copy(Ctxt& out,
          const Ctxt& in) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
}

void Ha(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, PubKey& pub_key) {
  Xor(z, a, b, pub_key);
  And(co, a, b, pub_key);
}

void Fa(Ctxt& z, Ctxt& co, const Ctxt& a, const Ctxt& b, const Ctxt& ci, PubKey& pub_key) {
  Ctxt t0, t1, t2;

  Xor(t0, a, b, pub_key);
  And(t1, a, b, pub_key);
  And(t2, ci, t0, pub_key);
  Xor(z, ci, t0, pub_key);
  Or(co, t1, t2, pub_key);
}

void Rca(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  Ha(z[0], c[0], a[0], b[0], pub_key);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], c[i], a[i], b[i], c[i-1], pub_key);
  }
}

void Rca(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, PubKey& pub_key, uint8_t n) {
  Fa(z[0], co[0], a[0], b[0], *ci, pub_key);

  for (uint8_t i = 1; i < n; i++) {
    Fa(z[i], co[i], a[i], b[i], co[i-1], pub_key);
  }
}

void Mux(Ctxt* z, Ctxt* in0, Ctxt* in1, Ctxt* s, PubKey& pub_key, uint8_t n) {
  Ctxt p0[n];
  Ctxt p1[n];
  Ctxt is;

  Not(is, *s);

  for (uint8_t i = 0; i < n; i++) {
    And(p0[i], in0[i], is, pub_key);
    And(p1[i], in1[i], *s, pub_key);
  }

  for (uint8_t i = 0; i < n; i++) {
    Or(z[i], p0[i], p1[i], pub_key);
  }
}

void Csa(Ctxt* z, Ctxt* co, Ctxt* a, Ctxt* b, Ctxt* ci, PubKey& pub_key, uint8_t n) {
  Ctxt t0[(n+1)/2], t1[(n+1)/2];
  Ctxt c0[(n+1)/2], c1[(n+1)/2];

  if(n >= 2){
    Csa(z, co, a, b, ci, pub_key, n/2);

    Csa(t0, c0, a+n/2, b+n/2, pub_key, (n+1)/2);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, pub_key, (n+1)/2);
  }
  else{
  Rca(z, co, a, b, ci, pub_key, n/2);

  Rca(t0, c0, a+n/2, b+n/2, pub_key, (n+1)/2);
  Rca(t1, c1, a+n/2, b+n/2, &ct_one, pub_key, (n+1)/2);
}

  Mux(z+n/2, t0, t1, co+n/2-1, pub_key, (n+1)/2);
  Mux(co+n/2, c0, c1, co+n/2-1, pub_key, (n+1)/2);
}

void Csa(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  Ctxt t0[(n+2)/2], t1[(n+2)/2]; 
  Ctxt c0[(n+2)/2], c1[(n+2)/2];

  if(n >= 2){
    Csa(z, c, a, b, pub_key, n/2);

    Csa(t0, c0, a+n/2, b+n/2, pub_key, (n+1)/2);
    Csa(t1, c1, a+n/2, b+n/2, &ct_one, pub_key, (n+1)/2);
  }
  else{
    Rca(z, c, a, b, pub_key, n/2);

    Rca(t0, c0, a+n/2, b+n/2, pub_key, (n+1)/2);
    Rca(t1, c1, a+n/2, b+n/2, &ct_one, pub_key, (n+1)/2);
}

  Mux(z+n/2, t1, t0, c+n/2-1, pub_key, (n+1)/2);
  Mux(c+n/2, c1, c0, c+n/2-1, pub_key, (n+1)/2);

}


void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  Csa(z, c, a, b, pub_key, n);
}

void Add(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, Ctxt* s, PubKey& pub_key, uint8_t n) {
  Csa(z, c, a, b, s, pub_key, n);
}

void Sub(Ctxt* z, Ctxt* c, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  Ctxt t[n];

  for (uint8_t i = 0; i < n; i++) {
    Not(t[i], b[i]);
  }

  Add(z, c, a, t, &ct_one, pub_key, n);
}

void Mul(Ctxt* z, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
}

// a / b = z
void Div(Ctxt* z, Ctxt* a, Ctxt* b, PubKey& pub_key, uint8_t n) {
  Ctxt r[2*n];      // non-restoring reg
  Ctxt* s = r+n;      // 'working' index
  Ctxt t0[n], t1[n];    // temp
  Ctxt c[n];    // carry
  Ctxt bi[n];   // bi = -b

  // initialize
  for (int i = 0; i < n; i++) {
    Not(bi[i], b[i]);
    Copy(s[i], ct_zero);
    Copy(r[i], a[i]);
  }

  Rca(bi, c, bi, s, &ct_one, pub_key, n);

  // first iteration is always subtract (add bi)
  s--;
  Add(t0, c, s, bi, pub_key, n);

  for (int i = 0; i < n; i++) {
    Copy(s[i], t0[i]);
  }

  Not(z[s-r], s[n-1]);

  while (s > r) {
    s--;
    Add(t0, c, s, bi, pub_key, n);
    Add(t1, c, s, b, pub_key, n);
    Mux(s, t0, t1, s+n, pub_key, n);
    Not(z[s-r], s[n-1]);
  }
}

} // namespace cufhe
