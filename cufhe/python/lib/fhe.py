# file: fhe.py
#
# description: Test either the CPU or GPU implementation of cuFHE
#
# note: This file depends on at least one of either fhepy_gpu.so or fhepy_cpu.so
#

# Attempt to import GPU module, fallback to CPU module
try:
    import lib.fhepy_gpu as fhe
    use_gpu = True
except:
    import lib.fhepy_cpu as fhe
    use_gpu = False

import time
import timeit

def UseGPU():
    return use_gpu

def LoadPubKey(pubfile="pubkey.txt"):
    pubkey = fhe.PubKey()
    fhe.ReadPubKeyFromFile(pubkey, pubfile)
    return pubkey

def LoadPriKey(prifile="prikey.txt"):
    prikey = fhe.PriKey()
    fhe.ReadPriKeyFromFile(prikey, prifile)
    return prikey

def LoadKeys(pubfile="pubkey.txt", prifile="prikey.txt"):
    return LoadPubKey(), LoadPriKey()

def StorePubKey(pubkey, pubfile="pubkey.txt"):
    fhe.WritePubKeyToFile(pubkey, pubfile)
    print("Public key is stored in ./" + pubfile)

def StorePriKey(prikey, prifile="prikey.txt"):
    fhe.WritePriKeyToFile(prikey, prifile)
    print("Private key is stored in ./" + prifile)

def StoreKeys(pubkey, prikey, pubfile="pubkey.txt", prifile="prikey.txt"):
    StorePubKey(pubkey, pubfile)
    StorePriKey(prikey, prifile)

def PriKeyGen():
    prikey = fhe.PriKey()
    fhe.PriKeyGen(prikey)
    return prikey

def PubKeyGen(prikey):
    pubkey = fhe.PubKey()
    fhe.PubKeyGen(pubkey, prikey)
    return pubkey

def KeyGen():
    prikey = PriKeyGen()
    return PubKeyGen(prikey), prikey

def Init(pubkey):
    if use_gpu:
        fhe.Initialize(pubkey)
    else:
        pass

def PtxtMod():
    return fhe.Ptxt().PtxtSpace

def Encrypt(ptxt, prikey, count=1, pubkey=None):
    if pubkey is None:
        pubkey = PubKeyGen(prikey)
    if isinstance(ptxt, (int, long)):
        msg = ptxt
        ptxt = fhe.Ptxt()
        if count == 1:
            ptxt.message = msg;
            ctxt = Ctxt(pubkey)
            fhe.Encrypt(ctxt.ctxt_, ptxt, prikey)
            return ctxt

        msg_bin = bin(msg)[2:].zfill(count)
        msg_list = []
        ct = CtxtList(count, pubkey)
        for i in range(count):
            ptxt.message = int(msg_bin[i], 2)
            fhe.Encrypt(ct.ctxts_[count - i - 1].ctxt_, ptxt, prikey)
        return ct

def Decrypt(ctxt, prikey):
    ptxt = fhe.Ptxt()
    if isinstance(ctxt, Ctxt):
        fhe.Decrypt(ptxt, ctxt.ctxt_, prikey)
        return ptxt.message

    if isinstance(ctxt, CtxtList):
        ptxt_list = ""
        for c in reversed(ctxt.ctxts_):
            fhe.Decrypt(ptxt, c.ctxt_, prikey)
            ptxt_list += str(ptxt.message)
        return int(ptxt_list, 2)

def SetSeed():
    fhe.SetSeed(int(time.time()))

def Synchronize():
    if use_gpu:
        fhe.Synchronize()
    else:
        pass

def AND(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.AND(result, input1, input2, stream)
    else:
        fhe.AND(result, input1, input2, pubkey)

def NAND(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.NAND(result, input1, input2, stream)
    else:
        fhe.NAND(result, input1, input2, stream)

def OR(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.OR(result, input1, input2, stream)
    else:
        fhe.OR(result, input1, input2, pubkey)

def NOR(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.NOR(result, input1, input2, stream)
    else:
        fhe.NOR(result, input1, input2, pubkey)

def XOR(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.XOR(result, input1, input2, stream)
    else:
        fhe.XOR(result, input1, input2, pubkey)

def XNOR(result, input1, input2, stream=None, pubkey=None):
    if use_gpu:
        fhe.XNOR(result, input1, input2, stream)
    else:
        fhe.XNOR(result, input1, input2, pubkey)

def NOT(result, input1, stream=None):
    if use_gpu:
        fhe.NOT(result, input1, stream)
    else:
        fhe.NOT(result, input1)


class Stream:
    def __init__(self):
        if use_gpu:
            self.stream = fhe.Stream()
        else:
            self.stream = None

    def Create(self):
        if use_gpu:
            self.stream.Create()
        else:
            pass

        return self.stream


class Ctxt:
    def __init__(self, pubkey=None):
        self.ctxt_ = fhe.Ctxt()
        self.pubkey_ = pubkey

    def Decrypt(self, prikey):
        return Decrypt(self, prikey)

    def Encrypt(self, msg, prikey):
        Encrypt(msg, prikey, self.pubkey_)
        return self

    def __and__(self, other):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        AND(result.ctxt_, self.ctxt_, other.ctxt_, st, self.pubkey_)
        Synchronize()
        return result

    def __xor__(self, other):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        XOR(result.ctxt_, self.ctxt_, other.ctxt_, st, self.pubkey_)
        Synchronize()
        return result

    def __or__(self, other):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        OR(result.ctxt_, self.ctxt_, other.ctxt_, st, self.pubkey_)
        Synchronize()
        return result

    def __invert__(self):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        NOT(result.ctxt_, self.ctxt_, st)
        Synchronize()
        return result

    def __eq__(self, other):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        XNOR(result.ctxt_, self.ctxt_, other.ctxt_, st, self.pubkey_)
        Synchronize()
        return result

    def __ne__(self, other):
        result = Ctxt(self.pubkey_)
        st = Stream().Create()
        Synchronize()
        XOR(result.ctxt_, self.ctxt_, other.ctxt_, st, self.pubkey_)
        Synchronize()
        return result

    def __lt__(self, other):
        return ~self & other

    def __le__(self, other):
        return ~self | other

    def __gt__(self, other):
        return self & ~other

    def __ge__(self, other):
        return self | ~other


class CtxtList:
    def __init__(self, length=0, pubkey=None):
        self.ctxts_ = [Ctxt(pubkey) for i in range(length)]
        self.pubkey_ = pubkey

    def Decrypt(self, prikey):
        return Decrypt(self, prikey)

    def __and__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = [Stream().Create() for i in range(len(self.ctxts_))]
        Synchronize()
        for i in range(len(self.ctxts_)):
            AND(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i], self.pubkey_)
        Synchronize()
        return result

    def __xor__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = [Stream().Create() for i in range(len(self.ctxts_))]
        Synchronize()
        for i in range(len(self.ctxts_)):
            XOR(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i], self.pubkey_)
        Synchronize()
        return result

    def __or__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = [Stream().Create() for i in range(len(self.ctxts_))]
        Synchronize()
        for i in range(len(self.ctxts_)):
            OR(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i], self.pubkey_)
        Synchronize()
        return result

    def __invert__(self):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = [Stream().Create() for i in range(len(self.ctxts_))]
        Synchronize()
        for i in range(len(self.ctxts_)):
            AND(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, st[i], self.pubkey_)
        Synchronize()
        return result

    # ripple carry adder
    def __add__(self, other):
        x = CtxtList(len(self.ctxts_), self.pubkey_)    # temporaries
        y = CtxtList(len(self.ctxts_), self.pubkey_)
        z = CtxtList(len(self.ctxts_), self.pubkey_)
        c = CtxtList(len(self.ctxts_), self.pubkey_)    # carry
        r = CtxtList(len(self.ctxts_), self.pubkey_)    # result
        st = [Stream().Create() for i in range(2*len(self.ctxts_))]
        Synchronize()
        XOR(r.ctxts_[0].ctxt_, self.ctxts_[0].ctxt_, other.ctxts_[0].ctxt_, st[0], self.pubkey_)
        AND(c.ctxts_[0].ctxt_, self.ctxts_[0].ctxt_, other.ctxts_[0].ctxt_, st[1], self.pubkey_)
        for i in range(1, len(self.ctxts_)):
            XOR(x.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[2*i], self.pubkey_)
            AND(y.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[2*i+1], self.pubkey_)
        Synchronize()
        for i in range(1, len(self.ctxts_)):
            AND(z.ctxts_[i-1].ctxt_, x.ctxts_[i].ctxt_, c.ctxts_[i-1].ctxt_, st[0], self.pubkey_)
            XOR(r.ctxts_[i].ctxt_, x.ctxts_[i].ctxt_, c.ctxts_[i-1].ctxt_, st[1], self.pubkey_)
            Synchronize()
            OR(c.ctxts_[i].ctxt_, z.ctxts_[i-1].ctxt_, y.ctxts_[i].ctxt_, st[0], self.pubkey_)
            Synchronize()
        return r


 #    def __add__(self, other):
 #        k = len(self.ctxts_)
 #        st = []
 #        for i in range(3*k):
 #            st.append(Stream())
 #            st[i].Create()
 #        Synchronize()

 #        ksa_p = CtxtList(k, self.pubkey_)
 #        ksa_g = CtxtList(k, self.pubkey_)
	# ksa_c = CtxtList(k, self.pubkey_)
	# ksa_s = CtxtList(k, self.pubkey_)

 #        for i in range(k):
 #            AND(ksa_g.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i].stream, self.pubkey_)
 #            XOR(ksa_p.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i+1].stream, self.pubkey_)
 #            XOR(ksa_s.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i+2].stream, self.pubkey_)
	# Synchronize()

 #        begin = 0
 #        step = 1
	# while begin+step < k:
	#     for i in range(begin+step, k):
 #                id = i - begin - step
 #                ctxt = ksa_p.ctxts_[i].ctxt_
	#         AND(ksa_p.ctxts_[i].ctxt_, ctxt, ksa_p.ctxts_[i-step].ctxt_, st[2*id].stream, self.pubkey_)
	#         AND(ksa_c.ctxts_[i].ctxt_, ctxt, ksa_g.ctxts_[i-step].ctxt_, st[2*id+1].stream, self.pubkey_)
 #            Synchronize()

	#     for i in range(begin+step, k):
 #                id = i - begin - step
	#         OR(ksa_g.ctxts_[i].ctxt_, ksa_c.ctxts_[i].ctxt_, ksa_g.ctxts_[i].ctxt_, st[id].stream, self.pubkey_)
 #            Synchronize()
 #            step += 1
 #            begin += 1

 #        for i in range(1,k):
 #             XOR(ksa_s.ctxts_[i].ctxt_, ksa_s.ctxts_[i].ctxt_, ksa_g.ctxts_[i-1].ctxt_, st[i].stream, self.pubkey_)
 #        Synchronize()
 #        return ksa_s
