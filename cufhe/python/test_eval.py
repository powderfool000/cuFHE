################################################################################
# Copyright 2018 Gizem S. Cetin <gscetin@wpi.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import lib.fhe as fhe
import random
import operator
import timeit

def CheckResult(m, op, result):
        return op(m[0], m[1]) == result


pubkey, prikey = fhe.KeyGen()
#m = [random.randint(0,1), random.randint(0,1), random.randint(0,1)]
#c0, c1, s = fhe.Encrypt(m[0], prikey), fhe.Encrypt(m[1], prikey), fhe.Encrypt(m[2], prikey)
#inp1, inp2 = fhe.Encrypt(0, prikey), fhe.Encrypt(1, prikey)
inp = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s = [ 1, 1, 0, 0]
inp_c = [fhe.Encrypt(inp[0], prikey), fhe.Encrypt(inp[1], prikey), fhe.Encrypt(inp[2], prikey),fhe.Encrypt(inp[3], prikey),fhe.Encrypt(inp[4], prikey),fhe.Encrypt(inp[5], prikey),fhe.Encrypt(inp[6], prikey),fhe.Encrypt(inp[7], prikey), fhe.Encrypt(inp[8], prikey),fhe.Encrypt(inp[9], prikey),fhe.Encrypt(inp[10], prikey),fhe.Encrypt(inp[11], prikey),fhe.Encrypt(inp[12], prikey),fhe.Encrypt(inp[13], prikey),fhe.Encrypt(inp[14], prikey),fhe.Encrypt(inp[15], prikey)]

s_c = [fhe.Encrypt(s[0], prikey), fhe.Encrypt(s[1], prikey), fhe.Encrypt(s[2], prikey), fhe.Encrypt(s[3], prikey)]

#MUX
def MUX(inp1, inp2, sel):
	value1 = inp1 & ~sel
	value2 = inp2 & sel
	result = value1 | value2
	return result

#n_MUX
def n_MUX(inp, sel):
	start_time = timeit.default_timer()
	out = [0,0,0,0,0,0,0,0]
	out1 = [0,0,0,0]
	out2 = [0,0]
	
	for i in range (0,8):
		out[i] = MUX(inp[i], inp[8 + i], sel[3])
	for i in range (0,4):
		out1[i] = MUX(out[i], out[4+i], sel[2])
	for i in range (0,2):
		out2[i] = MUX(out1[i], out1[2+i], sel[1])
	out3 = MUX(out2[0], out2[1], sel[0])
	elapsed = timeit.default_timer() - start_time
	print elapsed, "sec"
	return out3


#Test n_MUX
print "result :", n_MUX(inp_c, s_c).Decrypt(prikey)

#inp1_p = inp1.Decrypt(prikey)
#inp2_p = inp2.Decrypt(prikey)
#s2 = s.Decrypt(prikey)
#out2 = out.Decrypt(prikey)
#print "c0: ", inp1_p
#print "c1: ", inp2_p
#print "s: ", s2
#print "out: ", MUX(inp1, inp2, s).Decrypt(prikey)

# AND Gate
#c2 = c0 & c1
#result = c2.Decrypt(prikey)
#print("AND gate : " + str(CheckResult(m, operator.__and__, result)))

# XOR Gate
#c3 = c0 ^ c1
#result = c3.Decrypt(prikey)
#print("XOR gate : " + str(CheckResult(m, operator.__xor__, result)))

# OR Gate
#c4 = c0 | c1
#result = c4.Decrypt(prikey)
#print("OR gate : " + str(CheckResult(m, operator.__or__, result)))

# NOT Complement
#c5 = ~c0
#result = c5.Decrypt(prikey)
#rint("NOT gate : " + str(result != m[0]))

# NAND Gate
#c6 = c0 & c1
#c7 = ~c6
#result = c7.Decrypt(prikey)
#print("NAND gate : " + str(not CheckResult(m, operator.__and__, result)))
