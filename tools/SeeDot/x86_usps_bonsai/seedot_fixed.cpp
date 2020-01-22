#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "library.h"
#include "seedot_fixed_model.h"

using namespace std;
using namespace bonsai_fixed;

int seedotFixed(MYINT **X) {
	MYINT tmp6[28][1];
	MYINT tmp7[28][1];
	MYINT node0;
	MYINT tmp9[10][1];
	MYINT tmp8[28];
	MYINT tmp11[10][1];
	MYINT tmp10[28];
	MYINT tmp12[10][1];
	MYINT tmp14[1][1];
	MYINT tmp13[28];
	MYINT node1;
	MYINT tmp16[10][1];
	MYINT tmp15[28];
	MYINT tmp18[10][1];
	MYINT tmp17[28];
	MYINT tmp19[10][1];
	MYINT tmp20[10][1];
	MYINT tmp22[1][1];
	MYINT tmp21[28];
	MYINT node2;
	MYINT tmp24[10][1];
	MYINT tmp23[28];
	MYINT tmp26[10][1];
	MYINT tmp25[28];
	MYINT tmp27[10][1];
	MYINT tmp28[10][1];
	MYINT tmp30[1][1];
	MYINT tmp29[28];
	MYINT node3;
	MYINT tmp32[10][1];
	MYINT tmp31[28];
	MYINT tmp34[10][1];
	MYINT tmp33[28];
	MYINT tmp35[10][1];
	MYINT tmp36[10][1];
	MYINT tmp37;



	// Z |*| X
	memset(tmp6, 0, sizeof(MYINT) * 28);
	SparseMatMul(&Zidx[0], &Zval[0], X, &tmp6[0][0], 257, 128, 128, 4);


	// tmp6 - mean
	MatSub(&tmp6[0][0], &mean[0][0], &tmp7[0][0], 28, 1, 1, 2, 1);

	node0 = 0;

	// W * ZX
	// W[15][10][28], tmp7[28][1], tmp9[10][1], tmp8[28]
	MatMulCN(&W[node0][0][0], &tmp7[0][0], &tmp9[0][0], &tmp8[0], 10, 28, 1, 128, 64, 0, 5);


	// V * ZX
	// V[15][10][28], tmp7[28][1], tmp11[10][1], tmp10[28]
	MatMulCN(&V[node0][0][0], &tmp7[0][0], &tmp11[0][0], &tmp10[0], 10, 28, 1, 128, 128, 0, 5);


	// tanh(V0)
	TanH(&tmp11[0][0], 10, 1, 4096);


	// W0 <*> V0_tanh
	MulCir(&tmp9[0][0], &tmp11[0][0], &tmp12[0][0], 10, 1, 64, 64);


	// T * ZX
	// T[7][1][28], tmp7[28][1], tmp14[1][1], tmp13[28]
	MatMulCN(&T[node0][0][0], &tmp7[0][0], &tmp14[0][0], &tmp13[0], 1, 28, 1, 128, 128, 0, 5);

	node1 = ((tmp14[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// W * ZX
	// W[15][10][28], tmp7[28][1], tmp16[10][1], tmp15[28]
	MatMulCN(&W[node1][0][0], &tmp7[0][0], &tmp16[0][0], &tmp15[0], 10, 28, 1, 128, 64, 0, 5);


	// V * ZX
	// V[15][10][28], tmp7[28][1], tmp18[10][1], tmp17[28]
	MatMulCN(&V[node1][0][0], &tmp7[0][0], &tmp18[0][0], &tmp17[0], 10, 28, 1, 128, 128, 0, 5);


	// tanh(V1)
	TanH(&tmp18[0][0], 10, 1, 4096);


	// W1 <*> V1_tanh
	MulCir(&tmp16[0][0], &tmp18[0][0], &tmp19[0][0], 10, 1, 64, 64);


	// score0 + tmp19
	MatAdd(&tmp12[0][0], &tmp19[0][0], &tmp20[0][0], 10, 1, 1, 1, 1);


	// T * ZX
	// T[7][1][28], tmp7[28][1], tmp22[10][1], tmp21[28]
	MatMulCN(&T[node1][0][0], &tmp7[0][0], &tmp22[0][0], &tmp21[0], 1, 28, 1, 128, 128, 0, 5);

	node2 = ((tmp22[0][0] > 0) ? ((2 * node1) + 1) : ((2 * node1) + 2));

	// W * ZX
	// W[15][10][28], tmp7[28][1], tmp24[10][1], tmp23[28]
	MatMulCN(&W[node2][0][0], &tmp7[0][0], &tmp24[0][0], &tmp23[0], 10, 28, 1, 128, 64, 0, 5);


	// V * ZX
	// V[15][10][28], tmp7[28][1], tmp26[10][1], tmp25[28]
	MatMulCN(&V[node2][0][0], &tmp7[0][0], &tmp26[0][0], &tmp25[0], 10, 28, 1, 128, 128, 0, 5);


	// tanh(V2)
	TanH(&tmp26[0][0], 10, 1, 4096);


	// W2 <*> V2_tanh
	MulCir(&tmp24[0][0], &tmp26[0][0], &tmp27[0][0], 10, 1, 64, 64);


	// score1 + tmp27
	MatAdd(&tmp20[0][0], &tmp27[0][0], &tmp28[0][0], 10, 1, 1, 1, 1);


	// T * ZX
	// V[7][1][28], tmp7[28][1], tmp30[1][1], tmp29[28]
	MatMulCN(&T[node2][0][0], &tmp7[0][0], &tmp30[0][0], &tmp29[0], 1, 28, 1, 128, 128, 0, 5);

	node3 = ((tmp30[0][0] > 0) ? ((2 * node2) + 1) : ((2 * node2) + 2));

	// W * ZX
	// W[15][10][28], tmp7[28][1], tmp32[10][1], tmp31[28]
	MatMulCN(&W[node3][0][0], &tmp7[0][0], &tmp32[0][0], &tmp31[0], 10, 28, 1, 128, 64, 0, 5);


	// V * ZX
	// V[15][10][28], tmp7[28][1], tmp34[10][1], tmp33[28]
	MatMulCN(&V[node3][0][0], &tmp7[0][0], &tmp34[0][0], &tmp33[0], 10, 28, 1, 128, 128, 0, 5);


	// tanh(V3)
	TanH(&tmp34[0][0], 10, 1, 4096);


	// W3 <*> V3_tanh
	MulCir(&tmp32[0][0], &tmp34[0][0], &tmp35[0][0], 10, 1, 64, 64);


	// score2 + tmp35
	MatAdd(&tmp28[0][0], &tmp35[0][0], &tmp36[0][0], 10, 1, 1, 1, 1);


	// argmax(score3)
	ArgMax(&tmp36[0][0], 10, 1, &tmp37);


	return tmp37;
}
