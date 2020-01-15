#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "library.h"
#include "seedot_fixed_model.h"

using namespace std;
using namespace protonn_fixed;

const MYINT EXP12A[64] = {
	8192, 7695, 7229, 6791, 6379, 5993, 5630, 5289, 4968, 4667, 4384, 4119, 3869, 3635, 3414, 3208, 3013, 2831, 2659, 2498, 2347, 2204, 2071, 1945, 1827, 1717, 1613, 1515, 1423, 1337, 1256, 1180, 1108, 1041, 978, 919, 863, 811, 761, 715, 672, 631, 593, 557, 523, 491, 462, 434, 407, 383, 359, 338, 317, 298, 280, 263, 247, 232, 218, 205, 192, 180, 170, 159, 
};
const MYINT EXP12B[64] = {
	16009, 15994, 15978, 15962, 15947, 15931, 15916, 15900, 15885, 15869, 15854, 15838, 15823, 15807, 15792, 15776, 15761, 15746, 15730, 15715, 15700, 15684, 15669, 15654, 15638, 15623, 15608, 15593, 15577, 15562, 15547, 15532, 15517, 15501, 15486, 15471, 15456, 15441, 15426, 15411, 15396, 15381, 15366, 15351, 15336, 15321, 15306, 15291, 15276, 15261, 15246, 15231, 15216, 15202, 15187, 15172, 15157, 15142, 15128, 15113, 15098, 15083, 15069, 15054, 
};

int seedotFixed(MYINT **X) {
	MYINT tmp4;
	MYINT tmp5[25][1];
	MYINT i;
	MYINT tmp6[25][1];
	MYINT tmp7;
	MYINT tmp8[1][25];
	MYINT tmp10[1][1];
	MYINT tmp9[25];
	MYINT tmp11[1][1];
	MYINT tmp15[1][1];
	MYINT tmp12;
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp17[10][1];
	MYINT tmp16[1];
	MYINT tmp18[10][1];
	MYINT tmp19;

	tmp4 = 9126;


	// W |*| X
	memset(tmp5, 0, sizeof(MYINT) * 25);
	SparseMatMul(&Widx[0], &Wval[0], X, &tmp5[0][0], 256, 128, 128, 8);

	memset(tmp18, 0, sizeof(MYINT) * 10);
	i = 0;
	for (MYINT i0 = 0; (i0 < 55); i0++) {

		// WX - B
		MatSub(&tmp5[0][0], &B[i][0][0], &tmp6[0][0], 25, 1, 1, 8, 1);

		tmp7 = (-tmp4);

		// del^T
		Transpose(&tmp6[0][0], &tmp8[0][0], 1, 25);


		// tmp8 * del
		MatMulNN(&tmp8[0][0], &tmp6[0][0], &tmp10[0][0], &tmp9[0], 1, 25, 1, 128, 64, 0, 5);


		// tmp7 * tmp10
		ScalarMul(&tmp7, &tmp10[0][0], &tmp11[0][0], 1, 1, 128, 128);


		// exp(tmp11)
		if (((-tmp11[0][0]) < 94)) {
			tmp13 = 0;
			tmp14 = 0;
		} else {
			tmp12 = (((-tmp11[0][0]) - 94) << 2);
			tmp13 = ((tmp12 >> 10) & 63);
			tmp14 = ((tmp12 >> 4) & 63);
		}
		tmp15[0][0] = ((EXP12A[tmp13] >> 7) * (EXP12B[tmp14] >> 7));

		// Z * tmp15
		MatMulCN(&Z[i][0][0], &tmp15[0][0], &tmp17[0][0], &tmp16[0], 10, 1, 1, 128, 128, 0, 0);

		for (MYINT i1 = 0; (i1 < 10); i1++) {
			for (MYINT i2 = 0; (i2 < 1); i2++) {
				tmp18[i1][i2] = (tmp18[i1][i2] + (tmp17[i1][i2] / 32));
			}
		}
		i = (i + 1);
	}

	// argmax(res)
	ArgMax(&tmp18[0][0], 10, 1, &tmp19);


	return tmp19;
}
