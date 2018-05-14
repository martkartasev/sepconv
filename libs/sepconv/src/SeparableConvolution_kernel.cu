/*
 * Created by: Simon Niklaus, Long Mai, Feng Liu
 * https://github.com/sniklaus/pytorch-sepconv
 */

#include <THC.h>
#include <THCGeneral.h>

#define FILTER_LENGTH 51

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

__global__ void kernel_SeparableConvolution_updateOutput(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterY = 0; intFilterY < 51; intFilterY += 1) {
		for (int intFilterX = 0; intFilterX < 51; intFilterX += 1) {
			dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(vertical, intBatch, intFilterY, intY, intX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
		}
	}

	output[intIndex] = dblOutput;
}

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}

// Code from here to the bottom from: https://github.com/ekgibbons/pytorch-sepconv

__global__ void kernel_SeparableConvolution_updateGradVertical(
    const int n,
    const float* gradLoss, const long4 gradLoss_stride,
    const float* input, const long4 input_stride,
    const float* horizontal, const long4 horizontal_stride,
    float* gradVertical, const long4 gradVertical_size,
    const long4 gradVertical_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	// Determine where we are with each dimension
	int intBatch   = ( intIndex / VEC_3(gradVertical_size) / VEC_2(gradVertical_size) / VEC_1(gradVertical_size) ) % VEC_0(gradVertical_size);
	int intFilterY = ( intIndex / VEC_3(gradVertical_size) / VEC_2(gradVertical_size)                            ) % VEC_1(gradVertical_size);
	int intY       = ( intIndex / VEC_3(gradVertical_size)                                                       ) % VEC_2(gradVertical_size);
	int intX       = ( intIndex                                                                                  ) % VEC_3(gradVertical_size);

	float floatOutput = 0.0;

	for (int intFilterX = 0; intFilterX < FILTER_LENGTH; intFilterX++) {
	    floatOutput += IDX_4(gradLoss, intBatch, 0, intY, intX)*              // channel 0
		IDX_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)*
		IDX_4(horizontal, intBatch, intFilterX, intY, intX) +
		IDX_4(gradLoss, intBatch, 1, intY, intX)*                          // channel 1
		IDX_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)*
		IDX_4(horizontal, intBatch, intFilterX, intY, intX) +
		IDX_4(gradLoss, intBatch, 2, intY, intX)*                          // channel 2
		IDX_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)*
		IDX_4(horizontal, intBatch, intFilterX, intY, intX);
	}

	gradVertical[intIndex] = floatOutput;
}


__global__
void kernel_SeparableConvolution_updateGradHorizontal(
	const int n,
	const float* gradLoss, const long4 gradLoss_stride,
	const float* input, const long4 input_stride,
	const float* vertical, const long4 vertical_stride,
	float* gradHorizontal, const long4 gradHorizontal_size, const long4 gradHorizontal_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	int intBatch   = ( intIndex / VEC_3(gradHorizontal_size) / VEC_2(gradHorizontal_size) / VEC_1(gradHorizontal_size) ) % VEC_0(gradHorizontal_size);
	int intFilterX = ( intIndex / VEC_3(gradHorizontal_size) / VEC_2(gradHorizontal_size)                              ) % VEC_1(gradHorizontal_size);
	int intY       = ( intIndex / VEC_3(gradHorizontal_size)                                                           ) % VEC_2(gradHorizontal_size);
	int intX       = ( intIndex                                                                                        ) % VEC_3(gradHorizontal_size);

	float floatOutput = 0.0;

	// Because the matrix needs to be transposed...
	for (int intFilterY = 0; intFilterY < FILTER_LENGTH; intFilterY++) {
	    floatOutput += IDX_4(gradLoss, intBatch, 0, intY, intX)*             // channel 0
		IDX_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)*
		IDX_4(vertical, intBatch, intFilterY, intY, intX) +
		IDX_4(gradLoss, intBatch, 1, intY, intX)*                         // channel 1
		IDX_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)*
		IDX_4(vertical, intBatch, intFilterY, intY, intX) +
		IDX_4(gradLoss, intBatch, 2, intY, intX)*                         // channel 2
		IDX_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)*
		IDX_4(vertical, intBatch, intFilterY, intY, intX);
	}

	gradHorizontal[intIndex] = floatOutput;
}

void SeparableConvolution_kernel_backward(
    THCState* state,
    THCudaTensor* gradLoss,
    THCudaTensor* input,
    THCudaTensor* vertical,
    THCudaTensor* horizontal,
    THCudaTensor* gradInput,
    THCudaTensor* gradVertical,
    THCudaTensor* gradHorizontal
) {
    int n = 0;

    cudaStream_t stream = THCState_getCurrentStream(state);

    n = THCudaTensor_nElement(state, gradVertical);
    kernel_SeparableConvolution_updateGradVertical<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
	n,
	THCudaTensor_data(state, gradLoss),
	make_long4(gradLoss->stride[0], gradLoss->stride[1],
		   gradLoss->stride[2], gradLoss->stride[3]),
	THCudaTensor_data(state, input),
	make_long4(input->stride[0], input->stride[1],
		   input->stride[2], input->stride[3]),
	THCudaTensor_data(state, horizontal),
	make_long4(horizontal->stride[0], horizontal->stride[1],
		   horizontal->stride[2], horizontal->stride[3]),
	THCudaTensor_data(state, gradVertical),
	make_long4(gradVertical->size[0], gradVertical->size[1],
		   gradVertical->size[2], gradVertical->size[3]),
	make_long4(gradVertical->stride[0], gradVertical->stride[1],
		   gradVertical->stride[2], gradVertical->stride[3])
	);

    n = THCudaTensor_nElement(state, gradHorizontal);
    kernel_SeparableConvolution_updateGradHorizontal<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
    	n,
    	THCudaTensor_data(state, gradLoss),
    	make_long4(gradLoss->stride[0], gradLoss->stride[1],
    		   gradLoss->stride[2], gradLoss->stride[3]),
    	THCudaTensor_data(state, input),
    	make_long4(input->stride[0], input->stride[1],
    		   input->stride[2], input->stride[3]),
    	THCudaTensor_data(state, vertical),
    	make_long4(vertical->stride[0], vertical->stride[1],
    		   vertical->stride[2], vertical->stride[3]),
    	THCudaTensor_data(state, gradHorizontal),
    	make_long4(gradHorizontal->size[0], gradHorizontal->size[1],
    		   gradHorizontal->size[2], gradHorizontal->size[3]),
    	make_long4(gradHorizontal->stride[0], gradHorizontal->stride[1],
    		   gradHorizontal->stride[2], gradHorizontal->stride[3])

    	);

    THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif