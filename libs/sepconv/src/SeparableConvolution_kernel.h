/*
 * Created by: Simon Niklaus, Long Mai, Feng Liu
 * https://github.com/sniklaus/pytorch-sepconv
 */

#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
);

void SeparableConvolution_kernel_backward(
    THCState* state,
    THCudaTensor* gradLoss,
    THCudaTensor* input,
    THCudaTensor* vertical,
    THCudaTensor* horizontal,
    THCudaTensor* gradInput,
    THCudaTensor* gradVertical,
    THCudaTensor* gradHorizontal
);

#ifdef __cplusplus
	}
#endif