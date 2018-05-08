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

#ifdef __cplusplus
	}
#endif