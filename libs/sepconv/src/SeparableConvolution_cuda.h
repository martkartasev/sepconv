/*
 * Created by: Simon Niklaus, Long Mai, Feng Liu
 * https://github.com/sniklaus/pytorch-sepconv
 */

int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
);