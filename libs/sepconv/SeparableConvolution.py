#
# Created by: Simon Niklaus, Long Mai, Feng Liu
# https://github.com/sniklaus/pytorch-sepconv
#

import torch
import libs.sepconv._ext as _ext
import libs.sepconv._ext.cunnex

class SeparableConvolution(torch.autograd.Function):
    def __init__(self):
        super(SeparableConvolution, self).__init__()
    # end

    @staticmethod
    def forward(context, input, vertical, horizontal):

        context.save_for_backward(input, vertical, horizontal)

        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert(intInputHeight - 51 == intOutputHeight - 1)
        assert(intInputWidth - 51 == intOutputWidth - 1)
        assert(intFilterSize == 51)

        assert(input.is_contiguous() == True)
        assert(vertical.is_contiguous() == True)
        assert(horizontal.is_contiguous() == True)

        output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

        if input.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_forward(
                input,
                vertical,
                horizontal,
                output
            )

        elif input.is_cuda == False:
            raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

        # end

        return output
    # end

    @staticmethod
    def backward(context, grad_output):

        _input, vertical, horizontal = context.saved_tensors

        grad_input = _input.new().resize_(_input.size()).zero_()
        grad_vertical = vertical.new().resize_(vertical.size()).zero_()
        grad_horizontal = horizontal.new().resize_(horizontal.size()).zero_()

        if grad_output.is_cuda:
            _ext.cunnex.SeparableConvolution_cuda_backward(
                grad_output,
                _input,
                vertical,
                horizontal,
                grad_input,
                grad_vertical,
                grad_horizontal
            )
        # end

        return grad_input, grad_vertical, grad_horizontal
    #end
# end
