# Video Frame Interpolation


## License

This project is released under the MIT license. See `LICENSE` for more information.


## Third-party Libraries

The following dependencies are bundled with this project, but are under terms of a separate license:
* [pytorch-sepconv](https://github.com/sniklaus/pytorch-sepconv) by [sniklaus](https://github.com/sniklaus)
* [CUDA Gradient for sepconv] (https://github.com/ekgibbons/pytorch-sepconv)
## Installation

Clone this directory on your local machine:
git clone https://url.to.remote.repository

Move to the project directory:
cd ./DeepLearningFrameInterpolation

Create a new configuration file:
echo -e "from src.default_config import *\r\n\r\n# ...custom constants here" > ./src/config.py

To train the network, run main.py as a module:
python3 -m src.main


## References

\[1\] Video Frame Interpolation via Adaptive Separable Convolution, Niklaus 2017, [arXiv:1708.01692](https://arxiv.org/abs/1708.01692)
