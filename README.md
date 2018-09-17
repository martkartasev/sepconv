# Video Frame Interpolation

A pretrained model can be downloaded from [here](https://people.kth.se/~carlora/sepconv/pretrained.pth).

<p align="center">
  <img alt="Final Result" src="https://people.kth.se/~carlora/sepconv/vimeo.png" width="auto" height="600">
</p>

---

## Installation

### Install pip3:
```
sudo apt-get install python3-setuptools
sudo easy_install3 pip
```

### Install GCC & friends:
```
sudo apt-get install gcc libdpkg-perl python3-dev
```

### Install FFmpeg:
```
sudo apt-get install ffmpeg
```

### Install CUDA:
```
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0 -y
```

### Install NVCC:
```
sudo apt-get install nvidia-cuda-toolkit
```

### Install dependencies:
```
sudo pip3 install -r ./path/to/requirements.txt
```

-------
## How to setup the project and train the network:

### Move to the project directory:
```
cd ./DeepLearningFrameInterpolation
```

### Create a new configuration file:
```
echo -e "from src.default_config import *\r\n\r\n# ...custom constants here" > ./src/config.py
```

### To train the network, run `main.py` as a module:
```
python3 -m src.main
```

-------
## License

This project is released under the MIT license. See `LICENSE` for more information.

## Third-party Libraries

The following dependencies are bundled with this project, but are under terms of a separate license:
* [pytorch-sepconv](https://github.com/sniklaus/pytorch-sepconv) by [sniklaus](https://github.com/sniklaus)
* [CUDA Gradient for sepconv](https://github.com/ekgibbons/pytorch-sepconv)

## References

\[1\] Video Frame Interpolation via Adaptive Separable Convolution, Niklaus 2017, [arXiv:1708.01692](https://arxiv.org/abs/1708.01692)
