# Implementing Adaptive Separable Convolution for Video Frame Interpolation

This is a fully functional implementation of the work of Niklaus et al. \[[1](#references)\] on Adaptive Separable Convolution, which claims high quality results on the video frame interpolation task. We apply the same network structure trained on a smaller dataset and experiment with various different loss functions, in order to determine the optimal approach in data-scarce scenarios.

For detailed information, please see our report on [arXiv:1809.07759](https://arxiv.org/abs/1809.07759).

The video below is an example of the capabilities of this implementation. Our pretrained model (used in this instance) can be downloaded from [here](https://people.kth.se/~carlora/sepconv/pretrained.pth).

<a href="https://vimeo.com/272619630" target="_blank">
<img src="https://people.kth.se/~carlora/sepconv/vimeo.jpg" alt="Video">
</a>

---

## Installation
> Note that the following instructions apply in the case of a fresh Ubuntu 17 machine with a CUDA-enabled GPU. In other scenarios (ex. if you want to work in a virtual environment or prefer to use the CPU), you may need to skip or change some of the commands.

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
cd ./sepconv
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
* [CUDA Gradient for sepconv](https://github.com/ekgibbons/pytorch-sepconv) by [ekgibbons](https://github.com/ekgibbons)

## References

\[1\] Video Frame Interpolation via Adaptive Separable Convolution, Niklaus 2017, [arXiv:1708.01692](https://arxiv.org/abs/1708.01692)
