docker run -ti --rm -p 9501:80 -v $PWD:/app -w=/app unet bash

nvidia-docker run -ti --rm -p 9501:80 -v $PWD:/app -w=/app unet bash

CUDA_VISIBLE_DEVICES="0,1" python3 unet.py