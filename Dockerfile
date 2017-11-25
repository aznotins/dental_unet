FROM ufoym/deepo:all-py35

WORKDIR /mkl

RUN apt-get update && apt-get -qq -y install cmake git curl \
    && git clone https://github.com/01org/mkl-dnn.git \
    && cd mkl-dnn \
    && cd scripts && ./prepare_mkl.sh && cd .. \
    && mkdir -p build && cd build && cmake .. && make \
    && make install
# https://software.intel.com/en-us/articles/intel-mkl-dnn-part-1-library-overview-and-installation

ENV LD_LIBRARY_PATH /mkl/mkl-dnn/external/mklml_lnx_2018.0.1.20171007/lib:${LD_LIBRARY_PATH}

WORKDIR /app

#RUN pip3 --no-cache-dir install --upgrade Pillow
