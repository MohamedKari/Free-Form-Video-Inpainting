##### CUDA #####
FROM nvidia/cuda:9.0-cudnn7-devel

##### CONDA #####
RUN apt-get update -y && \
    apt-get install -y \
        wget
  
RUN wget --progress=dot:mega https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV PATH="/root/miniconda3/condabin:${PATH}"

##### PYTHON PACKAGE DEPENDENCIES #####
WORKDIR /app

# required by opencv-python, https://github.com/conda-forge/pygridgen-feedstock/issues/10#issuecomment-365914605 
RUN apt-get install -y libgl1-mesa-glx

# required by cmake
RUN apt-get update && \
    apt-get install -y build-essential cmake libgtk-3-dev libboost-all-dev

# environment.yaml contains desired torch version
COPY environment.yaml environment.yaml
RUN conda env update -f environment.yaml --name base

##### APPLICATION #####
ENV PYTHONUNBUFFERED=.

COPY . /app

ENTRYPOINT [ "bash", "run.sh" ]
# cd src && \
# python train.py -r ../data/v0.2.3_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD128_1_1_10_1_VOR_allMasks_load135_e135_pdist0.1256.pth --dataset_config  other_configs/inference_example.json -od ../data/test_outputs