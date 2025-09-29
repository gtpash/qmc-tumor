##################################################################################################
# 
# This Dockerfile is used to build a Docker image with a FEniCS environment.
# 
# In particular, this will set up an environment with the following tools:
#   - miniconda3 with python 3.11
#   - FEniCS v2019.1.0
#   - h5py
# 
# The image is available at: https://hub.docker.com/r/gtpash/tumor_qmc_conda:latest
# 
##################################################################################################

FROM gtpash/tacc-ubuntu20-mvapich2.3-ib:latest

ARG DEBIAN_FRONTEND=noninteractive

########################################
# Install conda
########################################
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
# Download and install miniforge
RUN wget -q -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && conda config --system --set default_threads 4 \
    && conda install --yes --no-update-deps python=3.11 \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && docker-clean

RUN conda init

########################################
# Install FEniCS
########################################
RUN conda install --yes -c conda-forge fenics==2019.1.0 matplotlib scipy

########################################
# Install the rest of the pipeline
########################################
RUN python3 -m pip install --upgrade pip \ 
    && python3 -m pip install --no-cache-dir h5py
