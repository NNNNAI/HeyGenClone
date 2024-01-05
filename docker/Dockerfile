# base image
# FROM nvcr.io/nvidia/pytorch:22.02-py3 as base
# FROM nvcr.io/nvidia/pytorch:23.05-py3 as base
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel as base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel as base


ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
#     && find /etc/apt/sources.list /etc/apt/sources.list.d/ -type f -exec sed -Ei 's!http://!'http://devops.io:3142'/!g' {} \; \
#     && find /etc/apt/sources.list /etc/apt/sources.list.d/ -type f -exec sed -Ei 's!https://!'http://devops.io:3142'/HTTPS///!g' {} \;


# RUN echo "Acquire::http::Proxy \"http://devops.io:3142\";" > /etc/apt/apt.conf.d/00aptproxy
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt update --allow-insecure-repositories \
    && apt-get install -y gcc openssh-server git sudo ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx ffmpeg wget g++ fonts-wqy-zenhei tmux make\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN conda clean --all

# RUN pip install torch==2.0.0+cpu torchvision==0.15.1+cpu -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html  -i https://mirror.baidu.com/pypi/simple 
# 换中科大源
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
# pip install basicsr -i https://mirrors.aliyun.com/pypi/simple  --use-deprecated legacy-resolver
# Development image
FROM base as development

ARG USERNAME=myuser
ARG USER_UID
ARG USER_GID=${USER_UID}

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chmod -R 777 /home/$USERNAME

# Switch to the new user
USER $USERNAME
WORKDIR /home/$USERNAME


