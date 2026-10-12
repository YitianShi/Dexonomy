FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        git \
        curl \
        openssh-client \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY dexonomy ./dexonomy
COPY assets/hand ./assets/hand
COPY third_party ./third_party
RUN pip install --no-cache-dir -e .

CMD ["bash"]
