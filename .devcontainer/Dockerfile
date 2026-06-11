# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/debian/.devcontainer/base.Dockerfile

# [Choice] Debian version (use bullseye on local arm64/Apple Silicon): bullseye, buster
ARG VARIANT="buster"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# Combine the apt-get commands and set noninteractive to prevent freezing
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get upgrade -y \
    && apt-get install -y git build-essential gdb-multiarch qemu-system-misc gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu novnc websockify

# enable loading custom .gdbinit per folder.
RUN echo "set architecture riscv:rv64" > /root/.gdbinit