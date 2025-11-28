# Installation Guide: lightning.amdgpu

Here are instructions for installing the `lightning.amdgpu` device on the AMD Dev Cloud.

**System Requirements:**
* Spin up a VM with one MI300X and ROCm 7.0/7.1. Here we try with 7.1.

## Prerequisites

```bash
sudo apt install python3.12-venv -y
mkdir PL
cd PL
python3 -m venv venv-PL
source venv-PL/bin/activate
```

Clone this repository:

```bash
git clone [https://github.com/josephleekl/amd-lightning-install-demo.git](https://github.com/josephleekl/amd-lightning-install-demo.git)
```

---

## Installation Methods

### 1. Pip install wheel

```bash
pip install <wheel>
# this will automatically install pennylane, lightning-qubit, lightning-amdgpu
```

### 2. Docker

**Step A: Configure AMD Container Toolkit**

Enable running docker images with AMD GPU.
*Reference: [AMD Container Toolkit Quick Start](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html)*

```bash
sudo apt update
sudo usermod -a -G render,video $LOGNAME
sudo apt update
sudo apt install vim wget gpg
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget [https://repo.radeon.com/rocm/rocm.gpg.key](https://repo.radeon.com/rocm/rocm.gpg.key) -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] [https://repo.radeon.com/amd-container-toolkit/apt/](https://repo.radeon.com/amd-container-toolkit/apt/) noble main" | sudo tee /etc/apt/sources.list.d/amd-container-toolkit.list
sudo apt update
sudo apt install amd-container-toolkit
sudo amd-ctk runtime configure
sudo systemctl restart docker

# Verify configuration
docker run --runtime=amd -e AMD_VISIBLE_DEVICES=0 rocm/dev-ubuntu-24.04 amd-smi monitor
```

**Step B: Run Container**

```bash
docker run -it --rm --runtime=amd --gpus 1  pennylaneai/pennylane:latest-lightning-kokkos-rocm /bin/bash
```

### 3. From source

For full details, see the documentation:
[PennyLane-Lightning Build Guide](https://xanaduai-pennylane--1297.com.readthedocs.build/projects/lightning/en/1297/)

**Install dependencies:**

```bash
sudo apt install cmake ninja-build gcc-11 g++-11

# Install Kokkos
wget https://github.com/kokkos/kokkos/archive/refs/tags/4.5.00.tar.gz
tar xvfz 4.5.00.tar.gz
cd kokkos-4.5.00

export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.5.0/GFX942
mkdir -p ${KOKKOS_INSTALL_PATH}

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PATH} \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_TESTING:BOOL=OFF \
    -DKokkos_ENABLE_SERIAL:BOOL=ON \
    -DKokkos_ENABLE_HIP:BOOL=ON \
    -DKokkos_ARCH_AMD_GFX942:BOOL=ON \
    -DKokkos_ENABLE_COMPLEX_ALIGN:BOOL=OFF \
    -DKokkos_ENABLE_EXAMPLES:BOOL=OFF \
    -DKokkos_ENABLE_TESTS:BOOL=OFF \
    -DKokkos_ENABLE_LIBDL:BOOL=OFF
cmake --build build && cmake --install build
export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":/opt/rocm:$CMAKE_PREFIX_PATH

git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
# Temporary branch
git checkout amd-v0.43.0
pip install -r requirements.txt
pip install pennylane

# First Install Lightning-Qubit
PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
python -m pip install . -vv

# Install Lightning-AMDGPU
PL_BACKEND="lightning_amdgpu" python scripts/configure_pyproject_toml.py
export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=hipcc \
                   -DCMAKE_CXX_FLAGS='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11/'"
python -m pip install . -vv

```
