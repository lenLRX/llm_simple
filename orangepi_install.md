# 安装依赖(root权限)
```apt install libeigen3-dev libsentencepiece-dev libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libgtest-dev libspdlog-dev nlohmann-json3-dev libre2-dev```
# 安装python依赖
```pip install transformers ml_dtypes```
# 编译
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
