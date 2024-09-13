# 安装依赖(root权限)
```apt install libeigen3-dev libsentencepiece-dev libboost-program-options-dev libboost-system-dev libboost-filesystem-dev gtest-dev```
# 编译
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```