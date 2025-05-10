./build/src/llama2_main \
--config=/ssd/models/Qwen2.5-7B-Instruct/config.json \
--tokenizer=/ssd/models/Qwen2.5-7B-Instruct/ \
--weight=/ssd/models/Qwen2.5-7B-Instruct_converted \
--model_type=qwen2 \
--device_type=npu \
--max_seq_len=8192 \
--max_gen_token=256 \
--temperature=0.0 \
--debug_print=false \
--log_level=info \
--rope_is_neox_style=true \
--prompt="深度优化指南请求：通用矩阵乘法（GEMM）的全栈性能工程实践

请以高性能计算专家的身份，系统论述GEMM优化的完整技术体系。要求从晶体管层面到算法层进行跨抽象层分析，包含以下维度：

一、硬件感知优化基础（展开以下每个子项）

现代CPU内存层级解剖

详述如何通过cache blocking适应L1/L2/L3缓存行

示例：针对不同缓存容量（如8MB L3）的分块策略计算公式

数据预取模式设计（软件预取指令的最佳插入距离）

SIMD指令工程实践

AVX-512与ARM SVE的寄存器压力对比

汇编级循环展开策略（展示8x8分块的双缓冲汇编模板）

FMA指令流水线冒险规避技巧

GPU架构深度适配

CUDA warp-level同步优化（共享内存bank conflict量化分析）

全局内存合并访问模式设计（展示2D tile的访存对齐公式）

Tensor Core编程范式（WMMA API使用陷阱与性能调优日志）

二、算法革新路线（需数学推导）

复杂分治策略

Strassen算法在实践中的递归终止条件选择（给出浮点误差传播模型）

Winograd变换的数值稳定性改进方案

基于分块秩的近似算法误差界证明

稀疏化与量化

结构化稀疏的硬件友好模式设计（NVIDIA A100稀疏特性适配）

混合精度训练中的动态缩放因子推导

低秩近似与GEMM的耦合优化（给出SVD截断误差分析）

三、编译工程化进阶

LLVM中间表示调优

Polly循环优化编译指示实战

MLIR GEMM方言生成技术路线

自动向量化失败的补救模式

自动调参系统设计

遗传算法参数空间剪枝策略

贝叶斯优化中的协方差矩阵自适应

多目标优化Pareto前沿的筛选标准

四、异构计算协同

多芯片负载均衡

CPU-GPU流水线深度分析（计算/通信重叠的数学模型）

基于RDMA的跨设备零拷贝实现

异构内存一致性模型解决方案

五、验证方法论

Roofline模型深度应用

实测不同架构的运算强度阈值

性能偏离度诊断流程图

瓶颈定位的热力图分析法

技术写作要求：

每个优化点需提供理论依据（附复杂度公式推导）

关键路径给出CUDA/C++代码片段及编译器内联汇编示例

包含主流硬件实测数据（如A100 vs Xeon Platinum对比表格）

讨论商业化实现差异（对比oneDNN vs cuBLAS设计哲学）

最后给出优化决策树（包含分支判断条件）

请采用学术论文写作规范，分章节编号至三级标题，使用LaTeX公式描述关键技术指标，总输出保持工程技术文档的严谨性同时具备可操作性。"
