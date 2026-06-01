# 版本基线

> 最后核查：2026-06-01。GPU / AI 系统生态迭代很快，本页记录学习材料采用的时间基线；真正部署前仍应以官方文档、release notes 和本机 `--help` 输出为准。

## Python / AI 框架

以下版本来自 2026-06-01 对 PyPI metadata 的查询，用于判断文档是否明显滞后。

| 包 | 当前 PyPI 版本 | 文档中的使用建议 |
|----|----------------|------------------|
| `torch` | 2.12.0 | 新代码优先使用 `torch.amp`、`torch.compile`、`torch.nn.attention.sdpa_kernel` 等当前接口；旧教程里的 `torch.cuda.amp` 和 `torch.backends.cuda.sdp_kernel` 需要留意迁移 |
| `triton` | 3.7.0 | 作为手写 GPU kernel 和 PyTorch Inductor 代码生成的重要工具，但性能结论必须结合矩阵形状和 GPU 代际实测 |
| `vllm` | 0.22.0 | CLI、调度器、量化、多模态和结构化输出支持变化较快，部署前以 `vllm serve --help` 与官方文档为准 |
| `sglang` | 0.5.12.post1 | 重点关注 RadixAttention、结构化输出、前缀缓存和多模态支持；和 vLLM 的优劣要按流量形态 benchmark |
| `tensorrt-llm` | 1.2.1 | 更适合模型和 shape 相对稳定、愿意维护 build/engine 发布链路的 NVIDIA 数据中心 GPU 场景 |
| `flash-attn` | 2.8.3 | PyPI 稳定包仍以 FlashAttention-2 生态为主；FlashAttention-3 主要面向 Hopper，使用方式需查项目 README |
| `deepspeed` | 0.19.1 | 仍适合作为 ZeRO / Offload / Megatron 组合的一种选择；与 PyTorch FSDP 的取舍要按团队生态和训练栈决定 |
| `kubernetes` | 36.0.1 | Python client 版本不等于集群版本；GPU 集群重点看 NVIDIA GPU Operator、device plugin、RDMA device plugin 与调度器能力 |

## 官方入口

- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [CUDA Toolkit 文档](https://docs.nvidia.com/cuda/)
- [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVIDIA TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)
- [vLLM 文档](https://docs.vllm.ai/)
- [SGLang 文档](https://docs.sglang.ai/)
- [FlashAttention 项目](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA GPU Operator 文档](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)

## 阅读原则

1. **稳定原理优先**：CUDA execution model、memory hierarchy、attention FLOPs、collective communication 这些内容变化慢，可以作为长期基础。
2. **接口和 CLI 必须复查**：`vllm serve`、`trtllm-build`、`torch.compile`、Kubernetes CRD/YAML 示例都可能随版本改变。
3. **性能结论只当方向感**：公开 benchmark 只能帮助缩小候选范围，真实选型必须使用自己的模型、请求长度分布、并发和硬件做压测。
4. **硬件数字说明口径**：H100/B200 的 Tensor Core 峰值常分 dense 与 sparse 宣传口径，做 roofline 或 MFU 估算时必须先统一口径。
