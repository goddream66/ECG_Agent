# 模型和运行时来源

| 本地路径 | 上游来源 |
|---|---|
| `1_lead_ECGFounder.pth` | [ECGFounder 单导联权重](https://huggingface.co/PKUDigitalHealth/ECGFounder/blob/main/1_lead_ECGFounder.pth) |
| `upstream/ecgfounder/net1d.py`、`tasks.txt` | [ECGFounder 实现和标签](https://github.com/PKUDigitalHealth/ECGFounder/tree/master) |
| `upstream/anyppg/resnet1d.py`、`anyppg_ckpt.pth` | [AnyPPG 发布代码与权重](https://github.com/PKUDigitalHealth/AnyPPG/tree/main/load_anyppg) |
| `upstream/cardiophon-release/heart_sound_classification/` | [CardioPHON 发布提交 a3bf9709](https://git.lih.lu/vdespotovic/cardiophon/-/tree/a3bf97097842ec16b61672960eb928124c4140a8/heart_sound_classification) |
| `llm/Qwen3-4B-Q4_K_M.gguf` | [Qwen3-4B GGUF 权重](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| `llama-local/` | [llama.cpp b12283 Windows Vulkan x64](https://github.com/ggml-org/llama.cpp/releases/tag/b12283) |

每个模型按自身发布代码的结构加载。任务头能力以保存的权重和标签为准，软件只从配置目录执行已核验来源的模型实现。本机下载的权重与运行时归档的 SHA-256 记录于同目录的 `SHA256SUMS.csv`。
