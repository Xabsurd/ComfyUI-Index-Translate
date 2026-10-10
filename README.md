# ComfyUI Index-Translate

Independent ComfyUI integration for [Bilibili Index-Translate](https://github.com/bilibili/Index-Translate). Includes simple/pro text translation nodes, 32 target languages, 2B/9B/35B-A3B preview, automatic Hugging Face → ModelScope downloads, and optional CPU offloading. UI translations: Chinese, English and Japanese. Both nodes output `STRING`.

## Installation / 安装

Clone into `ComfyUI/custom_nodes` and install dependencies in the same Python environment used by ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Xabsurd/ComfyUI-Index-Translate.git
conda activate comfyui
python -m pip install -r ComfyUI-Index-Translate/requirements.txt
```

Restart ComfyUI and refresh the browser. Search for **Index-Translate** (simple) or **Index-Translate Pro**. Connect the text output to a text/prompt input in your workflow.

## Usage / 使用

本节点使用 Bilibili 的 Index-Translate 本地模型进行文本翻译，默认 `Index-Translate-2B`。

首次执行时下载模型到 `ComfyUI/models/index_translate/`：先尝试 Hugging Face，失败后自动使用 ModelScope。也可以设置 `INDEX_TRANSLATE_MODEL_DIR` 覆盖缓存目录。

在 ComfyUI 使用的 conda 环境中安装依赖：

```bash
pip install -r custom_nodes/ComfyUI-Index-Translate/requirements.txt
```

重启 ComfyUI 后，默认使用 **Index-Translate** 简单节点，提供输入文字、目标语言下拉框、模型和 offload 开关。**Index-Translate Pro** 专业节点提供源语言、术语表（`术语:译名, 术语2:译名2`）、附加指令、文体、输出长度和温度。两个节点都只输出文本。

语言下拉框包含官方脚本列出的全部 32 种语言，使用语言代码与本地语言名称显示。模型支持 2B（默认）、9B、35B-A3B preview；所有模型均为本地推理。

简单版采用[官方默认推理设置](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/translate.py)：自动识别源语言、贪心解码（温度 0）、关闭思考、最多输出 1024 token。专业版默认相同，允许修改温度和输出长度。这是官方参考设置，不保证每段文本的最优结果。

UI 支持中文、英语、日语，跟随 ComfyUI 界面语言；无需增加语言选择控件。原 `IndexTranslate` 节点标识保留为专业版。

两个节点都有 `offload` 开关，默认开启：翻译后模型转移到 CPU 内存，下次翻译再加载到 GPU。关闭时模型留在当前设备，连续翻译无需重新搬运，但会继续占用显存。旧工作流未设置开关时仍默认卸载；HF 下载失败继续自动回退到 ModelScope。

## License / 许可证

Apache-2.0; see `LICENSE` and `NOTICE`. This is an independent integration. Model weights are not included and remain subject to their own licenses.
