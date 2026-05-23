# See-Point-Fly Extended: AirSim 仿真与 Tello 实机的 VLM 无人机导航实验

本项目是基于官方 [See-Point-Fly](https://github.com/Hu-chih-yao/see-point-fly/) 的复现与扩展版本。官方 SPF 提出了一个 `图像 + 自然语言指令 -> VLM 选点 -> 无人机动作` 的学习无关框架；当前仓库在这一思路上，围绕 AirSim 仿真复现、Tello 实机初步验证、候选点约束、安全过滤和轻量记忆导航做了阶段性改进。

仓库当前更适合作为实验代码与阶段性结果整理，而不是“开箱即用”的稳定自主飞行系统。README 以下内容只描述当前代码主线里已经存在的实现，不把历史尝试写成默认成果，也不把 AirSim 的能力误写成 Tello 已具备的能力。

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 与官方 SPF 的关系](#2-与官方-spf-的关系)
- [3. 当前主要改进](#3-当前主要改进)
- [4. 离线工具](#4-离线工具)
- [5. 项目结构](#5-项目结构)
- [6. 环境与依赖](#6-环境与依赖)
- [7. 配置说明](#7-配置说明)
- [8. 运行方式](#8-运行方式)
- [9. 输出与调试文件](#9-输出与调试文件)
- [10. 当前限制](#10-当前限制)
- [11. 后续计划](#11-后续计划)
- [12. Reference / 致谢](#12-reference--致谢)

## 1. 项目简介

本项目面向 VLM-based UAV navigation 的复现与扩展实验，核心闭环仍然是：

1. 获取无人机当前视角图像。
2. 输入自然语言任务描述。
3. 使用视觉语言模型在图像中选择下一步导航目标点。
4. 将 2D 图像点投影成相对运动指令。
5. 执行动作并进入下一帧闭环。

当前仓库主要包含两条实验线：

- `AirSim`：用于跑通仿真闭环、调试候选点、安全过滤、轻量记忆导航等机制。
- `Tello`：用于实机初步验证图像输入、VLM 选点、ActionPoint 投影与控制执行链路。

另外，仓库还保留了官方项目中的 `sim` 模式与若干工具脚本，但当前二次开发的重点是 `AirSim + Tello`。

## 2. 与官方 SPF 的关系

- Upstream: <https://github.com/Hu-chih-yao/see-point-fly/>
- 本项目不是从零实现，而是在官方 SPF 基础上做复现、调试和扩展。
- 保留了官方 SPF 的核心思想：`image + instruction -> VLM point selection -> UAV action`。
- 当前新增内容主要集中在：
  - AirSim 仿真闭环稳定性与调试可视化。
  - 固定候选点 `P1~P15` 约束，减少自由坐标漂移。
  - 基于深度图的轻量安全过滤。
  - 受 IEVE 启发的 `EXPLORE / VERIFY / EXPLOIT` 轻量状态机制。
  - Tello 实机链路恢复与实验性验证。

## 3. 当前主要改进

### 3.1 AirSim 仿真部分

当前代码中，AirSim 已经跑通 `图像采集 -> VLM 推理 -> 点位生成 -> 动作执行 -> 下一帧` 的闭环。主线实现包括：

- 支持从 AirSim 获取 `RGB`、`DepthPerspective` 深度图和 `vehicle pose`。
- 已将自由坐标输出收敛为固定候选点 `P1~P15`，降低 VLM 直接输出任意 2D 坐标时的漂移问题。
- 会生成候选点叠加图、VLM 输入图、最终选点图，以及对应的 `decision_*.json / .jpg` 调试结果。
- `depth safety`：
  - 对每个候选点采样深度。
  - 按 `safe / blocked / unknown` 标记候选点。
  - 在 prompt 中把候选点深度与安全标签一并提供给 VLM。
- `blocked replacement`：
  - 当 VLM 或 IEVE-Lite 选中的候选点被标记为 `blocked` 时，当前实现会优先替换为邻近的 `safe` 点。
  - 若附近没有 `safe` 点，再退化到 `unknown` 或中心点 `P8`。
- `ground risk` 处理：
  - 当前实现会检查候选点对应的垂直速度风险。
  - 现阶段主线做法不是“换点重规划”，而是对明显向下贴地的动作进行标记，并在最终 3D 动作里压掉向下风险分量。
  - 也就是说，地面风险当前更接近“下压抑制 / clamp”，不是完整避障器。
- `yaw clamp`：
  - 对单次大角度转向做限幅，避免一步转太猛导致控制不稳定。
- `SPF-IEVE-Lite`：
  - 轻量引入 IEVE 的 `EXPLORE / VERIFY / EXPLOIT` 状态思想。
  - 不是完整复现 IEVE，而是结合当前 SPF 点选链路做一个轻量状态机。
- `TargetMemory`：
  - 记录最近一次较可靠的目标观察。
  - 在 `EXPLOIT` 阶段短暂丢失目标时，可用最近可靠像素位置继续辅助决策。
- `SearchMemory`：
  - 记录 `recent choices`、`failed_search_count`、`successful_observation_count`、`suggested_alternative` 等统计信息。
  - 它不会直接替换最终动作。
  - 当前最多只会给下一帧 prompt 增加一个 advisory-only 的弱提示，主要仍用于日志和 debug 分析。
- 复杂建图主线已清理：
  - 当前活跃代码中没有把 `LocalMap`、复杂 depth projection 建图之类逻辑作为主线控制模块。
  - 如果需要回溯历史尝试，可看 `docs/` 下的历史记录文档。

一句话概括：AirSim 现在更像“候选点约束 + 深度安全过滤 + 轻量记忆导航”的可调试实验平台。

### 3.2 Tello 实机部分

Tello 部分当前定位是“实机初步验证”，不要把它理解成已经稳定的自主导航系统。

当前主线代码中，Tello 重点验证的是：

- 实时图像输入是否稳定获取。
- VLM 是否能根据当前图像和任务描述给出下一步点位。
- 2D 点是否能投影为 `ActionPoint`。
- `ActionPoint` 是否能转换为 Tello 的 `yaw / forward / vertical` 控制。

当前状态更接近“干净的基线控制链路 + 小步修正”，而不是复杂安全层：

- `adaptive_mode` 目前是默认配置，已经接入 `P1~P15` 候选点约束。
- 这里的 `P1~P15` 主要用于约束 VLM 选点，减少自由点漂移。
- 这不等同于 AirSim 那套完整的深度安全、记忆导航和状态切换机制。
- Tello 当前没有接入 AirSim 那样的实时深度安全过滤。
- Tello 当前也没有接入 AirSim 的 `TargetMemory / SearchMemory / IEVE-Lite` 主线逻辑。

与用户实际飞行体验更相关的几个点：

- 当前 Tello 控制链路已经做了轻量 `yaw` 限幅。
  - 具体是对单次 `yaw` 执行时长做 clamp，避免一下子转得太多。
- 这类小改动更偏向“先减少转向过头”，而不是引入复杂状态机。
- 代码里还保留了 `obstacle_mode`：
  - 该模式会让 VLM 输出自由点和障碍框，并带 keepalive / timeout 保护。
  - 它仍然属于实验性分支，不建议在 README 中把它写成已经稳定的默认方案。
- 当前仓库主线里没有把复杂 Safety Layer 写成默认执行链路。
  - 早期更复杂的安全状态机尝试可以视为历史探索方向，但不应在项目简介里当作现阶段成果夸大描述。

实机阶段需要明确的风险与现实约束：

- Wi-Fi 延迟会直接影响图像时效与控制闭环。
- 画面抖动会影响 VLM 选点稳定性。
- 推理延迟会让“看到的图像”和“执行动作的时刻”错位。
- Tello 的高度、姿态与垂直控制本身不够稳定。
- 因此当前实机测试必须保持安全距离，并随时准备手动降落。

## 4. 离线工具

### 4.1 `src/spf_tools/depth_pro_test.py`

仓库中存在 `src/spf_tools/depth_pro_test.py`，它是一个旁路、离线的深度评估工具，不接入实时飞控。

它当前用于：

- 测试 Apple `ml-depth-pro` 在 Tello 视频、单张图片或图片目录上的单目深度估计效果。
- 对每帧生成：
  - 原图。
  - 深度可视化图。
  - `P1~P15 + ROI` 标注叠加图。
  - 每帧 JSON。
- 在输出目录汇总：
  - `results.csv`
  - `summary.json`

当前定位：

- 仅用于离线分析。
- 不接入实时 Tello 控制。
- 代码里也明确写了它与飞控逻辑隔离。
- 结合本地实验经验，单目深度推理在 RTX 2050 上速度仍偏慢，暂时不适合作为实时安全层。

### 4.2 `spf-tools` 辅助命令

项目还提供了 `spf-tools` CLI，包含：

- `diagnostics`
- `capture`
- `vlm`
- `resolution`
- `monitors`

这些工具更偏向诊断与离线测试，不直接参与 AirSim / Tello 的闭环控制。

## 5. 项目结构

```text
.
├─ README.md
├─ pyproject.toml
├─ env.example
├─ config_airsim.yaml
├─ config_tello.yaml
├─ config_sim.yaml
├─ docs/
│  ├─ coordinate_system_guide.md
│  ├─ 0904_historical_reference.md
│  └─ obstacle_avoidance_hitorical_reference.md
└─ src/
   ├─ spf/
   │  ├─ __main__.py
   │  ├─ main.py
   │  ├─ base/
   │  │  ├─ action_projector.py
   │  │  └─ drone_space.py
   │  ├─ clients/
   │  │  └─ vlm_client.py
   │  ├─ airsim/
   │  │  ├─ main.py
   │  │  ├─ controller.py
   │  │  ├─ action_projector.py
   │  │  ├─ drone_space.py
   │  │  ├─ ieve_lite.py
   │  │  └─ settings.json.example
   │  ├─ tello/
   │  │  ├─ main.py
   │  │  ├─ controller.py
   │  │  ├─ action_projector.py
   │  │  └─ drone_space.py
   │  └─ sim/
   │     └─ ...
   └─ spf_tools/
      ├─ cli.py
      ├─ depth_pro_test.py
      ├─ diagnostics/
      ├─ capture/
      ├─ resolution/
      └─ vlm/
```

目录说明：

- `src/spf/airsim/`：AirSim 模式主线，包含图像采集、深度读取、候选点、安全过滤、IEVE-Lite 和记忆模块。
- `src/spf/tello/`：Tello 实机主线，包含视频流、VLM 点选、动作投影、RC 控制、录帧与录像。
- `src/spf/clients/vlm_client.py`：统一 VLM 客户端，支持 Gemini 和 OpenAI-compatible API。
- `src/spf_tools/`：诊断脚本、VLM 测试工具、Depth Pro 离线评估工具。
- `docs/`：坐标说明和历史方案记录，不代表当前默认控制主线。

## 6. 环境与依赖

### 6.1 Python 与包管理

`pyproject.toml` 当前要求：

- Python `>= 3.13`
- 推荐使用 `uv`

安装方式：

```bash
uv sync
uv run spf --help
```

项目依赖里已经包含：

- `airsim`
- `djitellopy`
- `openai`
- `google-genai`
- `opencv-python`
- `matplotlib`
- `python-dotenv`
- `pyyaml`

如果你要额外跑 `Depth Pro`，请根据本机环境单独准备 `apple/ml-depth-pro` 与对应权重。

### 6.2 AirSim / Unreal 环境

- 需要提前安装并启动 AirSim 场景。
- 代码提供了 `src/spf/airsim/settings.json.example`，用于提高相机分辨率。
- 默认 AirSim 相机太小会明显影响导航效果，建议先配置 `settings.json` 再测试。

### 6.3 VLM 服务

代码支持两类后端：

- `gemini`
- `openai`（OpenAI-compatible API）

对应环境变量在 `env.example` 中：

```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

说明：

- 如果使用本地 OpenAI-compatible 服务，通常需要在 `.env` 中改 `OPENAI_BASE_URL`。
- 具体模型名不在 `.env` 里，而在 `config_airsim.yaml` / `config_tello.yaml` 中设置 `model_name`。

### 6.4 Tello 实机环境

- 电脑需要先连接到 Tello Wi-Fi。
- 保证电量充足。
- 建议在空旷、无遮挡、便于紧急降落的环境下测试。

### 6.5 Windows / Ubuntu 注意事项

- 当前代码同时考虑了 Windows / Ubuntu 的常见用法，但 AirSim 设置文件位置、驱动环境和网络配置会随平台不同而变化。
- `Tello`、`AirSim`、本地 VLM 服务、Depth Pro 环境通常都需要按本机实际情况单独配置。
- 如果使用本地模型服务，请优先确认 `OPENAI_BASE_URL`、端口和模型名是否与实际服务一致。

## 7. 配置说明

### 7.1 `config_airsim.yaml`

建议重点关注：

- `api_provider`
- `model_name`
- `adaptive_mode`
- `command_loop_delay`
- `base_velocity`
- `base_yaw_rate`
- `min_command_duration`
- `camera_name`
- `wind_x / wind_y / wind_z`

说明：

- 当前配置文件中的具体模型名只是本地实验值，不代表项目固定要求。
- 迁移环境时请按你自己的 VLM 服务修改。

### 7.2 `config_tello.yaml`

建议重点关注：

- `api_provider`
- `model_name`
- `operational_mode`
- `command_loop_delay`
- `show_vlm_decision_window`

说明：

- 当前默认 `operational_mode: adaptive_mode`。
- `adaptive_mode` 更接近当前 README 所描述的“候选点约束基线”。
- `obstacle_mode` 仍在代码里，但应视为实验性分支，不要混同于默认实机方案。

### 7.3 `.env`

建议关注：

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

如果你使用 OpenAI-compatible 服务，通常需要同时确认两件事：

1. `.env` 里的 `OPENAI_BASE_URL`
2. YAML 配置里的 `model_name`

## 8. 运行方式

### 8.1 AirSim 仿真

先启动 AirSim / Unreal 场景，再运行：

```bash
uv run spf airsim
```

常用调试模式：

```bash
uv run spf airsim --debug
```

说明：

- 程序会读取 `config_airsim.yaml`。
- 启动后会检查 AirSim 相机分辨率。
- 需要手动输入高层任务描述。
- `--debug` 会打印更详细的等待队列、动作执行和选点日志。

### 8.2 Tello 实机

先连接 Tello Wi-Fi，再运行：

```bash
uv run spf tello
```

如果只想跳过相机就绪检查：

```bash
uv run spf tello --skip-camera-check
```

如果需要录制原始帧：

```bash
uv run spf tello --record --record-session test_run
```

如果需要录制 MP4：

```bash
uv run spf tello --video --video-session test_run
```

说明：

- 程序会读取 `config_tello.yaml`。
- 启动后需要手动输入初始自然语言命令。
- 飞行过程中还支持动态输入新命令。
- 实机功能仍在实验阶段，请务必在安全环境下测试，并随时准备接管或降落。

### 8.3 其他命令

查看 CLI：

```bash
uv run spf --help
uv run spf-tools --help
```

运行诊断：

```bash
uv run spf-tools diagnostics
```

### 8.4 Depth Pro 离线评估

示例：

```bash
uv run python src/spf_tools/depth_pro_test.py \
  --input path/to/video.mp4 \
  --output outputs/depth_pro_test \
  --frame-step 30
```

也可以处理图片目录：

```bash
uv run python src/spf_tools/depth_pro_test.py \
  --input path/to/image_dir \
  --output outputs/depth_pro_test_images
```

说明：

- 脚本会尝试查找本地 `ml-depth-pro` 仓库和权重。
- 如果没有安装，会报错并提示手动准备环境。
- 该脚本是离线分析工具，不参与 Tello 实时控制。

## 9. 输出与调试文件

当前代码里没有统一的 `output/debug/` 主目录；主要输出位置如下：

- `action_visualizations/<timestamp>/`
  - 由 `ActionProjector` 自动创建。
  - 包含 `debug_vlm_input.jpg`、`debug_vlm_selected.jpg`、`decision_*.jpg`、`decision_*.json`。
- `Tello_frame_capture/`
  - Tello 主循环中保存“实际送入 VLM 的帧”。
- `tello_debug_frames/`
  - Tello 相机初始化检查时保存的调试图。
- `raw_frames/`
  - `--record` 模式下按帧保存的原始图像。
- `tello_videos/`
  - `--video` 模式下保存的 MP4。
- `outputs/...`
  - 这是 `depth_pro_test.py` 常用的离线输出目录示例，实际路径由 `--output` 决定。

如果你准备整理实验结果或录屏素材，建议优先保留：

- `action_visualizations/`
- `Tello_frame_capture/`
- `tello_videos/`
- `depth_pro_test.py` 的输出目录

## 10. 当前限制

以下限制建议在 README 中明确写出，而不是回避：

- AirSim 仿真效果不等于 Tello 实机效果，两者优化程度不同。
- Tello 实机仍不稳定，受 Wi-Fi、视频时延、画面抖动、电量与姿态波动影响很大。
- VLM 推理延迟仍然偏高，会影响闭环时效。
- 自由点输出原本容易漂移，因此当前才重新收缩到候选点约束；但 Tello 候选点方案本身也仍在调试。
- Tello 的高度 / 垂直控制不稳定，实机阶段应谨慎对待上下动作。
- AirSim 已有较完整的候选点安全过滤与轻量记忆逻辑，但这些能力并没有等价迁移到 Tello。
- `Depth Pro` 当前没有接入实时飞控。
- `SearchMemory` 目前不直接控制最终动作，主要用于统计、日志和下一帧的弱提示。
- 当前代码仍属于实验性研究代码，不宜直接当作稳定产品使用。

## 11. 后续计划

当前比较现实的后续方向包括：

- 继续完善 Tello 侧的候选点约束与基线稳定性。
- 保持轻量 `yaw` 限幅思路，先减少“转向过头导致目标出视野”的问题。
- 优化 VLM 推理延迟，缩短闭环等待时间。
- 在 AirSim 上做更系统的消融实验，比较候选点、安全过滤、IEVE-Lite 和记忆模块的收益。
- 整理 demo、日志与阶段性实验结果，便于后续论文 / 大创材料使用。
- 继续谨慎评估单目深度是否适合作为离线分析工具，或低频辅助安全信息，而不是直接实时接管飞行。

## 12. Reference / 致谢

- Official See-Point-Fly repository: <https://github.com/Hu-chih-yao/see-point-fly/>
- See-Point-Fly paper:
  - OpenReview: <https://openreview.net/forum?id=AE299O0tph>
  - PMLR: <https://proceedings.mlr.press/v305/hu25e.html>
- IEVE inspiration:
  - Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation
  - arXiv: <https://arxiv.org/abs/2402.17587>
  - OpenReview: <https://openreview.net/forum?id=uR26imr3mi>
- Apple Depth Pro:
  - Repository: <https://github.com/apple/ml-depth-pro>
  - Paper: <https://arxiv.org/abs/2410.02073>

如果你是从官方 SPF 仓库来到这里，建议把当前仓库理解为“面向 AirSim 与 Tello 的阶段性复现 / 扩展实验分支”，而不是官方 README 的简单镜像。
