# 开发与服务器实验流程

## 执行约定

日常在本地 `D:\CardioAgent` 修改代码、检查功能并运行测试。普通修改不自动提交、不推送 GitHub，也不直接同步到服务器。只有准备运行已授权的服务器 GPU 实验时，才整理提交、推送并在服务器拉取；用户另有明确要求时按该要求执行。

```text
本地修改 → 本地测试 → 保留本地改动
准备 GPU 实验时：检查变更 → commit → push → 服务器 pull --ff-only → GPU 实验
```

本地测试用于检查代码可运行性、规则逻辑和接口回归。CUDA、服务器路径和真实模型的 GPU 执行仍需在服务器做小规模检查，再开始完整实验。

## 路径与 Git 范围

| 用途 | 路径 |
| --- | --- |
| 本地代码和测试 | `D:\CardioAgent` |
| GitHub origin | `https://github.com/goddream66/ECG_Agent.git` |
| 服务器代码 | `/home/gaojiajun/ECGagent` |
| 服务器数据、权重、环境、缓存和结果 | `/data/1shared/gaojiajun/ECGagent` |

Git 只管理源代码、配置模板、测试和文档。数据、权重、环境、生成结果、临时文件和认证信息不提交；机器专用的 `activate.sh` 保留在各自机器。忽略规则不能代替提交前检查，已经跟踪的文件仍需检查。

## 本地验证与提交

在本地项目目录运行：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
git status --short
git diff --stat
```

准备 GPU 实验时，逐项检查并暂存本次所需文件，检查暂存区后提交和推送。不要无差别暂存整个工作区，不覆盖已有未提交修改。若工具环境无法自动识别工作树，可使用显式路径：

```powershell
git --git-dir=D:/CardioAgent/.git --work-tree=D:/CardioAgent status --short
```

## 服务器更新与实验

```bash
cd /home/gaojiajun/ECGagent
git status --short
git fetch origin
# 确认没有需要保留的未提交变更，且当前分支已配置正确的上游后：
git pull --ff-only
git rev-parse HEAD
source ./activate.sh
```

遇到未提交变更，先检查、备份并整合到本地开发基线；遇到分支分叉，先检查差异。不要用强制推送、reset --hard 或 clean 覆盖代码。正在运行的实验不更新其代码目录。

每次实验记录 Git 提交号、工作区是否有未提交修改，以及实际使用的配置、输入数据和模型指纹。实验输出继续写入 data 盘的独立运行目录；本地缓存回放不作为 GPU 性能测量。

## 首次启用状态（2026-11-17）

本地和服务器已配置同一个 origin，拉取设置为 `pull.ff=only`。配置时远端查询成功但未返回任何分支；GitHub 推送认证与写入权限尚未验证。

两端均存在此前重构留下的未提交改动。首次准备 GPU 实验时，需要先核对并整合这些改动，建立经过本地测试的基线提交，推送成功后再处理服务器分支的首次对接。本次只配置流程，不提交、不推送，也不启动 GPU 实验。
