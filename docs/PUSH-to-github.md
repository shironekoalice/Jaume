# 把当前仓库推到 GitHub（代码）与 ModelScope（权重 + 日志）

HiDevLab 的 950 申请表里要填「开源项目链接」，填的是 **GitHub 代码仓**：

    https://github.com/shironekoalice/Jaume

审核员会点开这个链接。所以推之前它必须和申请里写的内容对得上：
现在 GitHub 上还是 11-15 的旧版（只有 `code/`，没有 `docs/`、没有 Actor），
而申请里写的是「Jaume-Actor 小模型浏览器/游戏决策智能体」。

本地已经准备好的部分（不需要你做任何事）：

| 已完成 | 说明 |
|---|---|
| 仓库瘦身 | 2.4 GB → 25.9 MB，651 个文件；权重/数据/日志已从索引移除 |
| `.gitignore` 统一 | 与 GitHub 镜像那份规则一致（权重归 ModelScope） |
| 补齐 `docs/` + `prototype/actor/` | 研究记录（89 份）与全部可复现脚本（81 个） |
| 清掉 26 个零字节垃圾文件 | AI 早期误建、文件名是文档标题的残留 |
| README 顶部加「Running on Ascend」 | 昇腾证据一眼可见 |
| 清理本地残留 | `CON` 等 |

---

## 你要做的：一条命令 + 一个 token

### 1. 拿 GitHub PAT

<https://github.com/settings/tokens> → Generate new token (classic) → 勾选 `repo`
（如果用 Fine-grained token，要给这个仓的 Contents: Read and write 权限）

### 2. 推送

```bash
cd "D:/项目/Jaume"
bash tools/push_all.sh
```

脚本会提示 `GitHub PAT (repo scope):`，直接粘贴（不回显，不会进历史）。
也可以提前放好：

```bash
# Windows
echo ghp_xxxxxxxx > ~/.github_token
```

### 3. 验证

```bash
# 应该能看到 90+ commits、docs/ 和 prototype/ 目录
git ls-remote github
```

浏览器打开 <https://github.com/shironekoalice/Jaume>，确认：
- 提交历史是最新的（不是 `v0.2: full family code`）
- 目录里有 `docs/` 和 `prototype/`
- README 顶部有 "Running on Ascend" 段

---

## 顺便推 ModelScope（权重 + 日志，可选）

ModelScope 那边是**模型页**（审核员也常看），应该同步 README 和 docs。
它需要另一个凭据：ModelScope 的 **access token**（`ms-` 开头，在
<https://modelscope.cn/my/myaccesstoken> 生成）。

```bash
export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
python code/repo/scripts/push_to_modelscope.py --repo LightNeko/Jaume --branch master
```

注意 ModelScope 的 access token **不能直接当 git 密码用**，脚本里已经处理了
这个转换（`HubApi().login()` 换 git token）。

---

## 如果 push 卡住

GitHub 网页直连在这个网络下不稳定（`github.com:443` 超时），但 **git 走 HTTPS
是通的**（`git ls-remote` 能拿到 HEAD）。所以正常情况不用配代理。

如果真的卡住，加代理：

```bash
export HTTPS_PROXY=http://127.0.0.1:<你的代理端口>
bash tools/push_all.sh
```

---

## 推完之后

申请表里：

| 栏 | 填 |
|---|---|
| 开源项目链接 | `https://github.com/shironekoalice/Jaume` |
| 补充说明 | 见 `docs/ascend-950-application.md` §1.5 的 169 字版 |

如果审核员追问「和华为是什么合作关系」，如实答：权重与代码已开源在
ModelScope 与 gitcode CANN 生态，950PR 上希望做的是**昇腾生态内的适配验证**。
若贵方确有正式立项项目，把补充说明里的项目名换成那个正式名称。
