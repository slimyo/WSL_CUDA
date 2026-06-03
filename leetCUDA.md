# 本项目使用学习leetcuda的方案
# 使用 Git Submodule (官方推荐)
这是 LeetCUDA 官方本身就在使用的方法。它将 LeetCUDA 作为独立仓库的一个引用链接到你的项目中，而不是直接把文件复制进来，因此主项目的 .git 历史会保持干净。

操作步骤：
添加子模块：在你的项目根目录下执行以下命令。
```bash
# 语法：git submodule add <仓库地址> <存放路径>
git submodule add https://github.com/xlite-dev/LeetCUDA.git third_party/LeetCUDA
```
这会在你的项目中生成两个文件：
third_party/LeetCUDA/: 存放 LeetCUDA 的代码，但它在 Git 眼里只是一个指向特定版本的“快捷方式”。
.gitmodules: 一个配置文件，记录了子模块的信息，需要提交到你的仓库中。

初始化并更新子模块：
```bash
cd third_party/LeetCUDA
# --init 是首次初始化，--recursive 会拉取 LeetCUDA 自己的子模块
git submodule update --init --recursive
```
这一步必不可少，因为 LeetCUDA 本身也依赖 cutlass、ffpa-attn 等外部库作为子模块，--recursive 参数会一次性把它们全部拉取到位。
（可选）切换到特定分支：
```bash
# 进入子模块目录，切换到 main 分支
cd third_party/LeetCUDA
git checkout main
```
默认情况下，子模块会处于一个 “detached HEAD” 状态，这通常没关系。但如果你需要跟踪 main 分支的最新更新，可以这样操作。