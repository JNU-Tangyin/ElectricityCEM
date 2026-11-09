# GPS 辅助策略表示实验

本目录是可独立运行的 GPS 实验代码。与 `code/vector_exploration/` 中的 shape 实验分开，不依赖原开发机器的绝对路径或 shape 模块。

## 三种输入

| 配置 | 输入坐标（原36维索引） | bias处理 |
|---|---|---|
| 34维 | 删除3（location权重）和35（bias） | 删除 |
| 35维 | 只删除3 | 保留原bias，不合并location贡献 |
| 36维 | 保留0–35 | 保留原bias |

这些配置只改变表示学习输入。策略回放始终执行完整36维θ，没有重新训练CEM。

## 文件布局

从交付包根目录看：

- `code/gps/`：脚本与数值环境。
- `data/gps/households/`：实际实验的71户预处理时序。
- `data/gps/policies/`：seed 0、1、42的213个策略及训练收益历史。
- `data/gps/coordinates_71.csv`：面板内71户的坐标表。
- `data/gps/input_manifest.json`：原始来源与输入哈希。
- `results/gps/34d/`、`35d/`、`36d/`：各维度完整结果。
- `results/gps/comparison/`：三种维度的对照CSV和离线HTML报告。

## 安装与复现

在 Python 3.13 环境验证，依赖版本见 requirements.txt。在交付包根目录执行：

```sh
python -m pip install -r code/gps/requirements.txt
python code/gps/run_suite.py --dims 35 --output results/gps/reproduce_35d
```

`--dims` 支持34、35、36。已发布结果目录不会被覆盖；复跑请选择新的 `--output`。默认最多3个实验进程，可用 `--jobs 1` 顺序执行。

运行入口自动完成 Exp32–37、谱聚类、可视化、Exp38候选家庭检查，全部使用原来的置换预算。新结果保存日志、状态及输入/脚本哈希。

汇总三个发布版本：

```sh
python code/gps/build_dimension_comparison.py
```

独立运行单个实验可设置 `ENERGY_GPS_INPUT_DIMS` 和 `ENERGY_GPS_RESULT_ROOT`；运行入口会自动配置它们。程序默认34维。

## 脚本对应

| 脚本 | 内容 |
|---|---|
| gps_panel.py / gps_context.py | 输入维度、家庭对齐及便携路径 |
| gps_triplet_pilot.py | Exp32，逐地点留出地理排序 |
| gps_cluster_comparison.py | Exp33，同地点/距离/混合对角度量；99次置换 |
| gps_network_discovery.py | Exp34，对角/三元组网络；39次地图置换 |
| gps_place_networks.py | Exp35，SupCon/同地点三元组；39次标签置换 |
| gps_distance_regression.py | Exp36，连续地理距离回归；99次地图置换 |
| gps_spectral_comparison.py | 54次固定表示谱聚类对照 |
| gps_behavior_increment.py | Exp37，GPS对行为预测的增量 |
| audit_candidates.py | Exp38，按当前维数重算小簇并检查数据/训练记录 |
| plot_place_clusters.py / plot_gps_distance_clusters.py | 不预设簇数的三seed成员和PCA图 |
| policy_replay.py / observation_matrix.py | 沿用原数值回放/观测构造内核 |
| energy_storage_env.py | 与原训练环境相同的物理与Benefit-Cost内核 |
| history/ | 历史报告/运行脚本快照，保留旧路径作来源记录；当前复现使用上面的入口 |

## 解释口径

HDBSCAN在完整输入表示上运行，PCA仅显示。跨seed匹配通过最大成员交集的一对一匹配；额外簇分别编号。整体家庭对Jaccard与逐簇交集分开报告，噪声标签不是一个类别。

同一户三个seed是重复训练，不是三户独立样本。地图置换/标签置换规则沿用各实验原协议。窗口、缩放、地点输入和收益定义见HTML报告；当前35维方案没有折叠有效偏置。

## Exp39：粗区域标签监督（新增）

在51个唯一坐标的单位球三维坐标上定义K=3/4/5个区域；主配置K=4、35维、全部家庭。正样本为同区域不同坐标家庭，5折留出坐标，199次坐标标签置换；同时运行exclude_Rain。原始θ使用相同最近区域中心分类器作为对照。

```sh
python code/gps/run_region_suite.py
python code/gps/summarize_region_supervision.py
python code/gps/build_dimension_comparison.py
```

输出 `results/gps/region_supervision_exp39/`，入口拒绝覆盖。预定配置见 `protocol_before_results.json`，独立报告为 `report.html`。结构、损失参考师兄代码，采样和评估已改造；并非原URL实验的逐项完整复现。
