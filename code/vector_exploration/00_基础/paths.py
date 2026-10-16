# -*- coding: utf-8 -*-
import os
import sys

# .../energy/06.github/code/vector_exploration/00_基础/paths.py -> .../energy
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 4)))
RESULTS = os.path.join(ROOT, "04.results")
DATA = os.path.join(ROOT, "00.data", "preprocessed", "german_households_weather")
CODE = os.path.join(ROOT, "02.code", "multiple_country")  # EnergyStorageEnv 等

# 脚本按模块名跨目录导入共用代码。import paths 时将编号目录加入 sys.path，
# 避免每个脚本重复配置导入路径。
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(PKG)):
    _p = os.path.join(PKG, _d)
    if os.path.isdir(_p) and not _d.startswith((".", "_")) and _p not in sys.path:
        sys.path.append(_p)

# θ 向量与逐户训练历史
CEM = os.path.join(RESULTS, "cem_npy")
SEED_ROOT = os.path.join(CEM, "seed_wise")                 # cem_seed0/1/42，三次独立重训
CANONICAL = os.path.join(CEM, "42_vectors_71_1.0")         # seed=42 正式版
MULTISEED = os.path.join(CEM, "multiseed_attractor_test")  # 10 户 × 10 种子

# 结构诊断结果，按实验编号
SHAPE = os.path.join(RESULTS, "shape")
FEATURES_CSV = os.path.join(SHAPE, "_inputs", "household_features_full.csv")

# 输出文件名 -> 实验目录
_ROUTE = {
    "power_analysis.csv": "exp1_power",
    "power_1d.csv": "exp1_power",
    "power_multivariate.csv": "exp1_power",
    "hopkins_nn.csv": "exp2_hopkins_nn",
    "isomap_d02.csv": "exp3_intrinsic_dim",
    "isomap_residual.csv": "exp3_intrinsic_dim",
    "pca_scree.csv": "exp3_intrinsic_dim",
    "knn_stability.csv": "exp4_knn_stability",
    "pair_coclustering_reproducibility.csv": "exp5_pair_coclustering",
    "pair_cofreq.csv": "exp5_pair_coclustering",
    "fiedler_coordinate.csv": "exp7_fiedler",
    "fiedler_cross_seed.csv": "exp7_fiedler",
    "theta_feature_screening.csv": "exp12_theta_screening",
    "dip_1d_projections.csv": "exp11_dip",
    "dip_pairwise_mc.csv": "exp11_dip",
    "dip_realworld_features.csv": "exp11_dip",
    "dip_rowwise.csv": "exp11_dip",
}

# 坐标约定。两个，按分析的性质选，调用处用函数名声明意图。
LOCATION_DIM, BIAS_DIM = 3, 35


def dims_geometry(n_dims):
    """家庭之间算距离或几何时用的坐标，排除 location 与 bias。

    location 在单户内恒定，优化器能确定的只是 w_loc*s_loc+b 这个和，拆成两个数的
    方式由 CEM 的随机游走决定。两户行为相同时这两维仍可以差很远，计入距离即纯噪声。
    """
    return sorted(set(range(n_dims)) - {LOCATION_DIM, BIAS_DIM})


def dims_all(n_dims):
    """逐坐标统计、或要把 θ 当策略执行时用的全部坐标。

    策略向量缺维度就跑不了环境；而逐坐标统计里这两维各自成列，不与其它维混在一起，
    不可识别性不会污染别的坐标。
    """
    return list(range(n_dims))


def to_space(theta):
    """θ 矩阵 -> 标准化后的几何坐标。标准化在传入的全部行上拟合一次。"""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(theta[:, dims_geometry(theta.shape[1])])


def exp_dir(name, makedirs=True):
    """按实验目录名取路径，如 exp8_cluster_methods。"""
    d = os.path.join(SHAPE, name)
    if makedirs:
        os.makedirs(d, exist_ok=True)
    return d


def shape_out(filename, makedirs=True):
    """该输出文件应落在的完整路径；未登记的文件名放进 _unrouted/ 而不是静默乱放。"""
    d = os.path.join(SHAPE, _ROUTE.get(filename, "_unrouted"))
    if makedirs:
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)
