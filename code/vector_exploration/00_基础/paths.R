# 本目录所有 R 脚本共用的路径与坐标定义，与 paths.py 保持一致。
# 脚本不写字面量路径，一律从这里取。
#
# ROOT 由本文件自身位置回推，而不是相对当前工作目录。R 的 source() 不把被引文件的
# 路径传给被引方，所以先从命令行的 --file= 取调用方脚本位置，交互式运行时退回 getwd()。
# 子目录里的脚本这样引入：
#   source(file.path("..", "00_基础", "paths.R"))

.caller_dir <- function() {
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}

# 从调用方所在的编号子目录往上一级，再进 00_基础
.PATHS_DIR <- {
  d <- .caller_dir()
  if (file.exists(file.path(d, "paths.R"))) d
  else file.path(dirname(d), "00_基础")
}

# .../energy/06.github/code/vector_exploration/00_基础 -> .../energy
ROOT <- normalizePath(file.path(.PATHS_DIR, "..", "..", "..", ".."), mustWork = FALSE)

RESULTS   <- file.path(ROOT, "04.results")
CEM       <- file.path(RESULTS, "cem_npy")
SEED_ROOT <- file.path(CEM, "seed_wise")             # cem_seed0/1/42，三次独立重训
CANONICAL <- file.path(CEM, "42_vectors_71_1.0")     # seed=42 正式版
MULTISEED <- file.path(CEM, "multiseed_attractor_test")
SHAPE        <- file.path(RESULTS, "shape")
FEATURES_CSV <- file.path(SHAPE, "_inputs", "household_features_full.csv")

# 输出文件名 -> 实验目录；一个脚本会一次产出分属不同实验的多个文件，故按文件名路由
.SHAPE_ROUTE <- c(
  power_1d.csv = "exp1_power", power_multivariate.csv = "exp1_power",
  knn_stability.csv = "exp4_knn_stability",
  fiedler_coordinate.csv = "exp7_fiedler", fiedler_cross_seed.csv = "exp7_fiedler",
  dip_1d_projections.csv = "exp11_dip", dip_pairwise_mc.csv = "exp11_dip",
  dip_realworld_features.csv = "exp11_dip", dip_rowwise.csv = "exp11_dip"
)

# 未登记的文件名放进 _unrouted/ 而不是静默乱放
shape_out <- function(filename) {
  sub <- .SHAPE_ROUTE[[filename]]
  if (is.null(sub)) sub <- "_unrouted"
  d <- file.path(SHAPE, sub)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, filename)
}

exp_dir <- function(name) {
  d <- file.path(SHAPE, name)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# 坐标约定。两个，按分析的性质选，调用处用函数名声明意图。
LOCATION_COL <- 4   # 1-based；对应 θ 的 dim3
BIAS_COL     <- 36  # 1-based；对应 θ 的 dim35

# 家庭之间算距离或几何时用的坐标，排除 location 与 bias：location 在单户内恒定，
# 优化器能确定的只是 w_loc*s_loc+b 这个和，拆分由随机游走决定，计入距离即纯噪声。
dims_geometry <- function(n_dims) setdiff(seq_len(n_dims), c(LOCATION_COL, BIAS_COL))

# 逐坐标统计、或要把 θ 当策略执行时用的全部坐标
dims_all <- function(n_dims) seq_len(n_dims)

# 同一约定的列名形式，供按列名排除的聚类脚本使用（θ 列名为 dim0..dim35）
GEOMETRY_EXCLUDE_COLS <- c("dim3", "dim35")

# θ 矩阵 -> 标准化后的几何坐标
to_space <- function(theta) scale(theta[, dims_geometry(ncol(theta)), drop = FALSE])
