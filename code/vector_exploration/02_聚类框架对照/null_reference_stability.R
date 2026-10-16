#!/usr/bin/env Rscript
# 逐维打乱构造保留边际分布、破坏维间关联的零参照，并比较固定 K 下 PAM 的 bootstrap
# 稳定性。候选 K 从真实数据读取，零数据中未重跑选模流程，因此百分位仅作探索性比较。

suppressMessages({
  library(RcppCNPy)
  library(cluster)
  library(fpc)
})

source(file.path("..", "00_基础", "paths.R"))
SPACE_NAME <- "identifiable34"   # 见 paths.R：排除 location 与 bias

args <- commandArgs(trailingOnly = TRUE)
VECTORS_DIR <- if (length(args) >= 1) args[1] else CANONICAL
BASE_OUTPUT_DIR <- if (length(args) >= 2) args[2] else file.path(SHAPE, "exp9_bootstrap_null")

N_NULL_DATASETS <- 100
N_BOOTSTRAP_NULL <- 200
SEED <- 42

# ---- 读取真实theta向量 ----
files <- sort(list.files(VECTORS_DIR, pattern = "_expert[.]npy$", full.names = TRUE))
ids <- sub("_expert[.]npy$", "", basename(files))
theta <- t(sapply(files, npyLoad))
rownames(theta) <- ids
colnames(theta) <- paste0("dim", 0:(ncol(theta) - 1))

pamCBI <- function(data, k, diss = TRUE, ...) {
  c1 <- pam(data, k, diss = diss, ...)
  cl <- c1$clustering
  list(result = c1, nc = k, clusterlist = lapply(1:k, function(i) cl == i), partition = cl, clustermethod = "pam")
}

run_null_reference_for_space <- function(space_name, exclude_cols, pam_dir, output_dir) {
  cat(sprintf("\n========== 特征空间 %s ==========\n", space_name))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  stability_csv <- file.path(pam_dir, "bootstrap_stability_r.csv")
  if (!file.exists(stability_csv)) {
    cat(sprintf("警告：找不到 %s，请先跑 pam_cluster_archetypes.R 产出这个空间的候选K，跳过。\n", stability_csv))
    return(NULL)
  }
  candidate_ks <- unique(read.csv(stability_csv)$k)
  cat(sprintf(">>> 候选K: %s\n", paste(candidate_ks, collapse = ", ")))

  x <- theta[, !(colnames(theta) %in% exclude_cols), drop = FALSE]
  x_std <- scale(x)
  cat(sprintf(">>> 该空间维度: %d\n", ncol(x_std)))

  set.seed(SEED)
  d_real <- dist(x_std, method = "euclidean")
  real_stats <- list()
  for (k in candidate_ks) {
    cb <- clusterboot(d_real, B = N_BOOTSTRAP_NULL, distances = TRUE, clustermethod = pamCBI,
                       k = k, seed = SEED, count = FALSE)
    real_stats[[as.character(k)]] <- min(cb$bootmean)
    cat(sprintf(">>> 真实数据 K=%d: min_jaccard=%.3f\n", k, real_stats[[as.character(k)]]))
  }

  cat(sprintf(">>> 开始零假设模拟：%d 次独立打乱，每次跑PAM+bootstrap(B=%d)\n", N_NULL_DATASETS, N_BOOTSTRAP_NULL))
  null_results <- list()
  for (k in candidate_ks) null_results[[as.character(k)]] <- numeric(N_NULL_DATASETS)

  for (i in 1:N_NULL_DATASETS) {
    x_null <- apply(x_std, 2, sample)
    d_null <- dist(x_null, method = "euclidean")
    for (k in candidate_ks) {
      cb <- clusterboot(d_null, B = N_BOOTSTRAP_NULL, distances = TRUE, clustermethod = pamCBI,
                         k = k, seed = SEED + i, count = FALSE)
      null_results[[as.character(k)]][i] <- min(cb$bootmean)
    }
    if (i %% 20 == 0) cat(sprintf("  已完成 %d/%d\n", i, N_NULL_DATASETS))
  }

  summary_rows <- list()
  for (k in candidate_ks) {
    ks <- as.character(k)
    null_vals <- null_results[[ks]]
    real_val <- real_stats[[ks]]
    percentile <- mean(null_vals < real_val) * 100
    p95 <- as.numeric(quantile(null_vals, 0.95))
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      space = space_name, k = k, real_min_jaccard = real_val,
      null_mean = mean(null_vals), null_sd = sd(null_vals), null_p95 = p95,
      real_exceeds_null_p95 = real_val > p95, real_percentile_vs_null = percentile
    )
    cat(sprintf("  K=%d: 真实值=%.3f | 零假设分布 mean=%.3f sd=%.3f p95=%.3f -> 第%.0f百分位, %s\n",
                k, real_val, mean(null_vals), sd(null_vals), p95, percentile,
                ifelse(real_val > p95, "超过零假设p95，支持存在真实结构", "没有明显超过零假设，证据不足")))
  }
  summary_df <- do.call(rbind, summary_rows)
  out_path <- file.path(output_dir, "null_reference_stability.csv")
  write.csv(summary_df, out_path, row.names = FALSE)
  cat(sprintf(">>> [%s] 结果已保存: %s\n", space_name, out_path))
  summary_df
}

METHOD_DIR <- file.path(BASE_OUTPUT_DIR, "null_reference")
res <- run_null_reference_for_space(SPACE_NAME, GEOMETRY_EXCLUDE_COLS,
                                    file.path(BASE_OUTPUT_DIR, "pam_ward"), METHOD_DIR)
if (!is.null(res)) {
  print(res, row.names = FALSE)
  write.csv(res, file.path(METHOD_DIR, "null_reference_summary.csv"), row.names = FALSE)
  cat(sprintf("\n>>> 已保存: %s\n", file.path(METHOD_DIR, "null_reference_summary.csv")))
}
