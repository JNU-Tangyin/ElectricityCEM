#!/usr/bin/env Rscript
# 在同一标准化向量上拟合高斯混合模型，并由 BIC 在指定范围内选择 K。
# 这是探索性模型比较；小样本、高维下，BIC 选择不等于确认存在离散簇。

suppressMessages({
  library(RcppCNPy)
  library(mclust)
})

source(file.path("..", "00_基础", "paths.R"))
SPACE_NAME <- "identifiable34"   # 见 paths.R：排除 location 与 bias

VECTORS_DIR <- CANONICAL
BASE_OUTPUT_DIR <- file.path(SHAPE, "exp8_cluster_methods")

K_RANGE <- 2:10
SEED <- 42

files <- sort(list.files(VECTORS_DIR, pattern = "_expert[.]npy$", full.names = TRUE))
ids <- sub("_expert[.]npy$", "", basename(files))
theta <- t(sapply(files, npyLoad))
rownames(theta) <- ids
colnames(theta) <- paste0("dim", 0:(ncol(theta) - 1))
cat(sprintf(">>> 载入 %d 户，theta 维度 %d\n", nrow(theta), ncol(theta)))

run_gmm_for_space <- function(space_name, exclude_cols, output_dir) {
  cat(sprintf("\n========== 特征空间 %s ==========\n", space_name))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  x <- theta[, !(colnames(theta) %in% exclude_cols), drop = FALSE]
  x_std <- scale(x)

  set.seed(SEED)
  # G=K_RANGE：让mclust在这个范围内同时搜索协方差结构和K，用BIC独立选最优K，
  gmm_fit <- Mclust(x_std, G = K_RANGE, verbose = FALSE)
  cat(sprintf(">>> BIC选出的最优K=%d，协方差结构=%s\n", gmm_fit$G, gmm_fit$modelName))
  bic_table <- data.frame(k = K_RANGE)
  bic_table$bic <- sapply(K_RANGE, function(k) {
    fit_k <- tryCatch(Mclust(x_std, G = k, verbose = FALSE), error = function(e) NULL)
    if (is.null(fit_k)) NA else fit_k$bic
  })
  write.csv(bic_table, file.path(output_dir, "gmm_bic_sweep.csv"), row.names = FALSE)
  cat(">>> BIC扫描(K越大BIC越高越好，mclust约定BIC取正、越大越优):\n")
  print(bic_table, row.names = FALSE)

  # 直接用BIC选出的最优模型的后验概率，不额外强制跑某个特定K。
  posterior <- gmm_fit$z
  colnames(posterior) <- paste0("posterior_", 0:(gmm_fit$G - 1))
  max_posterior <- apply(posterior, 1, max)

  result <- data.frame(
    household_id = ids,
    gmm_cluster = gmm_fit$classification - 1,
    max_posterior = max_posterior,
    posterior
  )

  low_confidence <- result[result$max_posterior < 0.8, ]
  cat(sprintf(">>> 后验概率最大值<0.8(分配不确定)的户数: %d/%d\n", nrow(low_confidence), nrow(result)))
  if (nrow(low_confidence) > 0) {
    print(low_confidence[, c("household_id", "gmm_cluster", "max_posterior")], row.names = FALSE)
  }

  cat(">>> 各簇大小:\n")
  print(table(result$gmm_cluster))

  out_path <- file.path(output_dir, sprintf("gmm_bic_k%d_posterior.csv", gmm_fit$G))
  write.csv(result, out_path, row.names = FALSE)
  cat(sprintf(">>> 已保存: %s\n", out_path))

  result
}

METHOD_DIR <- file.path(BASE_OUTPUT_DIR, "gmm")
run_gmm_for_space(SPACE_NAME, GEOMETRY_EXCLUDE_COLS, METHOD_DIR)
