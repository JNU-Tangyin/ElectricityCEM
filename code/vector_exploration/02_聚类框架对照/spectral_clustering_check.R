#!/usr/bin/env Rscript
# 谱聚类探索：kernlab 使用 RBF 相似度并扫描 K；Spectrum 使用自适应核和 eigengap。
# 两者均在同一数据上调参，输出用于方法敏感性比较，不是独立验证。

suppressMessages({
  library(RcppCNPy)
  library(kernlab)
  library(Spectrum)
})

source(file.path("..", "00_基础", "paths.R"))
SPACE_NAME <- "identifiable34"   # 见 paths.R：排除 location 与 bias

VECTORS_DIR <- CANONICAL
BASE_OUTPUT_DIR <- file.path(SHAPE, "exp8_cluster_methods")

K_RANGE <- 2:4
SEED <- 42

files <- sort(list.files(VECTORS_DIR, pattern = "_expert[.]npy$", full.names = TRUE))
ids <- sub("_expert[.]npy$", "", basename(files))
theta <- t(sapply(files, npyLoad))
rownames(theta) <- ids
colnames(theta) <- paste0("dim", 0:(ncol(theta) - 1))
cat(sprintf(">>> 载入 %d 户，theta 维度 %d\n", nrow(theta), ncol(theta)))

run_spectral_for_space <- function(space_name, exclude_cols, output_dir) {
  cat(sprintf("\n========== 特征空间 %s ==========\n", space_name))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  x <- theta[, !(colnames(theta) %in% exclude_cols), drop = FALSE]
  x_std <- scale(x)

  # ---- kernlab::specc：全连接RBF核，K需要手动指定，扫K_RANGE ----
  cat(">>> kernlab::specc (RBF核，全连接相似度图)\n")
  specc_rows <- list()
  for (k in K_RANGE) {
    set.seed(SEED)
    fit <- specc(x_std, centers = k)
    clustering <- as.integer(fit)
    sizes <- table(clustering)
    cat(sprintf("  K=%d: 各簇大小=%s\n", k, paste(sizes, collapse = ",")))
    min_size <- min(sizes)
    if (min_size <= 10) {
      minority_label <- names(sizes)[which.min(sizes)]
      members <- ids[clustering == as.integer(minority_label)]
      cat(sprintf("    最小簇(n=%d)成员: %s\n", min_size, paste(sort(members), collapse = ", ")))
    }
    for (i in seq_along(ids)) {
      specc_rows[[length(specc_rows) + 1]] <- data.frame(
        method = "kernlab_specc", k = k, household_id = ids[i], cluster = clustering[i]
      )
    }
  }
  specc_df <- do.call(rbind, specc_rows)

  # ---- Spectrum：自适应核 + eigengap自动定K，不需要手动指定 ----
  cat("\n>>> Spectrum (自适应密度核，特征值间隙自动定K)\n")
  # Spectrum要求数据是"特征为行、样本为列"(基因组学惯例)，跟我们平时的行=样本相反，需要转置。
  spectrum_input <- as.data.frame(t(x_std))
  set.seed(SEED)
  spec_fit <- Spectrum(spectrum_input, showres = FALSE, silent = TRUE)
  auto_k <- length(unique(spec_fit$assignments))
  cat(sprintf(">>> Spectrum自动选出的K=%d\n", auto_k))
  sizes <- table(spec_fit$assignments)
  cat(sprintf("  各簇大小=%s\n", paste(sizes, collapse = ",")))
  min_size <- min(sizes)
  if (min_size <= 10) {
    minority_label <- names(sizes)[which.min(sizes)]
    members <- ids[spec_fit$assignments == as.integer(minority_label)]
    cat(sprintf("  最小簇(n=%d)成员: %s\n", min_size, paste(sort(members), collapse = ", ")))
  }
  spectrum_df <- data.frame(
    method = "Spectrum_autoK", k = auto_k, household_id = ids, cluster = spec_fit$assignments
  )

  result <- rbind(specc_df, spectrum_df)
  out_path <- file.path(output_dir, "spectral_clustering_r.csv")
  write.csv(result, out_path, row.names = FALSE)
  cat(sprintf("\n>>> 已保存: %s\n", out_path))

  result
}

METHOD_DIR <- file.path(BASE_OUTPUT_DIR, "spectral")
run_spectral_for_space(SPACE_NAME, GEOMETRY_EXCLUDE_COLS, METHOD_DIR)
