#!/usr/bin/env Rscript
# 探索 PAM/Ward 分组、silhouette 与 bootstrap 稳定性。候选 K 先由真实数据筛选，
# 因而输出用于描述和后续探索，不是经过完整选模校正的确认性检验。
# 使用 paths.R 定义的 34 维几何坐标；通过筛选的 K 会输出成员及 medoid。

suppressMessages({
  library(RcppCNPy)
  library(cluster)
  library(fpc)
})

source(file.path("..", "00_基础", "paths.R"))
SPACE_NAME <- "identifiable34"   # 见 paths.R：排除 location 与 bias

args <- commandArgs(trailingOnly = TRUE)
VECTORS_DIR <- if (length(args) >= 1) args[1] else CANONICAL
BASE_OUTPUT_DIR <- if (length(args) >= 2) args[2] else file.path(SHAPE, "exp8_cluster_methods")

K_RANGE <- 2:10
N_BOOTSTRAP <- 200
SEED <- 42

# ---- 读取 theta 向量 ----
files <- sort(list.files(VECTORS_DIR, pattern = "_expert[.]npy$", full.names = TRUE))
ids <- sub("_expert[.]npy$", "", basename(files))
theta <- t(sapply(files, npyLoad))
rownames(theta) <- ids
colnames(theta) <- paste0("dim", 0:(ncol(theta) - 1))
cat(sprintf(">>> 载入 %d 户，theta 维度 %d\n", nrow(theta), ncol(theta)))

pamCBI <- function(data, k, diss = TRUE, ...) {
  c1 <- pam(data, k, diss = diss, ...)
  cl <- c1$clustering
  clusterlist <- lapply(1:k, function(i) cl == i)
  list(result = c1, nc = k, clusterlist = clusterlist, partition = cl, clustermethod = "pam")
}

run_pipeline_for_space <- function(space_name, exclude_cols, output_dir) {
  cat(sprintf("\n========== 特征空间 %s (排除: %s) ==========\n", space_name,
              ifelse(length(exclude_cols) == 0, "无", paste(exclude_cols, collapse = ","))))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  x <- theta[, !(colnames(theta) %in% exclude_cols), drop = FALSE]
  x_std <- scale(x)
  cat(sprintf(">>> 该空间维度: %d\n", ncol(x_std)))
  d <- dist(x_std, method = "euclidean")

  set.seed(SEED)

  # ---- 阶段1：silhouette 扫描 (K-Medoids vs Ward) ----
  sweep_rows <- list()
  for (k in K_RANGE) {
    pam_fit <- pam(d, k, diss = TRUE)
    ward_fit <- cutree(hclust(d, method = "ward.D2"), k = k)
    sil_pam <- mean(silhouette(pam_fit$clustering, d)[, "sil_width"])
    sil_ward <- mean(silhouette(ward_fit, d)[, "sil_width"])
    sweep_rows[[length(sweep_rows) + 1]] <- data.frame(k = k, kmedoids_silhouette = sil_pam, ward_silhouette = sil_ward)
  }
  sweep_df <- do.call(rbind, sweep_rows)
  write.csv(sweep_df, file.path(output_dir, "silhouette_sweep_r.csv"), row.names = FALSE)
  cat(">>> 阶段1：silhouette 扫描\n")
  print(sweep_df, row.names = FALSE)

  sweep_df$combined <- (sweep_df$kmedoids_silhouette + sweep_df$ward_silhouette) / 2
  candidate_ks <- sweep_df$k[order(-sweep_df$combined)][1:4]
  cat(sprintf(">>> 候选K（按平均silhouette前4）: %s\n", paste(candidate_ks, collapse = ", ")))

  # ---- 阶段2：bootstrap 稳定性 (fpc::clusterboot，逐簇Jaccard，取min不取mean) ----
  cat(">>> 阶段2：bootstrap 稳定性过滤 (Jaccard, Hennig 2007)\n")
  stability_rows <- list()
  stable_ks <- c()
  for (k in candidate_ks) {
    cb <- clusterboot(d, B = N_BOOTSTRAP, distances = TRUE, clustermethod = pamCBI,
                       k = k, seed = SEED, count = FALSE)
    per_cluster_jaccard <- cb$bootmean
    min_jaccard <- min(per_cluster_jaccard)
    mean_jaccard <- mean(per_cluster_jaccard)
    stable <- min_jaccard >= 0.6
    stability_rows[[length(stability_rows) + 1]] <- data.frame(
      space = space_name, k = k, min_jaccard_stability = min_jaccard,
      mean_jaccard_stability = mean_jaccard, stable = stable
    )
    cat(sprintf("  K=%d: 逐簇Jaccard=[%s] -> min=%.3f, mean=%.3f -> %s\n", k,
                paste(sprintf("%.3f", per_cluster_jaccard), collapse = ", "),
                min_jaccard, mean_jaccard, ifelse(stable, "稳定", "UNSTABLE, dropped")))
    if (stable) stable_ks <- c(stable_ks, k)
  }
  stability_df <- do.call(rbind, stability_rows)
  write.csv(stability_df, file.path(output_dir, "bootstrap_stability_r.csv"), row.names = FALSE)

  if (length(stable_ks) == 0) {
    cat(sprintf(">>> [%s] 没有候选K通过稳定性筛选。\n", space_name))
    return(stability_df)
  }

  for (k in stable_ks) {
    pam_fit <- pam(d, k, diss = TRUE)
    medoid_ids <- ids[pam_fit$id.med]
    membership <- data.frame(household_id = ids, cluster = pam_fit$clustering - 1,
                              is_medoid = ids %in% medoid_ids)
    out_path <- file.path(output_dir, sprintf("membership_k%d.csv", k))
    write.csv(membership, out_path, row.names = FALSE)
    cat(sprintf(">>> [%s] K=%d medoid户(真实户号): %s -> %s\n", space_name, k,
                paste(medoid_ids, collapse = ", "), out_path))
  }
  stability_df
}

METHOD_DIR <- file.path(BASE_OUTPUT_DIR, "pam_ward")
stability <- run_pipeline_for_space(SPACE_NAME, GEOMETRY_EXCLUDE_COLS, METHOD_DIR)
write.csv(stability, file.path(METHOD_DIR, "stability_summary.csv"), row.names = FALSE)
cat(sprintf("\n>>> 已保存: %s\n", file.path(METHOD_DIR, "stability_summary.csv")))
