#!/usr/bin/env Rscript
# 多峰性、距离分布与连续结构的探索性诊断，附模拟功效分析。
# 距离池化/逐行检验存在依赖；由同一数据选择的投影或特征也会影响常规 p 值。
# 本脚本含 seed 42 少数簇名单及按 zero_load_frac 定向的 Fiedler 分析；相关结果属事后探索，
# 不应作为无偏的确认性检验引用。

suppressMessages({
  library(RcppCNPy)
  library(diptest)
  library(MASS)
  library(multimode)
})

source(file.path("..", "00_基础", "paths.R"))
# 历史探索名单：用于敏感性/功效计算，不是预先指定的验证标签。
MINORITY_42 <- c("10338", "10408", "10495", "10676", "11130", "11424", "11450")

N_MC <- 2000          # (B) 蒙特卡洛零假设重复次数
N_POW_MV <- 300       # (D) 多元功效分析每个 delta 的重复次数
N_POW_1D <- 200       # (D) 一维功效分析每个 delta 的重复次数
B_SILV <- 500         # Silverman 检验的 bootstrap 次数（(A)用）
B_SILV_POW <- 200     # 功效分析里的 Silverman bootstrap 次数（要跑很多次，降低一些）
SEED <- 42

set.seed(SEED)


load_theta <- function(dir_path) {
  files <- sort(list.files(dir_path, pattern = "_expert[.]npy$", full.names = TRUE))
  ids <- sub("_expert[.]npy$", "", basename(files))
  theta <- t(sapply(files, npyLoad))
  rownames(theta) <- ids
  theta
}


dip_stat <- function(x) as.numeric(dip(x))
silv_p <- function(x, B) modetest(x, mod0 = 1, method = "SI", B = B)$p.value

# 分节开关：--sections=ABCDEF 任意子集。(D2) 的 Silverman 功效分析要跑约 20 分钟。
args_cli <- commandArgs(trailingOnly = TRUE)
sec_arg <- grep("^--sections=", args_cli, value = TRUE)
SECTIONS <- if (length(sec_arg)) toupper(sub("^--sections=", "", sec_arg[1])) else "ABCDEF"
run_sec <- function(x) grepl(x, SECTIONS, fixed = TRUE)
cat(sprintf(">>> 本次运行的小节: %s\n\n", SECTIONS))

# 真实世界特征表：(A) 和 (E) 都要用，放在分节开关外面加载
feat <- read.csv(FEATURES_CSV, colClasses = c(household_id = "character"))

# 只取真正含 θ 向量的子目录 —— SEED_ROOT 下还有 consolidated/ 这类派生产物目录
seed_dirs <- sort(list.dirs(SEED_ROOT, recursive = FALSE))
seed_dirs <- seed_dirs[sapply(seed_dirs, function(d)
  length(list.files(d, pattern = "_expert[.]npy$")) > 0)]
cat(sprintf(">>> seed 目录: %s\n", paste(basename(seed_dirs), collapse = ", ")))

# ======================================================================
if (run_sec("A")) {
cat("========================================================\n")
cat(">>> (A) 一维多峰性探索（投影/特征选择影响常规 p 值解释）\n")
cat("========================================================\n")
rows_a <- list()
for (sd_path in seed_dirs) {
  sd_name <- basename(sd_path)
  theta <- load_theta(sd_path)
  Xc <- to_space(theta)
  isc <- rownames(theta) %in% MINORITY_42

  soc_w  <- abs(theta[, 0 + 1])
  time_w <- sqrt(theta[, 4 + 1]^2 + theta[, 5 + 1]^2)
  pca    <- prcomp(Xc, center = FALSE, scale. = FALSE)

  for (nm in c("time_w", "soc_w", "PC1", "PC2")) {
    v <- switch(nm, time_w = time_w, soc_w = soc_w,
                PC1 = pca$x[, 1], PC2 = pca$x[, 2])
    # 少数组 vs 其余的标准化分离度（合并SD），用于跟 (D) 的功效曲线对照
    psd <- sqrt(((sum(isc) - 1) * var(v[isc]) + (sum(!isc) - 1) * var(v[!isc])) / (length(v) - 2))
    rows_a[[length(rows_a) + 1]] <- data.frame(
      seed_dir = sd_name, variable = nm, n = length(v),
      dip = dip_stat(v), dip_p = dip.test(v)$p.value,
      silverman_p = silv_p(v, B_SILV),
      minority_mean = mean(v[isc]), rest_mean = mean(v[!isc]),
      separation_sd = abs(mean(v[!isc]) - mean(v[isc])) / psd
    )
  }
}
df_a <- do.call(rbind, rows_a)
# BH 仅校正本表中的检验数；不校正此前按簇/结果选择变量和投影造成的选择偏差。
df_a$dip_p_bh <- p.adjust(df_a$dip_p, method = "BH")
df_a$silverman_p_bh <- p.adjust(df_a$silverman_p, method = "BH")
print(df_a, row.names = FALSE, digits = 3)

cat("\n>>> 真实世界数据特征\n")
rows_f <- list()
for (nm in c("zero_load_frac", "load_cv", "load_mean", "pv_mean")) {
  if (!nm %in% names(feat)) next
  v <- feat[[nm]][is.finite(feat[[nm]])]
  rows_f[[length(rows_f) + 1]] <- data.frame(
    variable = nm, n = length(v), dip = dip_stat(v),
    dip_p = dip.test(v)$p.value, silverman_p = silv_p(v, B_SILV)
  )
}
df_f <- do.call(rbind, rows_f)
df_f$dip_p_bh <- p.adjust(df_f$dip_p, method = "BH")
df_f$silverman_p_bh <- p.adjust(df_f$silverman_p, method = "BH")
print(df_f, row.names = FALSE, digits = 3)

}

# ======================================================================
if (run_sec("B")) {
cat("\n========================================================\n")
cat(">>> (B) 两两距离的 dip，零假设 = 单个多元高斯（连续、单峰、可拉长可相关）\n")
cat(sprintf("    蒙特卡洛 %d 次；p 小 = 比单个连续团块更多峰 = 支持离散\n", N_MC))
cat("========================================================\n")
rows_b <- list()
for (sd_path in seed_dirs) {
  Xc <- to_space(load_theta(sd_path))
  d_obs <- dip_stat(as.vector(dist(Xc)))
  mu <- colMeans(Xc); Sigma <- cov(Xc)
  nullv <- replicate(N_MC, dip_stat(as.vector(dist(MASS::mvrnorm(nrow(Xc), mu, Sigma)))))
  rows_b[[length(rows_b) + 1]] <- data.frame(
    seed_dir = basename(sd_path), dip_obs = d_obs,
    null_mean = mean(nullv), null_p95 = as.numeric(quantile(nullv, 0.95)),
    p_value = (1 + sum(nullv >= d_obs)) / (1 + N_MC)
  )
}
df_b <- do.call(rbind, rows_b)
print(df_b, row.names = FALSE, digits = 4)

}

# ======================================================================
if (run_sec("C")) {
cat("\n========================================================\n")
cat(">>> (C) 逐行 dip（dip-means 做法）：每户到其余户的距离分布是否多峰\n")
cat("========================================================\n")
rows_c <- list()
for (sd_path in seed_dirs) {
  theta <- load_theta(sd_path)
  D <- as.matrix(dist(to_space(theta)))
  pvals <- sapply(seq_len(nrow(D)), function(i) dip.test(D[i, -i])$p.value)
  p_bh <- p.adjust(pvals, method = "BH")
  sig <- which(p_bh < 0.05)
  rows_c[[length(rows_c) + 1]] <- data.frame(
    seed_dir = basename(sd_path), n_households = nrow(D),
    n_sig_raw = sum(pvals < 0.05), n_sig_bh = length(sig), min_p_bh = min(p_bh),
    sig_households = if (length(sig)) paste(rownames(D)[sig], collapse = ",") else ""
  )
}
df_c <- do.call(rbind, rows_c)
print(df_c, row.names = FALSE, digits = 4)

}

# ======================================================================
if (run_sec("D")) {
cat("\n========================================================\n")
cat(">>> (D1) 多元功效分析：64点主团 + 7点沿观测到的少数组方向偏移 delta\n")
cat("     检验 = (B) 的两两距离 dip；判读阈值 = 高斯零假设的 95 分位\n")
cat("========================================================\n")
theta42 <- load_theta(file.path(SEED_ROOT, "cem_seed42"))
X42 <- to_space(theta42)
isc42 <- rownames(theta42) %in% MINORITY_42
cc <- colMeans(X42[isc42, , drop = FALSE]); cr <- colMeans(X42[!isc42, , drop = FALSE])
sep_obs_mv <- sqrt(sum((cc - cr)^2))
dir_vec <- (cc - cr) / sep_obs_mv
cat(sprintf("     观测到的少数组质心 vs 其余质心距离 = %.2f（标准化空间）\n", sep_obs_mv))

mu42 <- colMeans(X42); Sig42 <- cov(X42)
nullv42 <- replicate(1500, dip_stat(as.vector(dist(MASS::mvrnorm(71, mu42, Sig42)))))
thr42 <- as.numeric(quantile(nullv42, 0.95))
rows_d1 <- list()
for (delta in c(0, 2, 4, 6, 8, 10, 12)) {
  hits <- replicate(N_POW_MV, {
    M <- MASS::mvrnorm(71, mu42, Sig42)
    M[1:7, ] <- M[1:7, ] + delta * dir_vec
    dip_stat(as.vector(dist(M))) > thr42
  })
  rows_d1[[length(rows_d1) + 1]] <- data.frame(delta = delta, detect_rate = mean(hits))
}
df_d1 <- do.call(rbind, rows_d1)
df_d1$note <- ifelse(abs(df_d1$delta - sep_obs_mv) <= 1.5, "<- 接近真实观测分离度", "")
print(df_d1, row.names = FALSE, digits = 3)

cat("\n========================================================\n")
cat(">>> (D2) 一维功效分析：64点 N(0,1) + 7点 N(delta,1)，dip 与 Silverman 对比\n")
cat("========================================================\n")
rows_d2 <- list()
for (delta in c(0, 1, 2, 3, 4, 5)) {
  r <- replicate(N_POW_1D, {
    v <- c(rnorm(64, 0, 1), rnorm(7, delta, 1))
    c(dip.test(v)$p.value < 0.05, silv_p(v, B_SILV_POW) < 0.05)
  })
  rows_d2[[length(rows_d2) + 1]] <- data.frame(
    delta = delta, dip_detect = mean(r[1, ]), silverman_detect = mean(r[2, ])
  )
}
df_d2 <- do.call(rbind, rows_d2)
obs_rng <- range(df_a$separation_sd[df_a$variable == "time_w"])
df_d2$note <- ifelse(df_d2$delta >= obs_rng[1] - 0.6 & df_d2$delta <= obs_rng[2] + 0.6,
                      "<- time_w 真实观测区间", "")
print(df_d2, row.names = FALSE, digits = 3)
cat(sprintf("\n     time_w 上观测到的分离度跨 seed 范围: %.2f ~ %.2f SD\n", obs_rng[1], obs_rng[2]))

}

# ======================================================================
if (run_sec("E")) {
# (E) 用 Fiedler 坐标描述图上的连续位置，并与家庭特征作探索性比较。
cat("\n========================================================\n")
cat(">>> (E) 谱嵌入 Fiedler 坐标：θ流形上的连续位置\n")
cat("========================================================\n")

knn_graph <- function(X, k) {
  D <- as.matrix(dist(X)); n <- nrow(D)
  W <- matrix(0, n, n)
  for (i in seq_len(n)) W[i, order(D[i, ])[2:(k + 1)]] <- 1
  pmax(W, t(W))   # 对称化（mutual-OR），跟其它 kNN 图脚本一致
}

fiedler_vec <- function(W) {
  d <- rowSums(W)
  Dm <- diag(1 / sqrt(pmax(d, 1e-12)))
  L <- diag(nrow(W)) - Dm %*% W %*% Dm
  ev <- eigen(L, symmetric = TRUE)
  ev$vectors[, order(ev$values)[2]]   # 第二小特征值对应的特征向量
}

ph_path <- file.path(SHAPE, "exp10_multiseed_signal", "household_signal_summary.csv")
ph_df <- if (file.exists(ph_path)) read.csv(ph_path, colClasses = c(household = "character")) else NULL

K_FIEDLER <- 10
rows_e <- list(); fiedler_store <- list()
for (sd_path in seed_dirs) {
  sd_name <- basename(sd_path)
  theta <- load_theta(sd_path); Xc <- to_space(theta)
  fv <- fiedler_vec(knn_graph(Xc, K_FIEDLER))
  names(fv) <- rownames(theta)

  zlf <- feat$zero_load_frac[match(names(fv), feat$household_id)]
  # 符号任意；按 zero_load_frac 定向后，不再用普通相关 p 值检验该特征。
  if (!is.na(cor(fv, zlf, method = "spearman")) &&
      cor(fv, zlf, method = "spearman") < 0) fv <- -fv
  fiedler_store[[sd_name]] <- fv

  tw <- sqrt(theta[, 4 + 1]^2 + theta[, 5 + 1]^2)
  ct <- cor.test(fv, zlf, method = "spearman", exact = FALSE)
  ct_tw <- cor.test(fv, tw, method = "spearman", exact = FALSE)
  r_ph <- NA; p_ph <- NA; n_ph <- 0
  if (!is.null(ph_df)) {
    idx <- match(ph_df$household, names(fv))
    ok <- !is.na(idx)
    if (sum(ok) >= 5) {
      cp <- cor.test(fv[idx[ok]], ph_df$p_minority_basin[ok], method = "spearman", exact = FALSE)
      r_ph <- cp$estimate; p_ph <- cp$p.value; n_ph <- sum(ok)
    }
  }
  rows_e[[length(rows_e) + 1]] <- data.frame(
    seed_dir = sd_name, k = K_FIEDLER,
    dip_p_fiedler = dip.test(fv)$p.value,
    rho_zero_load_frac = ct$estimate, p_zero_load_frac = ct$p.value,
    rho_time_w = ct_tw$estimate, p_time_w = ct_tw$p.value,
    rho_p_h = r_ph, p_p_h = p_ph, n_p_h = n_ph
  )
}
df_e <- do.call(rbind, rows_e)
# zero_load_frac 用于给 Fiedler 坐标定向，相关及其 p 值仅作描述，不作推断。
print(df_e, row.names = FALSE, digits = 3)

cat("\n>>> Fiedler 坐标本身跨 seed 是否一致（定向后的 Spearman 相关）\n")
sn <- names(fiedler_store); rows_e2 <- list()
for (i in seq_along(sn)) for (j in seq_along(sn)) if (i < j) {
  common <- intersect(names(fiedler_store[[i]]), names(fiedler_store[[j]]))
  r <- cor(fiedler_store[[i]][common], fiedler_store[[j]][common], method = "spearman")
  rows_e2[[length(rows_e2) + 1]] <- data.frame(seed_a = sn[i], seed_b = sn[j], rho = r, n = length(common))
}
df_e2 <- do.call(rbind, rows_e2)
print(df_e2, row.names = FALSE, digits = 3)

}

# ======================================================================
if (run_sec("F")) {
# (F) 若要用 kNN 给新家庭分配起点，"谁是谁的近邻"必须跨训练稳定。
#     检验的是近邻**身份**的跨 seed 重叠，对照随机子集零假设——"近邻比随机 pair 更近"
#     是同义反复，纯噪声也满足，不能用。
cat("\n========================================================\n")
cat(">>> (F) 近邻身份跨 seed 稳定性（Jaccard），对照随机子集零假设\n")
cat("========================================================\n")

neighbor_sets <- function(X, k) {
  D <- as.matrix(dist(X))
  lapply(seq_len(nrow(D)), function(i) rownames(D)[order(D[i, ])[2:(k + 1)]])
}

jacc <- function(a, b) length(intersect(a, b)) / length(union(a, b))

rows_f2 <- list()
for (k in c(3, 5, 10)) {
  nb <- list()
  for (sd_path in seed_dirs) {
    theta <- load_theta(sd_path); Xc <- to_space(theta)
    rownames(Xc) <- rownames(theta)
    ns <- neighbor_sets(Xc, k); names(ns) <- rownames(theta)
    nb[[basename(sd_path)]] <- ns
  }
  sn2 <- names(nb)
  obs <- c()
  for (i in seq_along(sn2)) for (j in seq_along(sn2)) if (i < j) {
    common <- intersect(names(nb[[i]]), names(nb[[j]]))
    obs <- c(obs, sapply(common, function(h) jacc(nb[[i]][[h]], nb[[j]][[h]])))
  }
  ids_all <- names(nb[[1]])
  null_j <- replicate(2000, {
    a <- sample(ids_all, k); b <- sample(ids_all, k); jacc(a, b)
  })
  rows_f2[[length(rows_f2) + 1]] <- data.frame(
    k = k, mean_jaccard = mean(obs), median_jaccard = median(obs),
    frac_zero_overlap = mean(obs == 0),
    null_mean_jaccard = mean(null_j),
    ratio_obs_over_null = mean(obs) / mean(null_j)
  )
}
df_f2 <- do.call(rbind, rows_f2)
print(df_f2, row.names = FALSE, digits = 3)
cat("\n     ratio 接近 1 = 近邻身份跨 seed 基本随机 -> kNN 分配起点不可用\n")
cat("     ratio 远大于 1 = 近邻身份稳定 -> kNN 分配起点有依据\n")

}

# ======================================================================
save_if <- function(obj_name, fname) {
  if (exists(obj_name)) write.csv(get(obj_name), shape_out(fname), row.names = FALSE)
}
save_if("df_a",  "dip_1d_projections.csv")
save_if("df_f",  "dip_realworld_features.csv")
save_if("df_b",  "dip_pairwise_mc.csv")
save_if("df_c",  "dip_rowwise.csv")
save_if("df_d1", "power_multivariate.csv")
save_if("df_d2", "power_1d.csv")
save_if("df_e",  "fiedler_coordinate.csv")
save_if("df_e2", "fiedler_cross_seed.csv")
save_if("df_f2", "knn_stability.csv")
cat(sprintf("\n>>> 已按实验编号保存到 %s/expN_*\n", SHAPE))

cat("\n>>> 判读说明\n")
cat("    p 显著(小) = 拒绝单峰 = 支持存在离散分组\n")
cat("    p 不显著   = 无法拒绝单峰 = 与连续分布一致\n")
cat("    看 (D)：若在真实分离度下检测率就很低，\"不显著\"说明检验没有分辨力，而不是证据支持连续。")
