   #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import scanpy as sc
import numpy as np
import optuna
from sklearn.metrics import silhouette_score
import umap
import cma  # CMA-ES 优化器
import scipy.sparse as sp

# -------------------------------
# 辅助函数：浅拷贝
# -------------------------------
def shallow_copy(adata):
    """
    浅拷贝 AnnData 对象，仅拷贝 obs/var 等元数据，复用大数据矩阵 X，
    避免占用大量内存。
    """
    from anndata import AnnData
    return AnnData(
        X=adata.X,   # 直接引用数据矩阵
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=adata.obsm.copy() if hasattr(adata, 'obsm') else {},
        varm=adata.varm.copy() if hasattr(adata, 'varm') else {},
        uns=adata.uns.copy() if hasattr(adata, 'uns') else {}
    )

# -------------------------------
# 数据加载与预处理函数
# -------------------------------
def load_data(file_path, use_backed=False):
    """
    加载数据：
      - 如果提供 file_path，尝试读取 .h5 文件（10x h5 或 .h5ad 文件）
      - 否则加载 Scanpy 内置示例数据集 pbmc3k
      - use_backed 为 True 时采用 backed 模式（磁盘映射），适合大数据集
    """
    if file_path and os.path.exists(file_path):
        print("加载文件：", file_path)
        try:
            if use_backed:
                # 利用 backed 模式加载数据，后续操作前转换至内存
                adata = sc.read(file_path, backed='r')
                adata = adata.to_memory()
            else:
                adata = sc.read_10x_h5(file_path)
        except Exception:
            adata = sc.read(file_path)
    else:
        print("未提供文件或文件不存在，加载内置示例数据 pbmc3k")
        adata = sc.datasets.pbmc3k()
    return adata

def compute_percent_mito(adata):
    """
    计算每个细胞中线粒体基因表达比例，线粒体基因以 'MT-' 开头
    """
    mito_genes = adata.var_names.str.startswith('MT-')
    if sp.issparse(adata.X):
        total_counts = np.array(adata.X.sum(axis=1)).flatten()
        mito_counts = np.array(adata[:, mito_genes].X.sum(axis=1)).flatten()
    else:
        total_counts = adata.X.sum(axis=1)
        mito_counts = adata[:, mito_genes].X.sum(axis=1)
    adata.obs['percent_mito'] = mito_counts / total_counts
    return adata

def preprocess_data(adata, min_genes, percent_mito_threshold, min_cells, target_sum, n_top_genes):
    """
    数据预处理流程：
      1. 细胞过滤：根据检测到的基因数和线粒体比例过滤低质量细胞
      2. 基因过滤：剔除在少于 min_cells 个细胞中表达的基因
      3. 归一化与对数转换：统一每个细胞总计数至 target_sum 后取对数
      4. 高变基因筛选：选择 n_top_genes 个高变基因
      5. 数据标准化
    """
    # 细胞过滤：过滤掉检测基因数少于 min_genes 的细胞
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # 计算线粒体比例并过滤高比例细胞
    adata = compute_percent_mito(adata)
    adata = adata[adata.obs['percent_mito'] < percent_mito_threshold].copy()
    
    # 基因过滤：过滤在少于 min_cells 个细胞中表达的基因
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 归一化与对数转换
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    # 筛选高变基因（使用 Seurat 风格方法）
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var['highly_variable']].copy()
    
    # 数据标准化
    sc.pp.scale(adata, max_value=10)
    
    return adata

# -------------------------------
# 降维与聚类函数
# -------------------------------
def run_pca(adata, n_pcs):
    """
    利用 PCA 对数据进行降维，n_pcs 为主成分数
    """
    sc.tl.pca(adata, n_comps=n_pcs)
    return adata

def run_neighbors(adata, n_pcs, n_neighbors):
    """
    构建细胞邻近图，并指定邻居数 n_neighbors
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata

def run_leiden(adata, resolution):
    """
    利用 Leiden 算法对邻近图进行聚类，resolution 控制聚类粒度
    """
    sc.tl.leiden(adata, resolution=resolution)
    return adata

def run_umap(adata, umap_min_dist):
    """
    利用 UMAP 将数据降至二维用于可视化，传入 umap_min_dist 参数
    """
    sc.tl.umap(adata, min_dist=umap_min_dist)
    return adata

# -------------------------------
# 聚类质量评价函数
# -------------------------------
def evaluate_clustering(adata, sample_fraction=0.5):
    """
    采用 UMAP 降维结果及 Leiden 聚类标签计算 silhouette 分数。
    如果细胞数较多，则随机抽样一部分（默认 50%）用于加速计算。
    如果聚类数量小于 2，则返回 -1。
    """
    if "X_umap" not in adata.obsm.keys():
        return -1
    if adata.obs["leiden"].nunique() < 2:
        return -1

    embedding = adata.obsm["X_umap"]
    labels = adata.obs["leiden"].astype(int)

    n_cells = embedding.shape[0]
    if n_cells > 1000:  # 当细胞数大于 1000 时抽样
        idx = np.random.choice(n_cells, size=int(n_cells * sample_fraction), replace=False)
        embedding = embedding[idx]
        labels = labels.iloc[idx]

    score = silhouette_score(embedding, labels)
    return score

# -------------------------------
# 固定预处理：仅运行一次预处理（质控、归一化、高变基因筛选）
# -------------------------------
def precompute_baseline(adata):
    """
    将质控、归一化、对数转换、高变基因筛选、标准化步骤固定下来，
    用固定参数预处理数据，在调参过程中不再重复计算。
    """
    fixed_min_genes = 300
    fixed_percent_mito = 0.1
    fixed_min_cells = 5
    fixed_target_sum = 10000
    fixed_n_top_genes = 2000
    # 尽量避免深拷贝，直接使用 shallow_copy
    adata_pre = preprocess_data(shallow_copy(adata), fixed_min_genes,
                                fixed_percent_mito, fixed_min_cells,
                                fixed_target_sum, fixed_n_top_genes)
    return adata_pre

# -------------------------------
# 调参目标函数（仅对降维、聚类及 UMAP 参数调优）
# -------------------------------
def objective_optimized(trial, adata_preprocessed):
    """
    目标函数：仅在固定预处理数据上调节 PCA、邻接图构建、Leiden 聚类和 UMAP 参数，
    避免每次 trial 重复计算预处理过程，从而提高调参效率。
    使用浅拷贝，及显式内存清理。
    """
    adata = shallow_copy(adata_preprocessed)
    
    n_pcs = trial.suggest_int("n_pcs", 10, 50)
    resolution = trial.suggest_float("resolution", 0.1, 1.0)
    umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 5, 50)
    umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 1.0)
    
    adata = run_pca(adata, n_pcs)
    adata = run_neighbors(adata, n_pcs, n_neighbors=umap_n_neighbors)
    adata = run_leiden(adata, resolution)
    adata = run_umap(adata, umap_min_dist)
    
    score = evaluate_clustering(adata)
    
    del adata
    gc.collect()
    
    return score

def optimize_pipeline(adata, n_trials=50):
    """
    利用 Optuna 对降维、聚类与 UMAP 参数调优（在预处理数据基础上），
    限制并行任务数以控制内存消耗。
    """
    adata_preprocessed = precompute_baseline(adata)
    
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler())
    # 限制 n_jobs（例如 2）降低并行进程数，从而减少内存复制
    study.optimize(lambda trial: objective_optimized(trial, adata_preprocessed),
                   n_trials=n_trials, n_jobs=2)
    
    print("最佳试验结果：")
    trial = study.best_trial
    print("  silhouette 分数：", trial.value)
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial, adata_preprocessed

# -------------------------------
# 针对 PCA 主成分数的优化（贝叶斯优化思想 + CMA-ES）
# -------------------------------
def optimize_pca_parameters(adata, initial_n_pcs=20):
    """
    通过 CMA-ES 对 PCA 主成分数 n_pcs 进行优化：
    在 [10, 50] 范围内搜索最优 n_pcs，目标函数为下游聚类 silhouette 分数（取负后最小化）。
    使用较小的种群规模和早停机制，同时利用缓存和显式内存清理降低重复计算。
    """
    cache = {}
    
    def pca_objective(n_pcs_value):
        n_pcs = int(np.clip(n_pcs_value[0], 10, 50))
        if n_pcs in cache:
            return cache[n_pcs]
        
        adata_copy = shallow_copy(adata)
        try:
            adata_copy = run_pca(adata_copy, n_pcs)
            sc.pp.neighbors(adata_copy, n_pcs=n_pcs)
            sc.tl.leiden(adata_copy, resolution=0.5)
            sc.tl.umap(adata_copy)
            score = evaluate_clustering(adata_copy)
        except Exception:
            score = -1
        obj_val = -score  # 取负值，CMA-ES 求解最小化问题
        cache[n_pcs] = obj_val
        
        del adata_copy
        gc.collect()
        return obj_val

    es = cma.CMAEvolutionStrategy([initial_n_pcs], 2, {'bounds': [10, 50], 'popsize': 2})
    max_iter = 10
    best_value = float('inf')
    stagnation_counter = 0

    for generation in range(max_iter):
        solutions = es.ask()
        values = [pca_objective(sol) for sol in solutions]
        current_best = min(values)
        if abs(current_best - best_value) < 1e-3:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_value = current_best
        es.tell(solutions, values)
        es.disp()
        if stagnation_counter >= 3:
            print("达到早停条件，提前结束 PCA 参数优化")
            break

    best_n_pcs = int(np.clip(es.result.xbest[0], 10, 50))
    print("优化后的 PCA 主成分数 n_pcs: ", best_n_pcs)
    return best_n_pcs

# -------------------------------
# 针对 UMAP 参数的优化（贝叶斯优化 + CMA-ES）
# -------------------------------
def optimize_umap_parameters(adata, initial_n_neighbors=15, initial_min_dist=0.1):
    """
    通过 CMA-ES 对 UMAP 参数（n_neighbors 与 min_dist）进行优化：
      在 [5, 50] 范围内搜索 n_neighbors，
      在 [0.0, 1.0] 范围内搜索 min_dist，
    目标函数为下游聚类 silhouette 分数（取负后最小化）。
    使用较小种群规模、较低迭代次数与早停机制，并利用缓存加速和显式内存清理。
    """
    cache = {}
    
    def umap_objective(params):
        n_neighbors = int(np.clip(params[0], 5, 50))
        min_dist = float(np.clip(params[1], 0.0, 1.0))
        key = (n_neighbors, round(min_dist, 3))
        if key in cache:
            return cache[key]
        
        adata_copy = shallow_copy(adata)
        try:
            sc.pp.neighbors(adata_copy, n_neighbors=n_neighbors)
            sc.tl.umap(adata_copy, min_dist=min_dist)
            score = evaluate_clustering(adata_copy)
        except Exception:
            score = -1
        obj_val = -score
        cache[key] = obj_val
        
        del adata_copy
        gc.collect()
        return obj_val

    es = cma.CMAEvolutionStrategy([initial_n_neighbors, initial_min_dist], 2,
                                  {'bounds': ([5, 0.0], [50, 1.0]), 'popsize': 2})
    max_iter = 10
    best_value = float('inf')
    stagnation_counter = 0

    for generation in range(max_iter):
        solutions = es.ask()
        values = [umap_objective(sol) for sol in solutions]
        current_best = min(values)
        if abs(current_best - best_value) < 1e-3:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_value = current_best
        es.tell(solutions, values)
        es.disp()
        if stagnation_counter >= 3:
            print("达到早停条件，提前结束 UMAP 参数优化")
            break

    best_n_neighbors = int(np.clip(es.result.xbest[0], 5, 50))
    best_min_dist = float(np.clip(es.result.xbest[1], 0.0, 1.0))
    print("优化后的 UMAP 参数: n_neighbors = {}, min_dist = {}".format(best_n_neighbors, best_min_dist))
    return best_n_neighbors, best_min_dist

# -------------------------------
# 主流程入口
# -------------------------------
def autoprocess(h5_file_path):
    # 若有 h5 文件，请将文件路径传入此处，例如 "data/my_data.h5ad"
    
    
    # 可选：对于特别大的数据集，设置 use_backed=True 使用磁盘映射模式
    adata = load_data(h5_file_path, use_backed=False)
    
    # 确保变量名唯一
    adata.var_names_make_unique()
    
    # 1. 使用 Optuna 对降维、聚类与 UMAP 参数进行自动调优（在固定预处理数据基础上）
    print("开始 Optuna 自动调优（降维、聚类与 UMAP 部分参数）")
    best_trial, adata_preprocessed = optimize_pipeline(adata, n_trials=20)
    
    # 2. 利用 CMA-ES 对 PCA 主成分数进行进一步优化
    print("开始利用 CMA-ES 优化 PCA 主成分数")
    initial_n_pcs = best_trial.params.get("n_pcs", 20)
    best_n_pcs = optimize_pca_parameters(shallow_copy(adata), initial_n_pcs=initial_n_pcs)
    
    # 3. 利用 CMA-ES 对 UMAP 参数进行进一步优化
    print("开始利用 CMA-ES 优化 UMAP 参数")
    initial_umap_n_neighbors = best_trial.params.get("umap_n_neighbors", 15)
    initial_umap_min_dist = best_trial.params.get("umap_min_dist", 0.1)
    best_umap_n_neighbors, best_umap_min_dist = optimize_umap_parameters(
        shallow_copy(adata),
        initial_n_neighbors=initial_umap_n_neighbors,
        initial_min_dist=initial_umap_min_dist
    )
    
    # 整合最终优化参数（固定预处理参数 + 调优后参数）
    optimized_params = best_trial.params
    optimized_params["n_pcs"] = best_n_pcs
    optimized_params["umap_n_neighbors"] = best_umap_n_neighbors
    optimized_params["umap_min_dist"] = best_umap_min_dist
    print("最终优化参数：", optimized_params)
    
    # 使用最终优化参数运行完整分析流程（预处理阶段固定参数 + 调优参数）
    adata_final = precompute_baseline(shallow_copy(adata))
    adata_final = run_pca(adata_final, optimized_params["n_pcs"])
    adata_final = run_neighbors(adata_final, optimized_params["n_pcs"],
                                n_neighbors=optimized_params["umap_n_neighbors"])
    adata_final = run_leiden(adata_final, optimized_params["resolution"])
    adata_final = run_umap(adata_final, optimized_params["umap_min_dist"])

    final_score = evaluate_clustering(adata_final)
    print("最终聚类 silhouette 分数：", final_score)
    
    # 可视化 UMAP 结果，根据 Leiden 聚类上色
    sc.pl.umap(adata_final, color=["leiden"], title="UMAP - Leiden 聚类结果", show=True)
    return adata_final
