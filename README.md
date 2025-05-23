# MLAflow

<p align="center">
  <img src="Logo.jpg" alt="MLAutoFlow Logo" width="450"/>
</p>

## 🚀 项目简介

**MLAflow** 是一款基于人工智能架构开发的、无监督的单细胞 RNA 测序全流程自动处理系统。  
通过集成 **贝叶斯优化（Bayesian Optimization）**、**Optuna-TPE**、**CMA-ES** 等高效超参数优化算法，  
实现从数据导入到聚类结果输出的全流程自动化与智能化。

该系统面向科研人员和数据分析师，致力于简化繁杂的分析流程，提高分析效率与结果稳定性。

---

## 🔧 系统模块功能一览

### 1️⃣ 自动化数据预处理模块
- **细胞质控（Quality Control）**  
  自动调节 `min_genes`、`percent_mito`、`min_cells` 等过滤参数，识别低质量细胞；
- **归一化与对数转换**  
  自适应设置 `target_sum`，确保细胞间表达量一致性；
- **高变基因筛选（HVGs）**  
  在 1000–3000 范围内自动确定最佳高变基因数量，提高信噪比和降维效果；

### 2️⃣ 降维与特征压缩模块
- 使用 **PCA 主成分分析** 对表达矩阵进行降维；
- 自动优化 `n_pcs` 参数，兼顾保留信息量与计算效率；
- 引入 **CMA-ES 自适应算法**，评估 PCA 在不同细胞类型（如神经元、免疫细胞）中的鲁棒性；

### 3️⃣ 聚类分析与图结构构建模块
- 构建 KNN 邻接图，采用 **Leiden 算法**进行聚类；
- 自动优化 `resolution` 参数，提升聚类精度；
- 通过 **silhouette score** 和 **Calinski-Harabasz 指数** 等指标，指导参数调节并增强聚类一致性；

---

## 📈 全流程优化引擎

- 利用 **Optuna + TPE 策略** 搜索参数组合，自动探索全局最优；
- **CMA-ES** 专用于 PCA 模块，确保主成分数 `n_pcs` 的自适应调整；
- 通过 **聚类效果反向反馈机制**（如 silhouette score），实现各阶段参数联动调节，打破传统分析“阶段割裂”的局限性；

---

## 🧠 技术亮点

- 🧩 **模块解耦 + 并行调参**：不同阶段独立运行，自动跳过无需重复的步骤；
- 🧠 **全流程无监督设计**：自动学习最优配置，无需人工干预；
- 🧠 **数据格式广泛支持**：兼容 `.h5`, `.h5ad`, `.mtx`, `.csv`, `.txt`, `.xlsx` 等常见格式；
- 💾 **内存友好型实现**：支持 AnnData 的 `backed` 模式，适应大规模数据集；
- 🛠️ **极简安装，开箱即用**

---

## 📦 安装方式

请确保使用 Python 3.8+ 环境，并安装以下依赖：

```bash
pip install scanpy optuna cma umap-learn
```

```bash
pip install mlaflow
```
## 🔍 使用方法
使用 MLAflow
在 Python 环境中导入 MLAflow 并使用 autoprocess 方法处理单细胞数据：
```python
import mlaflow
# 直接处理数据，指定数据文件路径
adata = mlaflow.autoprocess('path_to_your_data_file')
```
其中，path_to_your_data_file 为您数据文件的路径，可以是 .h5ad, .csv 等支持的格式。

---

## 🔮 未来展望

- ✅ 拓展对空间转录组（Spatial Transcriptomics）的支持；
- ✅ 引入 Transformer 模型进行表达数据的深层次表示学习；
- ✅ 增强可视化界面交互能力，打造无代码图形化分析平台；

---

## 🤝 合作者

| 贡献者 | GitHub 主页 |
|--------|-------------|
| guxiao2005 | [@guxiao2005](https://github.com/guxiao2005) |
| like-lkr   | [@like-lkr](https://github.com/like-lkr) |

---
"""
