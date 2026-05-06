# anime-hbayes-rank

基于层次贝叶斯模型的动画评分排名。纠正评分者习惯差异后的更公平排名。

## 方法

```
s_ui ~ Normal(q_i + b_u, σ_u)    观测层：评分 = 条目质量 + 用户偏差 + 噪声
q_i  ~ Normal(μ_q, τ_q · w_i)    条目层：精度加权先验，自动收缩
b_u  ~ Normal(0, τ_b)            用户层：偏差向零收缩
σ_u  ~ HalfNormal(1)             每用户噪声水平
```

- **精度加权先验**：评分人数越少，先验越紧（向全局均值收缩），n_ref 纯数据驱动
- **推断**：NumPyro SVI (变分推断) + JAX GPU 后端
- **数据**：Bangumi 15M 数据集，97,705 用户 × 7,556 条目 × 7,149,450 评分

## 运行

```bash
# 安装
pip install -r requirements.txt

# 准备数据 (需先下载 Bangumi 15M 数据集)
python load.py

# 跑模型
python run_hb_final.py

# 查看结果
cd web && python -m http.server 8080
```

## 目录

```
hbayes/         层次贝叶斯模型 (NumPyro + JAX)
fusion/         五层融合排名流水线 (Massey/Keener/OD/IMDb)
utils/          原始算法实现
web/            纯静态展示页面
```
