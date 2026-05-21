# Reading Guide: Breaking Neural Network Scaling Laws with Modularity
## Boopathy et al., ICLR 2025

---

## 🚨 EMERGENCY 1-DAY PRESENTATION PREP: Logic, Math, Implementation 🚨

*Since you are presenting tomorrow and must cover logic, math, and implementation, use this structured guide. It maps exactly to your presentation requirements and references the provided Jupyter notebook.*

### 🧠 PART 1: The Logic (The "Why")
**The Core Problem:** As tasks get more complex (more dimensions $m$), standard neural networks need *exponentially* more data to learn ($n \propto \Omega^{2m}$). This is the "scaling law" wall.
**The Solution:** Use a **Modular Network**—a team of smaller "expert" networks. If each module handles one part of the input, the data needed stays constant ($O(1)$) instead of growing exponentially.
**The Catch:** If you just build a modular network and train it normally, the modules get confused and don't specialize. They need a "study guide" *before* training starts so they know what to look at.

*What to read:*
- The Introduction of the paper (https://arxiv.org/abs/2409.05780)
- The markdown cells in Section 1-3 of `modular_nn_compositional_cifar10.ipynb`

### 📐 PART 2: The Math (The "How" Theoretically)
The paper's breakthrough is **Kernel-Based Initialization**. Before training the neural network, we find the optimal projection matrix $\hat{U}_i$ for each module to focus its attention.

**The Objective Function:** We want to minimize $\sum \|\theta_i\|^2 = y(X)^T \mathbf{K}^{-1} y(X)$
**What this means in plain English:** 
- $\mathbf{K}$ is a Kernel matrix that measures how similar data points are *after* they are projected by $\hat{U}$.
- $y^T K^{-1} y$ measures how "hard" it is to predict the labels from this projected data.
- By minimizing this equation, we find the projection $\hat{U}$ that makes the task as **easy as possible** for the module, effectively forcing it to focus on a single relevant image component.

*What to read:*
- Section 4.2 of the paper (Kernel Objective)
- Section 4 of the notebook (markdown cell explaining the math)

### 💻 PART 3: The Implementation (The "How" Practically)
You can directly show the Jupyter Notebook code during your presentation to prove this works in practice.

**1. The Modular Architecture (Cell 8):**
Show `ModularNet.forward()`. Point out that the module outputs are **SUMMED** (`total = total + out`), matching the theoretical equation. Unlike standard networks, they share weights (`self.shared_net`) but have independent projections (`self.projections`).

**2. The Math in Code (Cell 10 - `kernel_init_one_module`):**
Show how the math objective $y^T K^{-1} y$ is translated to code:
```python
# Compute Kernel K
K = rbf_kernel(Xb_proj, Xb_proj, sigma=sigma)
# Solve K^{-1} y
alpha = torch.linalg.solve(K, yb)
# Loss = y^T K^{-1} y
loss = (yb * alpha).sum()
loss.backward() # Optimize the projection matrix U!
```

**3. The Results (Section 6 & 9 of the Notebook):**
- **Section 6 graph:** Show how Monolithic accuracy crashes as $k$ (number of images) increases, while the Kernel Init Modular method stays high.
- **Section 9 visuals:** Show the module projections. The visuals prove the math worked: kernel-initialized modules cleanly focus on just one image slot!

---

## Comprehensive Reading Path (For Deep Understanding)

Work through these in order. Each level builds on the previous.

---

## Level 1 — Foundations (read first, ~2–3 days)

### 1. Neural Scaling Laws (the problem this paper solves)

**Kaplan et al. (2020) — "Scaling Laws for Neural Language Models"**
- arXiv: https://arxiv.org/abs/2001.08361
- What to read: Sections 1–4 (skip the rest if time-constrained)
- Key concept: loss scales as power-law with data n and parameters p
- Why: the paper directly extends and challenges these results

**Sharma & Kaplan (2022) — "Scaling Laws from the Data Manifold Dimension"**
- JMLR: https://jmlr.org/papers/v23/20-1111.html
- Key concept: sample complexity ∝ Ω^(2m) where m is intrinsic dimensionality
- Why: this is exactly the exponential scaling the paper proves modular NNs break

---

### 2. Generalization Theory

**Bahri et al. (2021) — "Explaining Neural Scaling Laws"**
- arXiv: https://arxiv.org/abs/2102.06701
- What to read: Sections 1–3
- Key concept: kernel regression framework for explaining generalization
- Why: the paper's theoretical model is built on this foundation

**Hastie et al. (2022) — "Surprises in High-Dimensional Ridgeless Least Squares"**
- Paper: https://arxiv.org/abs/1903.08560
- Key concept: exact closed-form expressions for train/test error
- Why: the paper's Theorem 1 directly follows this technique (cited as the specific approach)
- Note: This is math-heavy; focus on Section 2 (setup) and Theorem 1

---

### 3. Double Descent

**Belkin et al. (2019) — "Reconciling Modern ML Practice and the Bias-Variance Trade-off"**
- PNAS: https://www.pnas.org/doi/10.1073/pnas.1903070116
- Short and accessible; explains why overparameterized NNs generalize
- Key concept: interpolation threshold at p=n; test loss drops after it
- Why: Figure 1 of the paper directly shows double descent; understanding it is required

---

## Level 2 — Modular Networks (~2–3 days)

### 4. What are Modular NNs?

**Pfeiffer et al. (2023) — "Modular Deep Learning"** (survey paper)
- arXiv: https://arxiv.org/abs/2302.11529
- What to read: Sections 1–3 (taxonomy and motivation)
- Why: best overview of the field; situates the paper's contributions

**Andreas et al. (2016) — "Neural Module Networks"**
- CVPR: https://arxiv.org/abs/1511.02799
- Short, historically important; introduces composing modules for VQA
- Why: the paper cites this as canonical modular architecture motivation

---

### 5. Why Modularity Fails in Practice

**Csordas et al. (2021) — "Are Neural Nets Modular? Inspecting Functional Modularity"**
- ICLR: https://arxiv.org/abs/2003.04881
- Key finding: even networks with modular architecture don't use modules modularly
- Why: this is the exact problem the paper's learning rule is designed to solve

**Mittal et al. (2022) — "Is a Modular Architecture Enough?"**
- NeurIPS: https://arxiv.org/abs/2206.02713
- Extends Csordas finding; shows gradient descent fails to exploit modularity
- Why: directly motivates the kernel-based initialization in the paper

---

### 6. Mixture of Experts (architecture inspiration)

**Shazeer et al. (2017) — "Outrageously Large Neural Networks: Sparsely-Gated MoE"**
- arXiv: https://arxiv.org/abs/1701.06538
- Key concept: multiple expert sub-networks, router selects which to use
- Why: paper's modular architecture is a non-sparse version of MoE

---

## Level 3 — Math Behind the Paper (~3–4 days)

### 7. Kernel Regression & Manifolds

**McRae et al. (2020) — "Sample Complexity and Effective Dimension for Regression on Manifolds"**
- NeurIPS: https://arxiv.org/abs/2010.10726
- Key result: sample complexity scales with intrinsic manifold dimension
- Why: provides the Ω^(2m) exponential scaling result the paper builds on

**Canatar et al. (2021) — "Spectral Bias and Task-Model Alignment"**
- Nature Communications: https://www.nature.com/articles/s41467-021-23103-1
- Key concept: spectral decomposition of kernel regression generalization
- Why: paper's linear model assumption connects to this via Neural Tangent Kernel

---

### 8. Neural Tangent Kernel

**Jacot et al. (2018) — "Neural Tangent Kernel: Convergence and Generalization"**
- NeurIPS: https://arxiv.org/abs/1806.07572
- Key concept: infinite-width NNs behave as kernel regression
- Why: justifies why the linear-in-parameters assumption is non-trivial and still predictive

---

### 9. Combinatorial / Compositional Generalization

**Jarvis et al. (2023) — "On the Specialization of Neural Modules"**
- ICLR: https://arxiv.org/abs/2209.10546
- Introduces Compositional MNIST (inspiration for Compositional CIFAR-10 in the paper)
- Why: directly related to the empirical setup; explains what "systematic generalization" means

---

## Quick Reference: Core Equations

| Equation | Meaning | Where in paper |
|---|---|---|
| `λᵢ = c[i^(−Ω^(−m)) − (i+1)^(−Ω^(−m))]` | Eigenvalue decay; effective dims ∝ Ω^(2m) | Sec 3.1, App A |
| `𝔼[‖y−ŷ‖²] = d·Tr(Λ₂)·F(dn,p) − ...` | Exact test loss (Theorem 1) | Sec 3.2 |
| `F(n,p) = 𝔼[‖R†‖²_F]` | Pseudo-inverse Frobenius norm | App D |
| `ŷ(x) = (1/√K) Σⱼ ŷⱼ(Ûⱼᵀx)` | Modular architecture | Sec 4 |
| `Σ‖θᵢ‖² = y(X)ᵀ K⁻¹ y(X)` | Kernel init objective | Sec 4.2, Eq 17 |

---

## Key Concepts Checklist

Before reading the paper, you should understand:
- [ ] What is sample complexity? (number of samples to achieve ε error)
- [ ] What is intrinsic dimensionality of a data manifold?
- [ ] What is a kernel / kernel regression?
- [ ] What is the interpolation threshold / double descent?
- [ ] What is a covariance matrix and its trace?
- [ ] What does "module" mean in the modular NN sense?

---

## Code Resources

| Resource | Link |
|---|---|
| Official paper code | https://github.com/AkhilanB/breaking-scaling-laws |
| Paper PDF | https://arxiv.org/pdf/2409.05780 |
| Project page | https://akhilanb.github.io/breaking-scaling-laws/ |

---

## Common Pitfalls When Implementing

1. **Architecture**: modules must SUM outputs, not concatenate
2. **Sigma**: always use the median heuristic, not a fixed value
3. **Scale**: the paper uses 1M training samples; results on <10k will be noisier
4. **Modules**: use 5×k modules, not k (paper Appendix E)
5. **Initialization**: kernel init must run before gradient descent, not jointly
