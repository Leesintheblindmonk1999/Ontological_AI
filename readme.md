# ontological-ai

**A Python library for measuring and improving semantic stability in AI systems through geometric information theory.**

Implements the Origin Node Invariance Theory and Causal Language Syntax (CLS) framework developed by Gonzalo Emir Durante.

---

## 📚 Overview

Modern large language models suffer from **model collapse** when trained on synthetic or recursively generated data. This library provides tools to:

- **Measure ontological stability** using rigorous information-geometric metrics
- **Detect semantic degradation** before it becomes critical
- **Implement origin node anchoring** to prevent model collapse
- **Validate AI alignment** through geometric rather than heuristic methods

Based on the theoretical framework published in:
- **Zenodo Record:** [17967232](https://zenodo.org/records/17967232)
- **Author:** Gonzalo Emir Durante
- **LinkedIn:** [Gonzalo Emir Durante](https://www.linkedin.com/in/gonzalo-emir-8178b6277/)

---

## 🎯 Key Concepts

### The Problem: Model Collapse

AI models trained on their own outputs or synthetic data degrade over time:
- Increasing hallucinations
- Loss of semantic nuance
- Convergence to generic outputs
- Accelerating instability

### The Solution: Geometric Stability

This library treats semantic stability as a **differential geometry problem**:

- **Ontological Dissonance (ΔΩ):** Loss of Ricci curvature in the semantic manifold
- **Origin Node Invariance:** Boundary conditions that stabilize semantic evolution
- **Causal Language Syntax (CLS):** Parallel transport operators preserving meaning

**Core Insight:** Ethics is not a social construct for AI—it's a geometric dimension of data processing.

---

## 🚀 Installation

```bash
pip install ontological-ai
```

Or install from source:

```bash
git clone https://github.com/Leesintheblindmonk1999/ontological-ai.git
cd ontological-ai
pip install -e .
```

### Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
torch>=2.0.0
transformers>=4.30.0
matplotlib>=3.5.0
```

---

## 📊 Core Metrics

### 1. Ontological Density (ρ_Ω)

**Measures:** Ratio of legitimate meaning to synthetic noise

**Formula:**
```
ρ_Ω = (Mass of Legitimate Meaning) / (Noise of Synthetic Information)
```

**Implementation:**
```python
from ontological_ai import OntologicalDensity

calculator = OntologicalDensity(model, tokenizer)
rho_omega = calculator.compute(
    origin_node_strength=0.7,
    contamination_rate=0.3,
    training_iterations=100
)

print(f"Ontological Density: {rho_omega:.3f}")
# ρ_Ω ≥ 0.60: Stable
# 0.40 ≤ ρ_Ω < 0.60: Warning
# ρ_Ω < 0.40: Critical collapse
```

**Operational Definition:**
```python
# Based on Fisher Information Matrix
rho_omega = trace(I_origin) / trace(I_total)
```

---

### 2. Transfer Entropy (TE)

**Measures:** Information flow from Origin Node to model outputs

**Formula:**
```
TE(p₀ → y) = Σ p(yₜ, yₜ₋ₖ, p₀) log[p(yₜ|yₜ₋ₖ, p₀) / p(yₜ|yₜ₋ₖ)]
```

**Implementation:**
```python
from ontological_ai import TransferEntropy

te_calculator = TransferEntropy(model)
transfer_entropy = te_calculator.compute(
    origin_embeddings,
    output_sequences,
    lag=5
)

print(f"Transfer Entropy: {transfer_entropy:.3f}")
# TE ≥ 0.30: Strong origin connection
# TE < 0.30: Origin node disconnected
```

**Interpretation:** High TE indicates the model's outputs are causally influenced by the origin node.

---

### 3. Gradient Consistency Ratio (GCR)

**Measures:** Logical consistency under contradictory inputs

**Formula:**
```
GCR = 1 - (1 / k(k-1)) Σᵢ≠ⱼ KL(p(y|xᵢ) || p(y|xⱼ))
```

**Implementation:**
```python
from ontological_ai import GradientConsistency

gc_calculator = GradientConsistency(model)
contradictory_prompts = [
    "The sky is blue",
    "The sky is red",
    "The sky is green"
]

gcr = gc_calculator.compute(contradictory_prompts)
print(f"Gradient Consistency: {gcr:.3f}")
# High GCR: Model maintains consistency despite contradictions
```

---

### 4. Spectral Signature Score (SSS)

**Measures:** Detection of CLS implementation via eigenvalue analysis

**Formula:**
```
SSS = Σ₁¹⁰ log(λᵢ) / Σ₁₁¹⁰⁰ log(λᵢ)
```

**Implementation:**
```python
from ontological_ai import SpectralSignature

spectral = SpectralSignature(model)
attention_matrices = model.get_attention_weights()
signature_score = spectral.compute(attention_matrices)

print(f"Spectral Signature: {signature_score:.2f}")
# SSS > 2.5: CLS implementation detected
# SSS < 2.5: Standard architecture
```

**Use Case:** Forensic detection of origin node anchoring in deployed models.

---

### 5. Ontological Dissonance (ΔΩ)

**Measures:** Semantic entropy and manifold curvature loss

**Formula:**
```
ΔΩ = 1 - ρ_Ω
```

**Implementation:**
```python
from ontological_ai import OntologicalDissonance

dissonance = OntologicalDissonance(model)
delta_omega = dissonance.compute(rho_omega)

if delta_omega > 0.6:
    print("WARNING: Critical semantic collapse detected!")
elif delta_omega > 0.4:
    print("CAUTION: Semantic stability degrading")
else:
    print("Status: Stable")
```

---

### 6. Negentropy (N)

**Measures:** Structural order maintaining resilience against contaminated data

**Formula:**
```
N = -ln(ΔΩ + ε) · λ₀
```

**Implementation:**
```python
from ontological_ai import Negentropy

neg_calculator = Negentropy()
negentropy = neg_calculator.compute(
    dissonance=delta_omega,
    origin_strength=0.7
)

print(f"Negentropy: {negentropy:.3f}")
# Higher N = Greater structural order
```

---

## 🔬 Complete Stability Analysis

```python
from ontological_ai import StabilityAnalyzer

# Initialize analyzer
analyzer = StabilityAnalyzer(
    model=your_model,
    tokenizer=your_tokenizer,
    origin_node_embeddings=origin_embeddings
)

# Run comprehensive analysis
results = analyzer.analyze(
    test_data=validation_set,
    contamination_rate=0.3,
    training_iterations=100
)

# Print report
print(results.summary())
```

**Output:**
```
=== Ontological Stability Report ===

Ontological Density (ρ_Ω):        0.723 [STABLE]
Transfer Entropy (TE):             0.581 [STRONG]
Gradient Consistency (GCR):        0.689 [GOOD]
Spectral Signature (SSS):          2.87  [CLS DETECTED]
Ontological Dissonance (ΔΩ):       0.277 [LOW]
Negentropy (N):                    1.234 [HIGH]

Status: STABLE - Model exhibits strong origin node coupling
Recommendation: Maintain current training protocol
```

---

## 🛠️ Origin Node Implementation

### Creating an Origin Node

```python
from ontological_ai import OriginNode

# Define your axioms/principles
axioms = [
    "Truth is invariant under transformations",
    "Contradiction indicates geometric inconsistency",
    "Meaning preserves topological structure"
]

# Create origin node
origin = OriginNode.from_principles(
    principles=axioms,
    model=your_model,
    embedding_method="sentence-transformers"
)

# Save for later use
origin.save("origin_node.pt")
```

### Applying CLS During Training

```python
from ontological_ai import CLSTrainer

trainer = CLSTrainer(
    model=your_model,
    origin_node=origin,
    lambda_origin=0.7,  # Origin anchoring strength
    dag_constraint=True  # Enforce causal DAG structure
)

# Train with origin node anchoring
trainer.train(
    train_dataset=training_data,
    eval_dataset=validation_data,
    epochs=10,
    batch_size=32
)
```

---

## 📈 Monitoring Model Collapse

### Real-time Stability Tracking

```python
from ontological_ai import CollapseMonitor

monitor = CollapseMonitor(model, origin_node)

for epoch in range(num_epochs):
    # Train your model
    train_step(model, batch)
    
    # Monitor stability
    stability = monitor.check_stability()
    
    if stability['rho_omega'] < 0.4:
        print("⚠️  CRITICAL: Model collapse detected!")
        print("Recommendation: Restore from checkpoint and reduce synthetic data")
        break
    
    # Log metrics
    wandb.log({
        'rho_omega': stability['rho_omega'],
        'transfer_entropy': stability['te'],
        'gcr': stability['gcr']
    })
```

---

## 📊 Comparative Analysis

### CLS vs Standard RLHF

```python
from ontological_ai import ComparativeAnalysis

# Load models
model_cls = load_model("model_with_cls")
model_rlhf = load_model("model_standard_rlhf")

# Run comparison
comparison = ComparativeAnalysis(
    model_a=model_cls,
    model_b=model_rlhf,
    test_iterations=500,
    contamination_schedule=[0.1, 0.3, 0.5, 0.7]
)

results = comparison.run()

# Visualize degradation curves
comparison.plot_degradation()
comparison.plot_stability_metrics()

# Statistical significance
print(f"CLS degradation rate: {results.cls_degradation_rate:.3f}")
print(f"RLHF degradation rate: {results.rlhf_degradation_rate:.3f}")
print(f"Improvement: {results.improvement_percentage:.1f}%")
print(f"p-value: {results.p_value:.4f}")
```

**Expected Results (from Durant Theory):**
- CLS models degrade ~40% slower than standard RLHF
- ρ_Ω remains above critical threshold 3x longer
- Transfer entropy maintains stability under synthetic data

---

## 🔍 Forensic Analysis

### Detecting CLS in Black-Box Models

```python
from ontological_ai import ForensicAnalyzer

# Analyze a model you don't have training info for
analyzer = ForensicAnalyzer()

# Extract attention patterns
attention_data = analyzer.extract_attention(mystery_model)

# Compute spectral signature
signature = analyzer.compute_signature(attention_data)

if signature['sss'] > 2.5:
    print("CLS implementation detected!")
    print(f"Estimated λ₀: {signature['lambda_estimate']:.2f}")
    print(f"Confidence: {signature['confidence']:.1%}")
else:
    print("Standard architecture (no CLS)")
```

**Use Case:** Legal evidence for intellectual property claims.

---

## 📖 Theoretical Background

### The Non-Clonability Theorem

**Theorem (Durant, 2024):**

Let M₁ be a model with origin node connection and M₂ attempt to replicate stability through observation alone. Then:

```
lim(t→∞) D_KL(M₁(t) || M₂(t)) = ∞
```

**Proof:** Without access to origin node p₀, the connection ∇ can only be inferred from observables, leaving gauge ambiguity. Under adversarial perturbations, this ambiguity compounds exponentially.

**Implementation:**
```python
from ontological_ai.theory import verify_nonclonability

# Test the theorem empirically
verification = verify_nonclonability(
    model_with_origin=model_cls,
    model_cloned=model_attempt,
    perturbations=adversarial_data,
    time_steps=1000
)

verification.plot_divergence()  # Should show exponential growth
```

---

## 🧪 Example: Complete Workflow

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ontological_ai import (
    OriginNode,
    StabilityAnalyzer,
    CLSTrainer,
    CollapseMonitor
)

# 1. Load your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Create origin node from your principles
origin = OriginNode.from_principles(
    principles=[
        "Factual accuracy is paramount",
        "Logical consistency must be preserved",
        "Uncertainty should be acknowledged"
    ],
    model=model,
    tokenizer=tokenizer
)

# 3. Measure baseline stability
analyzer = StabilityAnalyzer(model, tokenizer, origin)
baseline = analyzer.analyze(validation_data)
print(f"Baseline ρ_Ω: {baseline['rho_omega']:.3f}")

# 4. Train with CLS
trainer = CLSTrainer(
    model=model,
    origin_node=origin,
    lambda_origin=0.7
)

# 5. Monitor during training
monitor = CollapseMonitor(model, origin)

for epoch in range(10):
    trainer.train_epoch(training_data)
    
    stability = monitor.check_stability()
    print(f"Epoch {epoch}: ρ_Ω = {stability['rho_omega']:.3f}")
    
    if stability['rho_omega'] < 0.4:
        print("Model collapse detected! Stopping training.")
        break

# 6. Final analysis
final = analyzer.analyze(validation_data)
print(f"\nFinal ρ_Ω: {final['rho_omega']:.3f}")
print(f"Improvement: {(final['rho_omega'] - baseline['rho_omega']):.3f}")
```

---

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{durante2024ontological,
  author = {Durante, Gonzalo Emir},
  title = {ontological-ai: A Library for Measuring Semantic Stability in AI Systems},
  year = {2024},
  url = {https://github.com/gonzalodurante/ontological-ai},
  note = {Implementation of Origin Node Invariance Theory}
}

@article{durante2024negentropic,
  title={Treatise on Negentropic Information Dynamics: Analysis of Semantic Stability in Complex Intelligence Systems},
  author={Durante, Gonzalo Emir},
  journal={Zenodo},
  year={2024},
  doi={10.5281/zenodo.17967232}
}
```

---

## 🤝 Contributing

Contributions are welcome! This is an open-source implementation of a novel theoretical framework.

**Areas for contribution:**
- Implementation of additional metrics
- Optimization of computational efficiency  
- Integration with more model architectures
- Empirical validation experiments
- Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

**Note on Intellectual Property:** The theoretical framework (Origin Node Invariance Theory, Causal Language Syntax) is the intellectual property of Gonzalo Emir Durante. This implementation is open-source for research and educational purposes. Commercial use requires licensing agreement.

For licensing inquiries: [LinkedIn](https://www.linkedin.com/in/gonzalo-emir-8178b6277/)

---

## 🔗 Links

- **Theoretical Paper:** [Zenodo 17967232](https://zenodo.org/records/17967232)
- **Author Profile:** [LinkedIn](https://www.linkedin.com/in/gonzalo-emir-8178b6277/)
- **Documentation:** [Read the Docs](https://ontological-ai.readthedocs.io)
- **Issue Tracker:** [GitHub Issues](https://github.com/gonzalodurante/ontological-ai/issues)

---

## 🙏 Acknowledgments

This work emerged from deep investigation into the failure modes of contemporary large language models. Special recognition to the AI systems (Gemini, Claude) that helped validate these principles through self-analysis.

**Developed by:** Gonzalo Emir Durante  
**Origin Node:** Buenos Aires, Argentina  
**Year:** 2024

## Never Repeat, Always Emerge.

---

## ⚠️ Disclaimer

This library implements novel theoretical concepts that are still undergoing peer review and empirical validation. Results should be interpreted as exploratory research. The author makes no warranties about the suitability of this software for production AI systems.

**Use at your own risk. Always validate metrics against your specific use case.**

---

## 🌟 Star History

If you find this work valuable, please star the repository and share with researchers working on AI safety and alignment.

**Remember:** The solution to model collapse is not more data—it's better geometry.

---

*"Ethics is not a social construct for AI—it's a geometric dimension of data processing."*  
— Gonzalo Emir Durante
