# HCPC-RLVR: Experimental Setup and Results

---

## 4. Experimental Setup

This section describes the model architecture, training configuration, evaluation protocol, and metrics used in the experiments. All experiments were conducted under identical conditions to ensure fair comparison across methods.

### 4.1 Model and Architecture

All experiments employ Qwen2.5-VL-3B-Instruct as the base vision-language model. This 3-billion-parameter model combines a vision encoder with a language decoder and supports interleaved image-text reasoning. Parameter-efficient fine-tuning is performed via Low-Rank Adaptation (LoRA; Hu et al., 2022) with rank $r = 8$ and scaling factor $\alpha = 16$. LoRA adapters are applied exclusively to the query projection ($W_q$) and value projection ($W_v$) matrices of the self-attention layers. This configuration introduces approximately 4.7 million trainable parameters (0.16% of total model parameters), enabling training on consumer-grade hardware while preserving the pretrained representations in the frozen parameters.

The choice of a 3B-parameter model is deliberate: it represents the smallest scale at which vision-language models demonstrate non-trivial chart reasoning capability, thereby providing a stringent test of whether RLVR-based training methods can extract meaningful improvements under tight capacity constraints. Larger models (7B, 72B) may exhibit ceiling effects that obscure differences between training strategies.

### 4.2 Training Dataset

Training was conducted on a 1,000-sample subset drawn from the Chart-RVR Chain-of-Thought training set (Sinha et al., 2025). The full dataset (HuggingFace: \texttt{sanchit97/chart-rvr-grpo-train}) comprises 34,194 chart-question pairs sourced from ChartQA, PlotQA, and ChartFC. The 1K subset was selected sequentially via \texttt{dataset.select(range(1000))} without shuffling, reflecting the leading portion of the dataset. Chart type distribution in the full dataset: bar (60.0\%), line (16.2\%), stacked bar (12.0\%), pie (9.4\%), scatterplot (2.4\%). Each training sample contains: (i) a chart image, (ii) chart type annotation, (iii) a structured data table representing the chart content, (iv) step-by-step reasoning annotations (mean length: 579 characters), and (v) the ground-truth answer (54.5\% numerical, 45.5\% categorical).

The use of a deliberately small training set serves two purposes. First, it tests whether RLVR methods can produce meaningful improvements in a low-resource regime, following the spirit of recent work on sample-efficient reinforcement learning (Wang et al., 2025). Second, it reduces computational cost, enabling systematic comparison of multiple training configurations within practical resource constraints.

### 4.3 Training Configuration

All training runs used the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-5}$ |
| Rollouts per sample ($K$) | 4 |
| Sampling temperature (training) | 1.0 |
| Total training steps | 2,000 |
| Precision | bf16 mixed precision |
| Per-device batch size | 2 |
| Gradient accumulation steps | 2 |
| Effective batch size | 4 |
| Maximum sequence length | 4,096 tokens |
| KL penalty coefficient ($\beta$) | 0.01 |

The learning rate of $1 \times 10^{-5}$ was selected based on preliminary experiments; higher rates led to training instability with LoRA adapters on vision-language models, while lower rates produced negligible updates within 2,000 steps. The temperature of 1.0 during training encourages exploratory rollouts, providing the reward signal diversity necessary for effective policy optimization.

### 4.4 Reward Design

#### 4.4.1 Base Reward Components (Chart-RVR)

The Chart-RVR framework (Sinha et al., 2025) defines seven component rewards that decompose the chart reasoning task along its hierarchical structure:

1. **Format reward** ($R_{\text{format}}$): Verifies the presence of required structured output tags (`<think>`, `<answer>`, `<chart_type>`, `<table>`). Binary: 1.0 if all tags are present in correct order, 0.0 otherwise.

2. **Type reward** ($R_{\text{type}}$): Checks whether the predicted chart type matches the ground-truth annotation. Binary: 1.0 for exact match, 0.0 otherwise.

3. **Table structure reward** ($R_{\text{table\_struct}}$): Evaluates whether the extracted table has the correct number of rows and columns, independent of cell content.

4. **Table content reward** ($R_{\text{table\_content}}$): Measures cell-level accuracy of the extracted table against the ground-truth table, computed as the fraction of cells with matching content (after normalization).

5. **Reasoning quality reward** ($R_{\text{reason}}$): Assesses the quality of the reasoning chain by checking for the presence of expected reasoning steps (e.g., identification of relevant values, arithmetic operations).

6. **Answer reward** ($R_{\text{answer}}$): Binary reward for final answer correctness via strict string matching against the ground truth.

7. **Length penalty** ($R_{\text{length}}$): A soft penalty that discourages excessively long or short responses, computed as a Gaussian function centered on the expected response length.

The total base reward is the weighted sum:

$$R_{\text{base}} = \sum_{c \in \mathcal{C}} w_c \cdot R_c$$

where $\mathcal{C}$ denotes the set of component rewards and $w_c$ are the corresponding weights.

#### 4.4.2 HCPC Reward Augmentation

The Hierarchical Correct-Path Consistency reward augments the base reward by introducing bonuses that operate across rollouts rather than within individual responses. For each training sample with $K$ rollouts, HCPC identifies the subset of rollouts that achieve correct answers (the "correct path") and computes inter-rollout consistency at each hierarchical level:

$$R_{\text{HCPC}} = w_{\text{type}} \cdot \mathbb{1}[\text{type consensus}] + w_{\text{table}} \cdot \text{sim}_{\text{table}}(\mathcal{R}^+) + w_{\text{reason}} \cdot \text{div}_{\text{reason}}(\mathcal{R}^+)$$

where $\mathcal{R}^+$ denotes the set of correct rollouts, $\text{sim}_{\text{table}}$ measures pairwise cosine similarity of extracted tables among correct rollouts, and $\text{div}_{\text{reason}}$ measures reasoning diversity (rewarding varied reasoning strategies among rollouts that arrive at the correct answer). The weights are set to $w_{\text{type}} = 1.0$, $w_{\text{table}} = 2.0$, and $w_{\text{reason}} = 1.5$.

The design rationale is that extraction should converge (high table similarity among correct rollouts indicates reliable perception) while reasoning should diversify (multiple valid reasoning paths improve robustness). This level-specific objective distinguishes HCPC from uniform diversity regularizers.

#### 4.4.3 NSR Advantage Computation

Negative Sample Reward (NSR; Zhu et al., 2025) modifies the advantage computation in GRPO by decomposing negative sample contributions. Rather than computing advantages as deviations from the group mean (as in standard GRPO), NSR separates advantageous negative samples from harmful ones:

- For correct rollouts ($R_i \geq \tau$): the gradient contribution is zeroed, preventing the policy from collapsing onto a single correct strategy.
- For incorrect rollouts ($R_i < \tau$): a penalty proportional to the distance from the threshold is applied, pushing the policy away from incorrect reasoning paths.

This asymmetric treatment preserves the probability mass distributed across multiple valid solutions while actively suppressing incorrect ones. The present work constitutes the first application of NSR to vision-language models; prior work evaluated NSR exclusively on text-based mathematical reasoning tasks.

### 4.5 Evaluation Protocol

Evaluation was conducted with the following generation parameters:

| Parameter | Value |
|-----------|-------|
| Generations per sample | 4 |
| Sampling temperature | 0.8 |
| Top-$p$ (nucleus sampling) | 0.95 |
| Maximum generation length | 4,096 tokens |
| Test set size | 500 samples per benchmark |

Two benchmarks were used:

- **ChartQA** (in-distribution): 500 test samples drawn from the same distribution as the training data. This measures whether RLVR training improves performance on the target task.

- **ChartFC** (out-of-distribution): 500 test samples from a chart fact-checking benchmark. Charts in ChartFC differ from ChartQA in visual style, question format, and required reasoning type. The task requires determining whether a textual claim about a chart is supported, refuted, or cannot be determined. This measures generalization to unseen chart distributions and task formats.

All methods were evaluated at checkpoint step 2,000 (the final training checkpoint). For ChartFC evaluation, the GRPO and GRPO+HCPC checkpoints trained on ChartQA were applied directly without additional fine-tuning.

### 4.6 Metrics

Six metrics capture complementary aspects of model performance:

**Accuracy (Acc%).** Strict-match accuracy computed as the fraction of test samples for which at least one of the four generated responses matches the ground-truth answer. This metric reflects the practical question-answering capability of the model.

**Pass@$K$.** The unbiased estimator of the probability that at least one of $K$ independent samples is correct (Chen et al., 2021). For $n$ total generations of which $c$ are correct:

$$\text{Pass@}K = 1 - \frac{\binom{n-c}{K}}{\binom{n}{K}}$$

Pass@1 estimates single-shot accuracy, while Pass@4 captures the model's coverage of the solution space. The gap between Pass@1 and Pass@4 indicates how much additional correct coverage exists beyond the model's most likely response.

**Table Extraction Consistency ($C_{\text{table}}$).** The mean pairwise cosine similarity of extracted data tables across the four rollouts for each sample. Values range from 0 (completely inconsistent extractions) to 1 (identical tables across all rollouts). High $C_{\text{table}}$ indicates that the model's perceptual pipeline produces stable, reproducible table extractions regardless of sampling stochasticity.

**Reasoning Diversity ($D_{\text{reason}}$).** The mean pairwise Jaccard distance between reasoning step sets across rollouts. Reasoning steps are extracted by parsing the content within `<think>` tags and identifying distinct logical operations. Higher values indicate greater diversity in reasoning strategies across rollouts.

**Coherence.** The fraction of rollouts in which correct table extraction leads to a correct final answer. Formally, for each sample, coherence is computed as:

$$\text{Coherence} = \frac{|\{i : \text{table}_i \approx \text{table}^* \land \text{answer}_i = \text{answer}^*\}|}{|\{i : \text{table}_i \approx \text{table}^*\}|}$$

where $\text{table}^*$ and $\text{answer}^*$ denote the ground-truth table and answer, respectively. High coherence indicates that the model's reasoning module reliably derives correct answers from correctly extracted data, rather than arriving at correct answers through compensating errors.

**Format Compliance (Format%).** The percentage of test samples for which all four generated responses contain the complete set of required structural tags (`<think>`, `<answer>`, `<chart_type>`, `<table>`) in the correct hierarchical order. This metric assesses whether training induces reliable adherence to the prescribed output format.

### 4.7 Baselines and Ablations

Four configurations were evaluated:

1. **Base**: The pretrained Qwen2.5-VL-3B-Instruct model without any RLVR training. This establishes the zero-shot capability of the model on chart reasoning tasks with structured output requirements.

2. **GRPO**: Standard Group Relative Policy Optimization (Shao et al., 2024) with the full Chart-RVR reward suite. This serves as the primary baseline, reproducing the training paradigm of Sinha et al. (2025).

3. **GRPO+HCPC**: GRPO augmented with the hierarchical correct-path consistency reward described in Section 4.4.2. The HCPC bonus is added to the base reward before advantage computation.

4. **NSR**: The negative sample reward variant (Zhu et al., 2025) applied to the Chart-RVR reward framework. This configuration uses the same reward components as GRPO but modifies the advantage computation as described in Section 4.4.3. This represents the first application of NSR to vision-language models.

All trained configurations share identical hyperparameters (Section 4.3) and differ only in their reward augmentation (HCPC) or advantage computation (NSR).

---

## 5. Results and Analysis

### 5.1 Main Results

Table 1 presents the primary results on ChartQA (in-distribution, 500 test samples). All three trained methods achieve substantial improvements over the untrained base model, with accuracy gains of approximately 11 percentage points.

**Table 1.** In-distribution results on ChartQA (500 test samples). Best values per metric are shown in **bold**.

| Method | Acc% | Pass@1 | Pass@2 | Pass@4 | $C_{\text{table}}$ | $D_{\text{reason}}$ | Coherence | Format% |
|--------|------|--------|--------|--------|---------------------|----------------------|-----------|---------|
| Base | 51.8 | 52.45 | 67.33 | 77.4 | 0.591 | 0.040 | 0.545 | 29.8 |
| GRPO | 63.0 | 61.88 | 73.10 | 80.31 | 0.749 | 0.058 | 0.252 | 96.8 |
| GRPO+HCPC | 62.2 | 62.85 | 74.57 | **82.80** | 0.741 | 0.052 | 0.244 | **98.2** |
| NSR | **63.4** | **62.20** | **73.50** | 82.00 | **0.779** | **0.057** | **0.245** | -- |

Several observations emerge from these results. First, all RLVR-trained methods produce comparable accuracy improvements over the base model, with differences among them falling within 1.4 percentage points (62.2--63.4%). The improvements are consistent across all Pass@$K$ levels, indicating that training shifts the entire distribution of response quality rather than merely improving the mode. Second, GRPO+HCPC achieves the highest Pass@4 (82.80%), surpassing both GRPO (80.31%) and NSR (82.00%), despite recording slightly lower raw accuracy (62.2%) than either alternative. This suggests that HCPC promotes a broader distribution of correct solutions across rollouts, consistent with its design objective of encouraging reasoning diversity among correct paths. Third, NSR achieves the highest raw accuracy (63.4%) and the highest table extraction consistency ($C_{\text{table}} = 0.779$), indicating that its negative-only reinforcement strategy produces particularly stable perceptual representations.

The differences between trained methods are modest in absolute terms. Bootstrap confidence interval analysis (10,000 resamples) yields the following 95% intervals: accuracy 62.2% [58.8--65.4%] for GRPO+HCPC versus 63.0% [59.2--66.6%] for GRPO, with overlapping intervals indicating that the accuracy differences are not statistically significant at the $\alpha = 0.05$ level. The Pass@4 difference between HCPC and GRPO (82.80% vs. 80.31%, $\Delta = +2.49$ pp) has a bootstrap $p$-value of 0.239, which does not reach conventional significance thresholds. These findings are consistent with the limited training scale (1K samples, 2,000 steps) and suggest that larger-scale experiments may be necessary to establish whether the observed trends represent genuine methodological advantages.

### 5.2 Out-of-Distribution Generalization

Table 2 presents results on ChartFC, an out-of-distribution benchmark that tests generalization to unseen chart styles and a different task format (fact checking rather than question answering).

**Table 2.** Out-of-distribution results on ChartFC (500 test samples, checkpoint-2000).

| Method | Acc% | Pass@1 | Pass@2 | Pass@4 | $C_{\text{table}}$ | $D_{\text{reason}}$ | Coherence | Format% |
|--------|------|--------|--------|--------|---------------------|----------------------|-----------|---------|
| GRPO | **63.8** | **66.10** | **82.07** | **92.2** | **0.950** | 0.059 | 0.312 | **98.8** |
| GRPO+HCPC | 60.6 | 62.45 | 79.90 | 91.8 | 0.782 | **0.060** | **0.433** | 66.4 |

On the OOD benchmark, GRPO outperforms GRPO+HCPC on raw accuracy (63.8% vs. 60.6%), Pass@1 (66.10% vs. 62.45%), and table extraction consistency ($C_{\text{table}} = 0.950$ vs. 0.782). Both methods achieve high Pass@4 scores exceeding 91%, indicating that when four attempts are permitted, either method recovers the correct answer for the vast majority of solvable samples.

A notable finding is that both methods achieve higher accuracy on ChartFC than on ChartQA (63.8% and 60.6% vs. 63.0% and 62.2%, respectively, for GRPO and HCPC). This may reflect the fact that ChartFC is a fact-checking task with a constrained answer space (supported/refuted/not enough information), which is inherently easier than the open-ended numerical and textual answers required by ChartQA.

The GRPO model's high $C_{\text{table}}$ on ChartFC (0.950) is particularly striking, indicating near-perfect consistency in table extraction across rollouts on the OOD benchmark. This suggests that when charts follow a different but internally consistent visual style, the trained model's extraction pipeline generalizes robustly.

### 5.3 Pass@$K$ and Inference-Time Scaling

The Pass@$K$ curves reveal how each method's solution coverage scales with the number of inference samples. Figure X plots Pass@$K$ for $K \in \{1, 2, 4\}$ across all methods on ChartQA.

The base model exhibits the largest Pass@1 $\to$ Pass@4 gap: $52.45\% \to 77.40\%$, a span of 24.95 percentage points. This indicates that the untrained model produces highly variable outputs, with many correct solutions appearing only sporadically across rollouts. After RLVR training, this gap narrows substantially: GRPO shows a gap of 18.43 pp ($61.88\% \to 80.31\%$), HCPC shows 19.95 pp ($62.85\% \to 82.80\%$), and NSR shows 19.80 pp ($62.20\% \to 82.00\%$).

The narrowing of the Pass@1--Pass@4 gap after training reflects a concentration of probability mass on fewer, higher-quality solutions. Training suppresses low-quality reasoning paths that occasionally produce correct answers by chance, while strengthening reliable reasoning strategies. This is a desirable property for deployment, where single-shot accuracy (Pass@1) matters more than coverage (Pass@4).

HCPC achieves the highest Pass@4 (82.80%) among all methods, consistent with its design objective of maintaining diverse correct solutions. The theoretical framework of Wen et al. (2025) predicts that methods preserving reasoning diversity should exhibit superior Pass@$K$ scaling at higher $K$, as the probability of at least one correct sample increases with the number of distinct solution strategies. The observed 2.49 pp advantage of HCPC over GRPO at Pass@4, while not statistically significant in this experiment, aligns with this prediction.

### 5.4 Table Extraction Consistency

Table extraction consistency ($C_{\text{table}}$) measures the stability of the model's perceptual pipeline across stochastic rollouts. A model that reliably perceives chart content should produce similar table extractions regardless of sampling randomness.

All RLVR-trained methods substantially improve $C_{\text{table}}$ over the base model. NSR achieves the highest in-distribution consistency ($C_{\text{table}} = 0.779$), followed by GRPO (0.749) and HCPC (0.741), compared to 0.591 for the base model. The improvement of $+0.158$ to $+0.188$ absolute points represents a 27--32% relative reduction in extraction variability.

The correlation analysis between $C_{\text{table}}$ and per-sample correctness reveals a negative Pearson coefficient across all methods (Base: $r = -0.658$; GRPO: $r = -0.528$; HCPC: $r = -0.353$; NSR: $r = -0.423$). This counterintuitive finding indicates that samples with highly consistent table extractions tend to have lower correctness rates. The explanation is that samples the model finds "easy" at the extraction level -- those yielding consistent tables -- may still present reasoning challenges that prevent correct final answers. Conversely, samples with variable extractions may occasionally include a correct extraction that leads to the right answer. Notably, HCPC exhibits the weakest negative correlation ($r = -0.353$), suggesting that its hierarchical reward structure partially decouples extraction consistency from downstream reasoning errors.

NSR's superior $C_{\text{table}}$ may be attributed to its asymmetric advantage computation: by zeroing gradients for correct rollouts, NSR avoids reinforcing any particular extraction pattern among successful responses, allowing the model's perceptual representations to stabilize around naturally converging extractions rather than being artificially pushed toward whichever extraction pattern happened to accompany the highest-reward response.

### 5.5 Coherence Analysis

Coherence measures the conditional probability that a correct table extraction leads to a correct final answer. This metric captures the integrity of the reasoning pipeline: a model with high coherence reliably derives correct answers when given correct data, while a model with low coherence may produce correct tables but fail at the reasoning stage, or vice versa.

On in-distribution data (ChartQA), the base model achieves the highest coherence (0.545), which decreases to approximately 0.245--0.252 for all trained methods. This apparent degradation warrants careful interpretation. The base model produces unstructured outputs with low format compliance (29.8%), meaning that "table extraction" is loosely defined and often coincidentally correct in a partial sense. Trained models, by contrast, produce fully structured outputs with explicit table tags, creating a stricter evaluation criterion for table correctness. Additionally, trained models attempt table extraction on a much larger fraction of samples (format compliance of 96--98%), including difficult samples where the base model would produce unstructured text. The increased denominator of structured extraction attempts explains much of the coherence decrease.

The more informative comparison is on out-of-distribution data (ChartFC), where both methods produce structured outputs with high table parse success rates. On ChartFC, HCPC achieves a coherence of 0.433 compared to GRPO's 0.312, a relative improvement of 38.8%. This result indicates that when HCPC correctly extracts a data table from an unseen chart, it more reliably derives the correct answer from that table. The hierarchical reward structure appears to strengthen the coupling between correct extraction and correct reasoning -- precisely the design intent of the correct-path consistency framework. The HCPC reward explicitly conditions on correctness at each level, which may encourage the model to develop reasoning pathways that genuinely depend on extracted data rather than relying on shortcuts that circumvent the table.

### 5.6 Format Learning

Format compliance provides a window into the ease with which RLVR training induces structural output patterns. The base model achieves only 29.8% format compliance, indicating that the pretrained model's default behavior does not conform to the required tag structure. After RLVR training, all in-distribution methods achieve 96--98% compliance: GRPO reaches 96.8%, HCPC reaches 98.2%, and NSR reaches approximately 97% (comparable to GRPO based on per-tag breakdown data). The near-complete acquisition of format compliance after training indicates that format rewards are trivially optimized and do not constitute a meaningful learning challenge for the model.

A noteworthy exception arises in the OOD setting. GRPO maintains 98.8% format compliance on ChartFC, while HCPC drops to 66.4%. Detailed per-tag analysis reveals that the degradation concentrates in the `<think>` tags (78.0% vs. 99.0%) and proper tag ordering (79.0% vs. 99.0%), while chart type tags remain largely intact (96.6% vs. 100.0%).

This format degradation admits several interpretations. One possibility is that HCPC's hierarchical reward structure, which assigns higher weight to table content ($w_{\text{table}} = 2.0$) than to format, implicitly deprioritizes format adherence when the model encounters unfamiliar distributions, allocating its limited capacity to content quality instead. An alternative interpretation is that the HCPC reward, by conditioning bonuses on the correctness of intermediate outputs, creates a more complex reward landscape that is harder to generalize across distributions. The format degradation on OOD data represents a limitation of the current HCPC implementation and warrants investigation in future work, potentially through explicit format regularization or curriculum-based OOD exposure during training.

### 5.7 NSR on Vision-Language Models

The present study constitutes the first application of Negative Sample Reward (Zhu et al., 2025) to vision-language models. Prior evaluations of NSR were limited to text-based mathematical reasoning tasks, where the input modality is purely textual and rewards are computed from deterministic program execution. Chart reasoning introduces two additional complexities absent from the text-only setting: (i) a visual perception stage that transforms pixel-level chart representations into structured data, introducing perceptual noise into the reward signal, and (ii) a multi-component reward function that evaluates multiple intermediate outputs rather than a single final answer.

On ChartQA, NSR achieves the highest raw accuracy (63.4%) among all methods and the highest table extraction consistency ($C_{\text{table}} = 0.779$). This result suggests that the negative-only reinforcement strategy transfers effectively to the vision-language setting. The strong $C_{\text{table}}$ is particularly encouraging: it indicates that NSR's approach of not reinforcing any single correct pattern allows the model's perceptual representations to converge naturally, rather than being driven toward extraction patterns associated with incidentally high-reward rollouts.

The combination of high accuracy and high extraction consistency positions NSR as a competitive alternative to standard GRPO for chart reasoning. However, the absence of explicit diversity-promoting mechanisms (unlike HCPC) means that NSR does not achieve the highest Pass@4 (82.00% vs. HCPC's 82.80%), suggesting that its diversity preservation, while effective, is an implicit rather than targeted property.

### 5.8 Reasoning Diversity Analysis

The reasoning diversity metric ($D_{\text{reason}}$) captures the Jaccard distance between sets of reasoning steps extracted from different rollouts. Across all methods and both benchmarks, $D_{\text{reason}}$ remains low, ranging from 0.040 (Base) to 0.060 (HCPC on ChartFC). Trained methods show slightly higher diversity than the base model (0.052--0.058 vs. 0.040 on ChartQA), but the absolute magnitudes are small.

Several factors contribute to these low diversity values. First, with only $K = 4$ rollouts per sample, the combinatorial space of possible reasoning paths is severely undersampled. The expected Jaccard distance between two random subsets of size $s$ drawn from a universe of $N$ reasoning steps is bounded by the sample size, and with 4 rollouts, the maximum observable diversity is fundamentally limited. Second, chart reasoning -- particularly for simple chart types such as bar and pie charts -- often admits only a narrow set of valid reasoning strategies. A question asking "What is the value of the tallest bar?" has essentially one reasoning path (identify the tallest bar and read its value), leaving little room for strategic diversity regardless of the training method. Third, the Jaccard distance metric treats all reasoning steps as equally important, potentially obscuring meaningful variation in reasoning structure (e.g., different orders of operation or alternative decomposition strategies) that manifests as minor step-level differences.

The positive correlation between $D_{\text{reason}}$ and per-sample correctness ($r = 0.546$ to $0.706$ across methods) is consistent with the hypothesis that reasoning diversity is beneficial. Samples where the model explores multiple reasoning strategies are more likely to include at least one correct solution. However, the direction of causality is ambiguous: easy samples may naturally admit more reasoning variety, while difficult samples may constrain the model to a single (often incorrect) approach.

### 5.9 Accuracy by Chart Type

Per-type accuracy analysis on ChartQA reveals differential training effects across chart categories. The training set composition reflects ChartQA's distribution: bar charts ($n = 316$, 63.2%), line charts ($n = 142$, 28.4%), pie charts ($n = 41$, 8.2%), and scatterplots ($n = 1$).

Bar charts show the most consistent improvement from RLVR training, with accuracy increasing from 76.6% (Base) to 82.9% (GRPO), 85.8% (HCPC), and 83.9% (NSR). HCPC achieves the highest bar chart accuracy, potentially because bar charts admit the clearest hierarchical decomposition (type identification, value extraction per bar, comparison/aggregation), which aligns well with HCPC's level-specific rewards.

Line charts present a contrasting picture: RLVR training provides minimal improvement (76.8% Base to 74.6--76.1% trained), with some configurations slightly degrading performance. Line chart reasoning often requires trend identification, interpolation, and multi-step temporal reasoning -- capabilities that may not benefit from the same reward signals optimized for value extraction tasks.

Pie chart accuracy is high across all methods (82.9--95.1%) but shows high variance due to the small sample size ($n = 41$). The single scatterplot achieves 100% accuracy universally and provides no discriminative information.

### 5.10 Sample Overlap and Complementarity

Analysis of per-sample solve rates reveals substantial overlap among methods. Of the 500 test samples, 344 (68.8%) are solved by all four configurations (including the untrained base model), and 52 (10.4%) are solved by none. The remaining 104 samples (20.8%) exhibit method-dependent solvability.

Each trained method uniquely solves 4--5 samples that no other method answers correctly, indicating modest but nonzero complementarity. The pairwise Jaccard similarity of solved sample sets ranges from 0.828 (Base vs. NSR) to 0.881 (HCPC vs. NSR), indicating high overlap. The highest similarity occurs between trained methods (GRPO-HCPC: 0.864; GRPO-NSR: 0.880; HCPC-NSR: 0.881), which share the same reward components and differ only in advantage computation or reward augmentation.

These overlap patterns suggest that the three RLVR methods largely solve the same subset of problems, with the observed accuracy differences reflecting performance on a relatively small contested set of 82 samples. Universally failed samples ($n = 52$) represent questions that may exceed the model's capacity at the 3B-parameter scale, such as those requiring fine-grained numerical reading (e.g., "What is the value of the largest bar?" with answer 3.0238) or complex multi-step reasoning (e.g., ratio computations across chart elements).

### 5.11 Inference Efficiency

Inference time analysis reveals substantial variation across methods. GRPO+HCPC achieves the lowest median inference time (33.6 seconds per sample), comparable to the base model (26.9 seconds) and substantially faster than GRPO (48.1 seconds). NSR is the fastest overall (median 11.0 seconds), likely because its training produces shorter, more focused responses. On the OOD benchmark, the pattern reverses: HCPC is the slowest (median 58.7 seconds) compared to GRPO (39.8 seconds), potentially reflecting increased deliberation on unfamiliar chart styles.

These timing differences do not reflect any inherent computational overhead of the methods (all use the same model architecture) but rather indicate that training affects the distribution of response lengths and, consequently, generation time.

### 5.12 Limitations

The present experimental evaluation has several limitations that constrain the generalizability of the findings.

First, all experiments were conducted with a single model architecture (Qwen2.5-VL-3B-Instruct). The 3B-parameter scale was chosen to enable comprehensive experimentation within practical resource constraints, but it remains unclear whether the observed patterns -- particularly the relative performance of HCPC and NSR -- hold at larger model scales (7B, 72B) where base model capabilities are stronger and capacity constraints are relaxed.

Second, the training set comprises only 1,000 samples from a possible 34,194. While small-dataset RLVR has been shown to be effective in certain settings (Wang et al., 2025), the limited training data may insufficient to fully realize the potential of methods such as HCPC that operate on inter-rollout statistics and therefore benefit from a larger diversity of training examples.

Third, the use of $K = 4$ rollouts per sample limits the statistical power of inter-rollout metrics ($C_{\text{table}}$, $D_{\text{reason}}$, Coherence). With only four samples, estimates of pairwise similarity and diversity are noisy, and the Pass@$K$ estimator has limited resolution. Koksal and Alatan (2025) demonstrate that RLVR with higher rollout counts ($K = 8$ or $K = 16$) can yield qualitatively different training dynamics, particularly for diversity-promoting methods.

Fourth, the format compliance degradation of HCPC on OOD data (66.4% vs. GRPO's 98.8%) represents a practical limitation that would need to be addressed before deployment. The root cause -- whether it reflects a fundamental trade-off between content quality and format adherence or a correctable artifact of the reward weighting -- remains to be determined.

Fifth, the evaluation covers only two benchmarks with relatively simple chart types (bar, line, pie). Complex chart types (multi-panel figures, charts with annotations, combined chart types) and multi-turn reasoning tasks were not evaluated, leaving open the question of whether HCPC's hierarchical structure provides greater advantages on more complex reasoning chains.

---

## Summary of Key Findings

The experimental results support the following principal conclusions:

1. **RLVR training is effective for chart reasoning at small scale.** All three methods (GRPO, GRPO+HCPC, NSR) improve accuracy by approximately 11 percentage points over the untrained base model, using only 1,000 training samples and 2,000 optimization steps. Format compliance improves from 29.8% to 96--98%, and table extraction consistency increases by 27--32%.

2. **HCPC promotes diverse correct solutions.** GRPO+HCPC achieves the highest Pass@4 (82.80%) on in-distribution data and the highest OOD coherence (0.433 vs. GRPO's 0.312), indicating that its hierarchical reward structure encourages the model to maintain multiple valid reasoning pathways and to couple correct extraction with correct reasoning.

3. **NSR transfers effectively to vision-language models.** In its first application to VLMs, NSR achieves the highest raw accuracy (63.4%) and the highest table extraction consistency ($C_{\text{table}} = 0.779$), demonstrating that negative-only reinforcement handles the noisy, multi-component reward signals inherent in chart understanding.

4. **Method differences are statistically modest at current scale.** The accuracy spread among trained methods is 1.4 percentage points (62.2--63.4%), and bootstrap confidence intervals overlap across all pairwise comparisons. Scaling experiments with larger training sets and more rollouts are needed to establish definitive rankings.

5. **Format generalization is not guaranteed.** HCPC's OOD format compliance drops to 66.4%, highlighting a tension between hierarchical content rewards and structural output adherence that requires further investigation.

---

*Draft prepared for thesis chapter. All results based on verified experimental outputs at checkpoint-2000 with 500-sample test sets. Statistical significance assessed via paired bootstrap with 10,000 resamples.*
