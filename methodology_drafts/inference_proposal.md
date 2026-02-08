# Test-Time Multi-Agent Reinforcement Learning for Chart Reasoning

**A Novel Inference-Time Framework for Visual Data Understanding**

---

## Executive Summary

This proposal presents a novel inference-time framework for chart reasoning that combines test-time reinforcement learning with multi-agent systems. Unlike existing training-based approaches, our method operates entirely at inference time, making it accessible to any vision-language model including closed-source APIs.

**Key Innovation**: First application of test-time compute scaling to chart question answering, combining structured representation extraction, multi-agent coordination, and iterative verification-guided refinement.

**Expected Impact**: 10-15% accuracy improvement over baseline VLMs with zero training requirements, making advanced chart reasoning accessible to practitioners with limited resources.

---

## 1. Problem Statement

### 1.1 Current Limitations

Vision-language models struggle with chart understanding, exhibiting three critical failure modes:

1. **Perceptual Errors**: Misidentifying chart types, incorrectly reading axis labels, failing to extract data values accurately
2. **Reasoning Failures**: Correct data extraction but incorrect mathematical operations or logical inferences  
3. **Consistency Issues**: Different answers across multiple attempts despite deterministic underlying reasoning

### 1.2 Existing Approach Limitations

Current solutions rely on:
- **Expensive fine-tuning**: Chart-RVR requires 6k training samples
- **High compute requirements**: Training 3B+ parameter models on 24GB+ GPUs
- **Limited accessibility**: Cannot apply to proprietary API models (GPT-4V, Gemini)
- **Data dependency**: Performance degrades significantly with <1k training samples

### 1.3 Research Gap

While test-time compute scaling has shown success in:
- Text reasoning (o1/o3 models)
- Chart generation (METAL - ACL 2025)
- Math problem solving (various RL approaches)

**No existing work applies test-time RL to chart understanding/question answering.**

---

## 2. Proposed Approach

### 2.1 System Architecture

Our framework consists of a **four-stage iterative pipeline**:

```
Chart Image 
    ↓
[1] Formal Representation Extraction (Perception Agent)
    ↓
[2] Verification (Verification Agent)
    ↓  
[3] Question Answering (Reasoning Agent)
    ↓
[4] Critique & Refinement (Critique Agent)
    ↓
    ← Iterative Loop (N=1-5 rounds) ←
    ↓
Final Answer
```

### 2.2 Detailed Tools & Methods

This section specifies **exact tools, libraries, and methods** for each component to eliminate ambiguity and clarify novelty.

---

## 3. Implementation Specifications

### 3.1 Component 1: Formal Representation Extraction

#### Tools & Libraries

**Vision Models** (choose one):
- **Option A - API-based**: 
  - GPT-4V (gpt-4-vision-preview) via OpenAI API
  - Claude 3.5 Sonnet via Anthropic API
  - Gemini 1.5 Pro via Google AI API
  
- **Option B - Local models**:
  - Qwen2.5-VL-3B (HuggingFace: `Qwen/Qwen2.5-VL-3B-Instruct`)
  - InternVL2-4B (HuggingFace: `OpenGVLab/InternVL2-4B`)
  - MiniCPM-V-2.6 (HuggingFace: `openbmb/MiniCPM-V-2_6`)

**OCR Tools**:
- **Primary**: Tesseract OCR (`pytesseract` library)
- **Backup**: EasyOCR (`easyocr` library) for better accuracy on small text
- **Code**: 
  ```python
  import pytesseract
  from PIL import Image
  text = pytesseract.image_to_string(Image.open('chart.png'))
  ```

**Chart Type Detection**:
- **Method**: Zero-shot classification with vision model
- **Prompt**: "What type of chart is this? Options: bar, line, pie, scatter, histogram, box plot, heatmap. Answer with one word."
- **Fallback**: Use matplotlib chart classifier if needed

**Representation Formats**:

1. **JSON Format** (for bar/pie charts):
   ```python
   import json
   representation = {
       "chart_type": "bar",
       "title": "extracted_title",
       "x_axis": {
           "label": "Year",
           "values": [2020, 2021, 2022]
       },
       "y_axis": {
           "label": "Revenue ($M)",
           "range": [0, 150]
       },
       "data_points": [
           {"x": 2020, "y": 100, "category": "Product A"},
           {"x": 2021, "y": 125, "category": "Product A"}
       ],
       "uncertainty": {
           "y_axis_precision": "approximate",
           "ocr_confidence": 0.87
       }
   }
   ```

2. **Code Format** (for scatter/complex charts):
   ```python
   # Use matplotlib code generation
   template = '''
   import matplotlib.pyplot as plt
   import numpy as np
   
   x = {x_values}
   y = {y_values}
   
   plt.figure(figsize=(8, 6))
   plt.scatter(x, y)
   plt.xlabel('{x_label}')
   plt.ylabel('{y_label}')
   plt.title('{title}')
   plt.show()
   '''
   ```

3. **Table Format** (for time series):
   ```python
   # Markdown table
   table = """
   | Year | Revenue |
   |------|---------|
   | 2020 | 100     |
   | 2021 | 125     |
   | 2022 | 140     |
   """
   ```

#### Prompting Strategy

**Extraction Prompt Template**:
```
You are a precise chart data extractor. Given this chart image:

1. Identify the chart type
2. Extract all axis labels and their values
3. Extract all data points with their exact coordinates
4. Note any uncertainty (e.g., "value approximately X" if hard to read)

Output format: {JSON/Code/Table based on chart type}

Include uncertainty markers where appropriate:
- "approximate" if axis labels are unclear
- "occluded" if parts of chart are hidden
- confidence score (0-1) for each extracted value
```

**Adaptive Selection Logic**:
```python
def select_representation(chart_type, complexity):
    if chart_type in ['bar', 'pie']:
        return 'json'
    elif chart_type in ['scatter', 'bubble', 'contour']:
        return 'code'
    elif chart_type in ['line', 'area', 'time_series']:
        return 'table'
    else:
        return 'json'  # default
```

#### Novelty Claim
- **Not Novel**: Chart-to-structured conversion (DePlot, StructChart, MatCha all do this)
- **Novel Aspect**: Adaptive representation selection + explicit uncertainty quantification in extraction

---

### 3.2 Component 2: Verification Agent

#### Tools & Methods

**For Code Verification** (when representation is Python code):

**Method**: Execute code and compare rendered output to original image

**Libraries**:
```python
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
```

**Implementation**:
```python
def verify_code_representation(generated_code, original_image):
    # Step 1: Execute code in sandboxed environment
    try:
        exec_globals = {}
        exec(generated_code, exec_globals)
        
        # Capture matplotlib output
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        generated_image = Image.open(buf)
        
    except Exception as e:
        return {'confidence': 0.0, 'error': str(e)}
    
    # Step 2: Compare images using MSE and SSIM
    original_array = np.array(original_image)
    generated_array = np.array(generated_image)
    
    # Mean Squared Error
    mse = np.mean((original_array - generated_array) ** 2)
    
    # Structural Similarity Index
    ssim_score = ssim(original_array, generated_array, multichannel=True)
    
    # Combined confidence score
    confidence = (ssim_score + (1 - mse/10000)) / 2
    
    return {
        'confidence': confidence,
        'mse': mse,
        'ssim': ssim_score,
        'errors': [] if confidence > 0.8 else ['low_visual_similarity']
    }
```

**For JSON/Table Verification**:

**Method**: Cross-check extracted values against original image using OCR + visual inspection

**Libraries**:
```python
import pytesseract
import cv2
import easyocr
from difflib import SequenceMatcher
```

**Implementation**:
```python
def verify_json_representation(extracted_data, original_image):
    # Step 1: OCR on original image to get all text
    ocr_text = pytesseract.image_to_string(original_image)
    
    # Step 2: Extract all values from JSON representation
    extracted_values = []
    def flatten_values(d):
        if isinstance(d, dict):
            for v in d.values():
                flatten_values(v)
        elif isinstance(d, list):
            for item in d:
                flatten_values(item)
        else:
            extracted_values.append(str(d))
    
    flatten_values(extracted_data)
    
    # Step 3: Check if extracted values appear in OCR text
    matches = 0
    total = len(extracted_values)
    
    for value in extracted_values:
        # Fuzzy matching (allows for small OCR errors)
        if any(SequenceMatcher(None, value, ocr_word).ratio() > 0.85 
               for ocr_word in ocr_text.split()):
            matches += 1
    
    confidence = matches / total if total > 0 else 0.0
    
    # Step 4: Visual validation using object detection
    # Detect chart elements (bars, points, lines)
    element_count = detect_chart_elements(original_image)
    data_point_count = len(extracted_data.get('data_points', []))
    
    element_match = 1.0 if abs(element_count - data_point_count) <= 2 else 0.5
    
    # Combined score
    final_confidence = (confidence * 0.7 + element_match * 0.3)
    
    return {
        'confidence': final_confidence,
        'value_match_rate': confidence,
        'element_match': element_match,
        'errors': identify_errors(confidence, element_match)
    }

def detect_chart_elements(image):
    # Use OpenCV to detect bars, points, or lines
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)
```

**Confidence Thresholds**:
```python
VERIFICATION_THRESHOLDS = {
    'high_confidence': 0.85,  # Proceed with reasoning
    'medium_confidence': 0.60,  # Refine extraction once
    'low_confidence': 0.40,  # Refine extraction twice
    'failed': 0.40  # Below this, mark as extraction failure
}
```

#### Error Localization

**Method**: Identify which specific parts of extraction are wrong

```python
def localize_errors(verification_result, extracted_data):
    errors = []
    
    if verification_result['confidence'] < 0.6:
        # Check specific fields
        if 'x_axis' in extracted_data:
            if not verify_axis(extracted_data['x_axis']):
                errors.append({
                    'location': 'x_axis',
                    'issue': 'values_mismatch',
                    'suggestion': 're-extract with focus on x-axis labels'
                })
        
        if 'data_points' in extracted_data:
            if len(extracted_data['data_points']) == 0:
                errors.append({
                    'location': 'data_points',
                    'issue': 'no_data_extracted',
                    'suggestion': 'increase OCR sensitivity'
                })
    
    return errors
```

#### Novelty Claim
- **Not Novel**: Code verification via re-rendering (RECODE does this with MSE)
- **Novel Aspect**: Multi-method verification (code + OCR + visual element detection) with error localization

---

### 3.3 Component 3: Reasoning Agent

#### Tools & Methods

**Language Models** (choose one):

**Option A - API**:
- GPT-4 Turbo (`gpt-4-turbo-preview`)
- Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)
- Gemini 1.5 Pro (`gemini-1.5-pro`)

**Option B - Local**:
- LLaMA 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- Qwen2.5 7B (`Qwen/Qwen2.5-7B-Instruct`)
- Mistral 7B v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)

**Reasoning Method**: Chain-of-Thought prompting on structured representation

**Prompt Template**:
```
You are a precise data analyst. Given this structured chart data:

{structured_representation}

Question: {question}

Think step-by-step:
1. Identify what data is needed to answer the question
2. Extract the relevant values from the structured data
3. Perform any necessary calculations
4. State your final answer

Format your response as:
Thought: [your reasoning process]
Calculation: [any math performed]
Answer: [final answer]
```

**Implementation**:
```python
def reason_over_representation(representation, question, model='gpt-4-turbo'):
    prompt = f"""
Structured chart data:
{json.dumps(representation, indent=2)}

Question: {question}

Provide step-by-step reasoning:
1. What values do I need?
2. Where are they in the data?
3. What calculation is needed?
4. What is the final answer?

Format:
Step 1: [identify needed values]
Step 2: [locate in data structure]
Step 3: [perform calculation if needed]
Answer: [final answer]
"""
    
    response = call_llm(model, prompt)
    
    # Parse response
    answer = extract_final_answer(response)
    reasoning_trace = extract_reasoning_steps(response)
    
    return {
        'answer': answer,
        'reasoning': reasoning_trace,
        'full_response': response
    }
```

**Answer Extraction**:
```python
import re

def extract_final_answer(response):
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*(.+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: last line
    return response.strip().split('\n')[-1]

def extract_reasoning_steps(response):
    # Extract numbered steps
    steps = re.findall(r'Step \d+:\s*(.+)', response)
    return steps if steps else [response]
```

#### Novelty Claim
- **Not Novel**: Chain-of-thought reasoning on structured data (standard practice)
- **No novel claim here**: This is a standard component

---

### 3.4 Component 4: Critique Agent

#### Tools & Methods

**Language Model**: Same as Reasoning Agent (GPT-4/Claude/LLaMA)

**Critique Generation Method**: Error-aware feedback generation

**Prompt Template**:
```
You are a quality control expert for chart data extraction and reasoning.

Original chart: [image]
Extracted representation: {representation}
Verification confidence: {confidence}
Identified errors: {errors}
Question: {question}
Answer generated: {answer}
Reasoning trace: {reasoning}

Your task: Provide targeted feedback for improvement.

If verification confidence < 0.6:
  Focus on EXTRACTION errors - which specific values are wrong?
  
If verification confidence > 0.8 but answer seems wrong:
  Focus on REASONING errors - which calculation step failed?

Output format:
Error type: [extraction / reasoning / both]
Specific issue: [detailed description]
Refinement strategy: [what to retry and how]
```

**Implementation**:
```python
def generate_critique(representation, verification_result, question, answer, reasoning):
    confidence = verification_result['confidence']
    errors = verification_result.get('errors', [])
    
    # Decision logic
    if confidence < 0.6:
        focus = "extraction"
        guidance = "Re-extract data with focus on: " + ", ".join([e['location'] for e in errors])
    elif confidence > 0.8:
        focus = "reasoning"
        guidance = "Re-check calculation steps, particularly: " + identify_weak_step(reasoning)
    else:
        focus = "both"
        guidance = "Refine both extraction and reasoning"
    
    critique_prompt = f"""
Analyze this chart understanding attempt:

Extracted data confidence: {confidence}
Errors detected: {errors}
Question: {question}
Answer: {answer}

What specifically went wrong? Provide targeted feedback for {focus}.

Format:
Issue: [specific problem]
Location: [where in the pipeline]
Fix: [concrete action to take]
"""
    
    critique = call_llm('gpt-4-turbo', critique_prompt)
    
    return {
        'focus': focus,
        'critique': critique,
        'refinement_target': errors[0]['location'] if errors else 'full_extraction',
        'suggested_action': guidance
    }

def identify_weak_step(reasoning_steps):
    # Heuristic: look for calculation steps with numbers
    for i, step in enumerate(reasoning_steps):
        if re.search(r'\d+\s*[+\-*/]\s*\d+', step):
            return f"calculation in step {i+1}"
    return "final answer derivation"
```

**Refinement Decision Tree**:
```python
def decide_refinement_action(critique):
    if critique['focus'] == 'extraction':
        return {
            'action': 're-extract',
            'parameters': {
                'focus_areas': critique['refinement_target'],
                'increase_ocr_sensitivity': True,
                'prompt_modification': 'Pay special attention to ' + critique['refinement_target']
            }
        }
    elif critique['focus'] == 'reasoning':
        return {
            'action': 're-reason',
            'parameters': {
                'prompt_modification': 'Double-check your calculation in: ' + critique['refinement_target'],
                'require_verification': True
            }
        }
    else:
        return {
            'action': 'both',
            'parameters': {
                'extraction_focus': critique['refinement_target'],
                'reasoning_emphasis': 'careful_calculation'
            }
        }
```

#### Novelty Claim
- **Not Novel**: Self-critique / reflection (used in various LLM papers)
- **Novel Aspect**: Targeted critique that distinguishes extraction vs. reasoning errors and guides selective refinement

---

### 3.5 Test-Time Reinforcement Learning

#### Method: Adaptive Iteration Policy

**Objective**: Learn when to stop iterating vs. continue refining

**Policy Network Architecture**:
```python
import torch
import torch.nn as nn

class IterationPolicy(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Continue or Stop
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)

# State representation
def get_state(iteration_num, confidence, answer_entropy, question_complexity):
    return torch.tensor([
        iteration_num / 5.0,  # Normalized iteration count
        confidence,  # Verification confidence
        answer_entropy,  # Uncertainty in answer
        question_complexity,  # Estimated difficulty
        # Add more features...
    ])
```

**Training Signal**:
```python
# Use small validation set (100 samples) to learn policy
validation_set = load_chartqa_subset(100)

for sample in validation_set:
    rewards = []
    
    for n_iterations in range(1, 6):
        accuracy = run_pipeline(sample, max_iterations=n_iterations)
        cost = n_iterations * 0.05  # $0.05 per iteration
        reward = accuracy - 0.1 * cost  # Accuracy - cost penalty
        rewards.append(reward)
    
    # Train policy to predict optimal stopping point
    optimal_n = np.argmax(rewards)
    train_step(policy_network, sample_features, optimal_n)
```

**Reward Function**:
```python
def calculate_reward(answer_correct, confidence, iterations, cost_per_iter=0.05):
    # Reward components
    accuracy_reward = 1.0 if answer_correct else 0.0
    confidence_bonus = confidence * 0.2  # Bonus for high confidence
    cost_penalty = iterations * cost_per_iter * 0.1  # Small cost penalty
    
    total_reward = accuracy_reward + confidence_bonus - cost_penalty
    return total_reward
```

**Inference-Time Usage**:
```python
def adaptive_iteration(chart, question, policy_network, max_iterations=5):
    for i in range(max_iterations):
        # Run pipeline iteration
        result = run_single_iteration(chart, question, iteration=i)
        
        # Get state
        state = get_state(
            iteration_num=i,
            confidence=result['confidence'],
            answer_entropy=calculate_entropy(result['answer']),
            question_complexity=estimate_complexity(question)
        )
        
        # Query policy
        action_probs = policy_network(state)
        action = torch.argmax(action_probs)  # 0=continue, 1=stop
        
        if action == 1 or i == max_iterations - 1:
            return result
    
    return result
```

#### Baseline: Fixed Iteration Count

For comparison, also implement:
```python
def fixed_iteration_baseline(chart, question, n=3):
    # Always do exactly N iterations
    for i in range(n):
        result = run_single_iteration(chart, question, iteration=i)
    return result
```

#### Novelty Claim
- **Novel**: First application of test-time RL policy learning for chart QA
- **Prior work**: TTRL (2024) does test-time RL but for text/code, not vision-language
- **Key difference**: Our policy operates over visual verification confidence + reasoning quality

---

### 3.6 Full Pipeline Integration

#### Complete Algorithm

```python
def multi_agent_chart_qa(chart_image, question, max_iterations=5, use_learned_policy=True):
    """
    Main pipeline integrating all components
    """
    iteration_history = []
    
    for iteration in range(max_iterations):
        # === STAGE 1: PERCEPTION ===
        representation = perception_agent.extract(
            chart_image=chart_image,
            critique_feedback=iteration_history[-1]['critique'] if iteration > 0 else None
        )
        
        # === STAGE 2: VERIFICATION ===
        verification = verification_agent.verify(
            representation=representation,
            original_image=chart_image
        )
        
        # === STAGE 3: REASONING ===
        reasoning_result = reasoning_agent.answer(
            representation=representation,
            question=question
        )
        
        # === STAGE 4: CRITIQUE ===
        critique = critique_agent.analyze(
            representation=representation,
            verification=verification,
            question=question,
            answer=reasoning_result['answer'],
            reasoning=reasoning_result['reasoning']
        )
        
        # Store iteration result
        iteration_history.append({
            'iteration': iteration,
            'representation': representation,
            'confidence': verification['confidence'],
            'answer': reasoning_result['answer'],
            'critique': critique
        })
        
        # === STOPPING DECISION ===
        if use_learned_policy:
            state = construct_state(iteration, verification, reasoning_result, question)
            should_stop = policy_network.decide(state)
        else:
            # Fixed threshold stopping
            should_stop = (
                verification['confidence'] > 0.85 and
                iteration > 0 and
                iteration_history[-1]['answer'] == iteration_history[-2]['answer']
            )
        
        if should_stop:
            break
    
    # Return best result
    return select_best_answer(iteration_history)

def select_best_answer(history):
    # Choose answer with highest verification confidence
    best = max(history, key=lambda x: x['confidence'])
    return {
        'answer': best['answer'],
        'confidence': best['confidence'],
        'iterations_used': len(history),
        'history': history
    }
```

#### Agent Implementations

```python
class PerceptionAgent:
    def __init__(self, model='gpt-4v'):
        self.model = model
        
    def extract(self, chart_image, critique_feedback=None):
        # Determine representation type
        chart_type = self.detect_chart_type(chart_image)
        repr_type = select_representation(chart_type)
        
        # Build prompt
        prompt = self.build_extraction_prompt(chart_type, repr_type, critique_feedback)
        
        # Call vision model
        response = call_vision_model(self.model, chart_image, prompt)
        
        # Parse into structured format
        representation = parse_representation(response, repr_type)
        
        return representation

class VerificationAgent:
    def verify(self, representation, original_image):
        if representation['type'] == 'code':
            return verify_code_representation(representation['content'], original_image)
        else:
            return verify_json_representation(representation['content'], original_image)

class ReasoningAgent:
    def __init__(self, model='gpt-4-turbo'):
        self.model = model
        
    def answer(self, representation, question):
        prompt = build_reasoning_prompt(representation, question)
        response = call_llm(self.model, prompt)
        
        return {
            'answer': extract_final_answer(response),
            'reasoning': extract_reasoning_steps(response),
            'full_response': response
        }

class CritiqueAgent:
    def __init__(self, model='gpt-4-turbo'):
        self.model = model
        
    def analyze(self, representation, verification, question, answer, reasoning):
        critique = generate_critique(representation, verification, question, answer, reasoning)
        return critique
```

#### Novelty Summary for Full Pipeline
- **Novel**: Complete integration of test-time RL + multi-agent + iterative refinement for chart QA
- **Novel**: Learned stopping policy based on visual verification confidence
- **Not Novel**: Individual components (extraction, reasoning, verification exist separately)
- **Key Contribution**: The full system working together for chart understanding

---

### 3.7 Baseline Methods (for Comparison)

To validate our approach, we implement these baselines:

#### Baseline 1: Direct VLM
```python
def baseline_direct_vlm(chart_image, question, model='gpt-4v'):
    prompt = f"Answer this question about the chart: {question}"
    response = call_vision_model(model, chart_image, prompt)
    return extract_final_answer(response)
```

#### Baseline 2: DePlot-style Pipeline
```python
def baseline_deplot(chart_image, question):
    # Step 1: One-shot extraction
    table = call_vision_model('gpt-4v', chart_image, 
                             "Extract this chart as a markdown table")
    
    # Step 2: LLM reasoning
    answer = call_llm('gpt-4-turbo', 
                     f"Given table:\n{table}\nQuestion: {question}")
    
    return extract_final_answer(answer)
```

#### Baseline 3: Self-Consistency (K=5)
```python
def baseline_self_consistency(chart_image, question, k=5, model='gpt-4v'):
    answers = []
    
    for i in range(k):
        prompt = f"Answer this question about the chart: {question}"
        response = call_vision_model(model, chart_image, prompt, temperature=0.7)
        answers.append(extract_final_answer(response))
    
    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

---

## 4. Novelty Verification Summary

### What is NOT Novel (Established Prior Work)

| Component | Prior Work | Citation |
|-----------|-----------|----------|
| Chart → Structured representation | DePlot, StructChart, MatCha | Liu+ 2023, Xia+ 2023 |
| Code verification via re-rendering | RECODE | arXiv 2510.13756 |
| Multi-agent for charts | ChartCitor, METAL | Goswami+ 2025, Li+ 2025 |
| Chain-of-thought reasoning | Standard practice | Wei+ 2022 |
| Self-consistency sampling | Established baseline | Wang+ 2023 |

### What IS Novel (Our Contributions)

| Component | Novelty Claim | Evidence of Gap |
|-----------|--------------|-----------------|
| **Test-time RL for chart QA** | First application | METAL does generation; TTRL does text; no work does chart QA |
| **Critique-guided refinement** | First for visual tasks | VISCO studies text critique; no visual critique agents exist |
| **Adaptive iteration policy** | First learned stopping for VLMs | Test-time scaling studied but not with RL policy |
| **Multi-method verification** | Novel combination | RECODE uses MSE only; we combine MSE + OCR + element detection |
| **Extraction vs. reasoning error separation** | First explicit distinction | Existing work doesn't distinguish failure modes |
| **Full integrated pipeline** | Novel system | No prior work combines all components for chart QA |

### Tools Summary

**Vision Models**: GPT-4V / Claude 3.5 Sonnet / Qwen2.5-VL-3B  
**Language Models**: GPT-4 Turbo / Claude 3.5 Sonnet / LLaMA 3.1 8B  
**OCR**: Tesseract (`pytesseract`) + EasyOCR  
**Image Comparison**: SSIM (`scikit-image`), MSE (`numpy`)  
**Object Detection**: OpenCV (`cv2`)  
**RL Framework**: PyTorch  
**Execution**: Python `exec()` with sandboxing  

### 2.2 Component Details

#### Component 1: Formal Representation Extraction

**Objective**: Convert chart images to structured formats enabling symbolic reasoning

**Representations**:
- **JSON**: Hierarchical structure for chart metadata, axes, legends, data points
  ```json
  {
    "chart_type": "bar",
    "x_axis": {"label": "Year", "values": [2020, 2021, 2022]},
    "y_axis": {"label": "Revenue ($M)", "range": [0, 150]},
    "data": [
      {"year": 2020, "value": 100, "category": "Product A"},
      {"year": 2021, "value": 125, "category": "Product A"}
    ]
  }
  ```

- **Code**: Executable Python/matplotlib that recreates the chart
  ```python
  import matplotlib.pyplot as plt
  years = [2020, 2021, 2022]
  revenue = [100, 125, 140]
  plt.bar(years, revenue)
  ```

- **Table**: Linearized CSV/markdown for sequential LLM processing
  ```
  Year | Revenue ($M)
  2020 | 100
  2021 | 125
  2022 | 140
  ```

**Adaptive Selection Logic**:
- Bar/pie charts → JSON (categorical data)
- Scatter/complex plots → Code (relationship visualization)
- Time series → Tables (sequential data)

**Innovation**: While chart-to-structured conversion exists (DePlot, StructChart), our adaptive representation selection based on chart characteristics is novel.

---

#### Component 2: Multi-Agent System

**Architecture**: Four specialized agents with distinct responsibilities

##### Agent 1: Perception Agent
- **Role**: Visual understanding and extraction
- **Input**: Chart image
- **Output**: Structured representation + uncertainty markers
- **Example**: "Value approximately 42 ± 2 (axis label partially occluded)"

##### Agent 2: Verification Agent  
- **Role**: Validate extraction accuracy
- **Method**: 
  - For code: Execute and compare rendered output to original (MSE)
  - For JSON/tables: Cross-check values against image (OCR + vision)
- **Output**: Confidence score (0-1) + error localization

##### Agent 3: Reasoning Agent
- **Role**: Answer questions using structured data
- **Input**: Structured representation + question
- **Output**: Answer + reasoning trace
- **Method**: Chain-of-thought over symbolic representation

##### Agent 4: Critique Agent
- **Role**: Generate targeted feedback for refinement
- **Analysis**:
  - If verification_score < 0.5 → "Re-extract data (suspected perception error)"
  - If verification_score > 0.8 AND answer incorrect → "Re-reason (extraction likely correct)"
- **Output**: Specific refinement instructions

**Novelty vs. Existing Work**:
- **ChartCitor** (2025): Has 6 agents but focuses on attribution, not iterative reasoning refinement
- **METAL** (2025): Multi-agent for chart generation, not understanding
- **Our contribution**: First critique-guided re-extraction AND re-reasoning loop for chart QA

---

#### Component 3: Iterative Refinement Loop

**Algorithm**:
```python
for iteration in range(N):  # N = 1 to 5
    # Extract
    representation = perception_agent(chart_image, critique_feedback)
    
    # Verify
    confidence, errors = verification_agent(representation, chart_image)
    
    # Reason
    answer, reasoning = reasoning_agent(representation, question)
    
    # Check stopping condition
    if confidence > threshold AND answer_consistent:
        return answer
    
    # Critique
    critique_feedback = critique_agent(
        representation, 
        confidence, 
        errors,
        answer,
        reasoning
    )
```

**Key Decisions**:
- **Stopping criteria**: High verification confidence + consistent answers across 2 rounds
- **Refinement target**: Critique agent decides whether to re-extract OR re-reason
- **Iteration limit**: Max 5 rounds to balance quality vs. cost

**Novelty**:
- **RECODE** (2025): Iterative refinement for code-based derendering only (MSE critic)
- **Our contribution**: Verification-guided refinement of BOTH extraction AND reasoning

---

#### Component 4: Test-Time Reinforcement Learning

**Objective**: Scale inference compute to improve accuracy

**Method**: Adaptive iteration based on question difficulty

**Reward Function**:
```
R = correctness * verification_confidence - λ * compute_cost

Where:
- correctness: 1 if answer matches ground truth (when available), else 0
- verification_confidence: [0, 1] from verification agent
- λ: Cost penalty weight (tunable parameter)
- compute_cost: Number of iterations * tokens used
```

**Policy**: Learn when to stop iterating vs. continue refining

**Training Signal**: 
- Use small validation set (100 samples) to learn iteration policy
- Policy network: Simple MLP (iteration_count, confidence, answer_entropy) → continue/stop

**Novelty**:
- **METAL** (2025): Studies test-time scaling for chart generation (not RL-based)
- **TTRL** (2024): Test-time RL for math/code, not vision-language
- **Our contribution**: First test-time RL application to chart QA

---

## 3. Novelty Analysis

### 3.1 Component-Level Novelty Assessment

| Component | Novelty Status | Prior Art | Our Contribution |
|-----------|---------------|-----------|------------------|
| **Structured Representations** | ❌ Not Novel | DePlot (2023), StructChart (2023), MatCha (2023) | Adaptive representation selection |
| **Multi-Agent Systems** | ⚠️ Partially Novel | ChartCitor (2025), METAL (2025) | Specialized perception/verification/reasoning/critique division |
| **Iterative Refinement** | ⚠️ Partially Novel | RECODE (2025) - code only | Refinement of data extraction AND reasoning |
| **Test-Time Scaling** | ✅ Novel | METAL (2025) - generation only | First for chart QA |
| **Test-Time RL** | ✅ Novel | TTRL (2024) - text only | First for vision-language charts |
| **Full Pipeline** | ✅ Novel | None | Complete integration |

### 3.2 Key Differentiators

#### vs. RECODE (2025)
- **RECODE**: Iterative refinement of code reconstruction using MSE critic (visual fidelity)
- **Ours**: Iterative refinement of reasoning correctness using verification + critique agents

#### vs. ChartCitor (2025)  
- **ChartCitor**: Multi-agent for attribution (6 agents: extraction, reformulation, augmentation, filtering, ranking, localization)
- **Ours**: Multi-agent for iterative reasoning refinement with critique-guided loops

#### vs. METAL (2025)
- **METAL**: Test-time scaling for chart generation (image → code rendering)
- **Ours**: Test-time RL for chart understanding (image + question → answer)

#### vs. DePlot Pipeline (2023)
- **DePlot**: Single-pass extraction → reasoning
- **Ours**: Multi-round extraction ↔ reasoning with verification feedback

### 3.3 Primary Novelty Claims

**Claim 1**: First application of test-time compute scaling to chart question answering
- **Evidence**: METAL only addresses generation; no QA benchmarks use test-time scaling

**Claim 2**: First verification-guided iterative refinement for both extraction AND reasoning
- **Evidence**: RECODE refines only code; ChartCitor has no iterative loops

**Claim 3**: First critique agent architecture for visual reasoning tasks  
- **Evidence**: Existing critique work (VISCO 2024) focuses on text; no visual critique agents exist

---

## 4. Expected Results

### 4.1 Performance Metrics

**Primary Metric**: Accuracy on ChartQA and EvoChart benchmarks

**Expected Improvements**:

| Method | ChartQA | EvoChart | Cost ($/sample) |
|--------|---------|----------|-----------------|
| GPT-4V Direct (baseline) | 78% | 52% | $0.02 |
| + Formal Representation | 83% (+5pp) | 57% (+5pp) | $0.04 |
| + Multi-Agent | 86% (+8pp) | 61% (+9pp) | $0.08 |
| + Test-Time RL (N=3) | **89% (+11pp)** | **65% (+13pp)** | $0.12 |

**Justification**:
- Formal representation: +5pp (established by DePlot)
- Multi-agent verification: +3pp (reduces perceptual errors)
- Iterative refinement (3 rounds): +3pp (fixes reasoning failures)

### 4.2 Ablation Studies

**RQ1**: Does formal representation improve reasoning?
- Compare: Direct VLM vs. Extract-then-Reason
- Expected: +5-7% (validated by prior work)

**RQ2**: Does multi-agent coordination help?
- Compare: Single model vs. Specialized agents
- Expected: +3-5% (verification reduces errors)

**RQ3**: Does iterative refinement add value?
- Compare: One-shot vs. N iterations
- Expected: Accuracy increases up to N=3, plateaus after

**RQ4**: What's the cost-accuracy tradeoff?
- Analyze: Iterations (1,2,3,5,10) vs. accuracy vs. cost
- Expected: "Diminishing returns after N=3"

### 4.3 Analysis Contributions

Beyond performance, we provide:

1. **Failure mode taxonomy**: When does extraction fail vs. reasoning fail?
2. **Compute scaling curves**: How does accuracy scale with iterations?
3. **Chart complexity analysis**: Which chart types benefit most from multi-agent?
4. **Practical guidelines**: When is test-time compute worth the cost?

---

## 5. Implementation Roadmap

### 5.1 Timeline (4 weeks)

#### Week 1: Infrastructure Setup
- **Days 1-2**: Implement perception agent (chart → JSON/code/table)
- **Days 3-4**: Implement verification agent (confidence scoring)
- **Days 5-7**: Implement reasoning + critique agents

#### Week 2: Integration & Testing
- **Days 8-9**: Build iterative refinement loop
- **Days 10-11**: Test on 100 ChartQA samples
- **Days 12-14**: Debug and optimize pipeline

#### Week 3: Experiments
- **Days 15-16**: Run full ChartQA experiments (baseline + ablations)
- **Days 17-18**: Run EvoChart experiments  
- **Days 19-21**: Cost-accuracy tradeoff analysis

#### Week 4: Analysis & Writing
- **Days 22-24**: Failure analysis and error categorization
- **Days 25-27**: Draft paper (4-6 pages)
- **Day 28**: Finalize and submit

### 5.2 Resource Requirements

**Compute**:
- **Option A**: GPT-4V API ($0.10/sample * 2000 samples = $200)
- **Option B**: Local 3B model (Qwen2.5-VL-3B) - free but slower

**Data**:
- ChartQA test set: 1,500 samples (publicly available)
- EvoChart: 600 samples (publicly available)
- Total: ~2,100 evaluation samples

**Storage**:
- Code + intermediate outputs: ~5GB
- No training data storage needed (inference-only)

### 5.3 Technical Stack

**Core Components**:
```python
# Perception Agent
- Vision model: GPT-4V / Qwen2.5-VL-3B
- OCR: Tesseract / Azure Computer Vision
- Output: JSON/Python code/CSV

# Verification Agent  
- Code execution: exec() in sandboxed environment
- Image comparison: MSE / SSIM metrics
- OCR cross-check: Compare extracted vs. actual values

# Reasoning Agent
- LLM: GPT-4 / local LLaMA-3.1
- Method: Chain-of-thought prompting on structured data

# Critique Agent
- LLM: Same as reasoning agent
- Method: Error analysis + targeted feedback generation
```

**Dependencies**:
- Python 3.10+
- OpenAI API / local transformers
- matplotlib (for code execution)
- PIL, cv2 (for image processing)
- pytest (for testing)

---

## 6. Evaluation Plan

### 6.1 Datasets

**Primary Benchmarks**:
1. **ChartQA** (1,500 test samples)
   - Chart types: Bar, line, pie, scatter
   - Question types: Retrieval, computation, comparison
   - Difficulty: Easy to hard

2. **EvoChart** (600 samples)
   - Out-of-distribution charts
   - Tests generalization

**Validation Set** (for iteration policy):
- 100 samples from ChartQA training set
- Used only to learn stopping criteria

### 6.2 Baselines

**Baseline 1**: Direct VLM (GPT-4V / Qwen2.5-VL)
- One-shot prompting: "Answer this question about the chart"

**Baseline 2**: DePlot-style Pipeline
- Chart → table (one-shot) → LLM reasoning

**Baseline 3**: Self-Consistency (K=5)
- Sample 5 answers, majority vote
- Established baseline for test-time scaling

**Our Method**:
- Full pipeline with N=3 iterations
- Ablations: -multiagent, -refinement, -testtime_RL

### 6.3 Metrics

**Primary**:
- **Accuracy**: Exact match (relaxed matching for numerical answers ±2%)

**Secondary**:
- **Extraction quality**: F1 score on extracted data values
- **Reasoning correctness**: % of correct intermediate steps
- **Consistency**: Agreement across multiple runs (κ coefficient)
- **Efficiency**: Average iterations needed per question
- **Cost**: Total API cost / total samples

**Analysis Metrics**:
- **By chart type**: Accuracy breakdown (bar, line, pie, scatter)
- **By question type**: Retrieval vs. computation vs. comparison
- **By iteration**: Accuracy at iteration 1, 2, 3, 4, 5

---

## 7. Expected Contributions

### 7.1 Technical Contributions

1. **First test-time RL framework for chart QA**
   - Demonstrates compute scaling works for vision-language tasks
   - Provides blueprint for other visual reasoning domains

2. **Multi-agent architecture for visual reasoning**
   - Specialized perception/verification/reasoning/critique division
   - Reusable design pattern for document understanding, diagram parsing, etc.

3. **Verification-guided iterative refinement**
   - Targeted re-extraction vs. re-reasoning based on error analysis
   - More efficient than random iterative refinement

### 7.2 Practical Contributions

1. **Zero-training deployment**
   - Works with any VLM (API or local)
   - No expensive fine-tuning required
   - Accessible to practitioners with limited resources

2. **Cost-accuracy tradeoffs**
   - Concrete guidance: "Use N=3 for production systems"
   - ROI analysis: "Extra $0.10/sample buys +11% accuracy"

3. **Failure mode taxonomy**
   - When does perception fail vs. reasoning fail?
   - Which chart types need more iterations?

### 7.3 Research Insights

1. **Compute scaling for VLMs**
   - First evidence that test-time compute helps vision-language tasks
   - Comparison to text-only scaling (o1/o3)

2. **Multi-agent benefits**
   - Quantify value of agent specialization
   - Design principles for visual reasoning agents

3. **Iteration diminishing returns**
   - Characterize accuracy saturation point
   - Optimal iteration budget for different difficulty levels

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

**Scope Limitations**:
- Focus on standard chart types (bar, line, pie, scatter)
- Does not handle: complex infographics, hand-drawn charts, 3D visualizations
- English-only datasets

**Technical Limitations**:
- Requires access to VLM (API cost or local compute)
- Iteration increases latency (3x slower than one-shot)
- No guarantee of convergence (may not find correct answer)

**Evaluation Limitations**:
- Tested only on ChartQA and EvoChart
- May not generalize to domain-specific charts (medical, financial)

### 8.2 Future Directions

**Short-term (3-6 months)**:
1. Extend to more chart types (heatmaps, network graphs, sankey diagrams)
2. Multi-lingual chart understanding (non-English axis labels)
3. Optimize iteration policy (learn adaptive stopping criteria)

**Medium-term (6-12 months)**:
1. Apply to related tasks (table QA, document understanding, diagram parsing)
2. Investigate multi-modal verification (audio descriptions, accessibility)
3. Develop benchmark for test-time compute on vision tasks

**Long-term (1+ years)**:
1. Unified framework for all visual reasoning tasks
2. Learned critique agents (train small models to generate feedback)
3. Interactive refinement (human-in-the-loop for critical applications)

---

## 9. Publication Strategy

### 9.1 Target Venues

**Primary Targets**:
1. **CVPR 2026** (Main conference)
   - Track: Vision & Language
   - Deadline: ~November 2025
   - Fit: Novel method with strong empirical results

2. **ICLR 2027** (Main conference)
   - Track: Test-Time Adaptation / Multi-Agent Systems
   - Deadline: ~October 2026
   - Fit: Inference-time learning emphasis

**Backup Targets**:
1. **CVPR 2026 Workshop**: Test-Time Adaptation
2. **NeurIPS 2026 Workshop**: Agent Systems
3. **ACL 2026 Findings**: Analysis track (if empirical results weaker than expected)

### 9.2 Positioning

**Main Message**:
"Test-time compute scaling works for vision-language tasks, enabling advanced reasoning without expensive fine-tuning"

**Narrative Arc**:
1. **Problem**: Chart reasoning is hard; training-based methods are expensive
2. **Insight**: Test-time compute helps text reasoning (o1/o3) - does it help vision?
3. **Method**: Multi-agent iterative refinement with verification feedback
4. **Results**: +11% accuracy at $0.10/sample with zero training
5. **Impact**: Democratizes advanced chart reasoning for resource-constrained practitioners

**Differentiation from Prior Work**:
- vs. METAL: We do QA, not generation
- vs. RECODE: We refine reasoning, not just code reconstruction  
- vs. ChartCitor: We have iterative loops, not one-shot attribution
- vs. Training methods: We require zero training data

---

## 10. Risk Mitigation

### 10.1 Technical Risks

**Risk 1**: Iteration doesn't improve accuracy
- **Mitigation**: Start with self-consistency (established to work)
- **Fallback**: Contribution becomes "analysis of when iteration helps"

**Risk 2**: Multi-agent overhead outweighs benefits
- **Mitigation**: Measure cost-accuracy tradeoffs explicitly
- **Fallback**: Recommend single-agent for cost-sensitive applications

**Risk 3**: Verification agent unreliable
- **Mitigation**: Use multiple verification methods (MSE + OCR + visual check)
- **Fallback**: User verification confidence as signal, not ground truth

### 10.2 Experimental Risks

**Risk 1**: Baselines stronger than expected
- **Mitigation**: Compare against self-consistency (K=5), not just one-shot
- **Fallback**: Focus on cost-efficiency rather than peak accuracy

**Risk 2**: Results don't generalize across datasets
- **Mitigation**: Test on both ChartQA and EvoChart from start
- **Fallback**: Analyze failure modes and provide domain-specific guidance

**Risk 3**: Resource constraints (API costs)
- **Mitigation**: Use local 3B model (Qwen2.5-VL) for development
- **Fallback**: Scale down experiment size (500 samples instead of 2000)

### 10.3 Publication Risks

**Risk 1**: Reviewers say "not novel enough"
- **Mitigation**: Emphasize test-time RL for chart QA (clear gap)
- **Fallback**: Submit to workshop or Findings track

**Risk 2**: Reviewers want stronger baselines
- **Mitigation**: Include Chart-RVR comparison (training-based SOTA)
- **Fallback**: Highlight zero-training advantage

**Risk 3**: Timeline slips
- **Mitigation**: 4-week aggressive plan, 6-week realistic plan
- **Fallback**: Target ICLR 2027 instead of CVPR 2026

---

## 11. Success Criteria

### 11.1 Minimum Viable Contribution

**Must achieve**:
1. ✅ Implement working pipeline (all 4 components)
2. ✅ Demonstrate accuracy improvement over baseline (any gain > 0%)
3. ✅ Show iteration helps (N=3 > N=1)
4. ✅ Measure cost-accuracy tradeoffs

**Publishable if**:
- Accuracy gain ≥ 5% on at least one benchmark
- Clear novelty claim (test-time RL for chart QA)
- Practical insights (when to use iteration)

### 11.2 Target Contribution

**Ideal outcome**:
1. ✅ +10% accuracy on ChartQA
2. ✅ +10% accuracy on EvoChart (generalization)
3. ✅ Competitive with Chart-RVR (no training needed)
4. ✅ Clear cost-accuracy Pareto frontier

**Top-tier venue if**:
- Match or exceed Chart-RVR performance
- Demonstrate test-time scaling law (accuracy ~ log(compute))
- Provide reusable multi-agent framework

### 11.3 Stretch Goals

**Exceptional outcome**:
1. ✅ Beat all training-based methods
2. ✅ Generalize to other visual reasoning tasks (document QA)
3. ✅ Learn optimal iteration policy (RL-based stopping)
4. ✅ Release benchmark + code for test-time compute on vision

---

## 12. Conclusion

This proposal presents a novel, feasible, and impactful research direction for chart reasoning. By combining test-time reinforcement learning with multi-agent systems, we address a clear gap in the literature while providing practical value to practitioners.

**Key Strengths**:
- ✅ **Novel**: First test-time RL for chart QA
- ✅ **Feasible**: 4-week implementation with available resources
- ✅ **Impactful**: Zero-training deployment democratizes advanced reasoning
- ✅ **Timely**: Aligns with current interest in test-time compute (o1/o3)

**Next Steps**:
1. Implement perception agent (Week 1)
2. Build iterative pipeline (Week 2)
3. Run experiments (Week 3)
4. Write paper (Week 4)

**Expected Outcome**: A strong empirical paper demonstrating that test-time compute scaling works for vision-language tasks, with practical guidance for production deployment.

---

**Document Version**: 1.0  
**Last Updated**: January 24, 2026  
**Status**: Ready for Implementation
