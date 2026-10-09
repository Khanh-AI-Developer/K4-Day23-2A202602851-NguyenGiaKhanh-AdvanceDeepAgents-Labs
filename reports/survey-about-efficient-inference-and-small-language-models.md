# Efficient Inference and Small Language Models: A Research Survey

## TL;DR

Efficient inference is not a single optimization but a stack that spans model design, training data, compression, decoding algorithms, memory management, kernels, scheduling, and evaluation methodology. Small language models (SLMs) improve deployment by reducing parameter memory, compute, and often latency, but their usefulness depends on the workload: an on-device assistant, a cloud speculative-decoding draft model, a task-specific distilled model, and a high-throughput server all optimize different objectives. The strongest pattern across the literature is that parameter count alone is a poor proxy for efficiency. Compact models such as MobileLLM, TinyLlama, and Phi-style models show that architecture, token budget, and data quality can shift the capability/latency frontier [1], [2], [3], [4]. Systems papers show that even a fixed model can behave very differently under quantization, paged KV-cache allocation, FlashAttention-style kernels, continuous batching, and chunked prefill [5], [6], [7], [8]. Evaluation must therefore report quality together with TTFT, inter-token latency/TPOT, throughput, peak memory, energy, workload shape, hardware, and software stack [9], [10], [11], [12].

## Background

Modern autoregressive language-model inference has two distinct phases. In **prefill**, the model processes the input prompt and builds key-value (KV) cache state. In **decode**, it generates one or a few new tokens per iteration while repeatedly reading the cache. SLMs reduce the base cost of both phases, but long prompts, large batches, and long outputs can make memory bandwidth, KV-cache capacity, and scheduler policy as important as raw FLOPs. This is why efficient inference research includes both model-level methods—smaller architectures, grouped-query attention, distillation, and data-centric training—and runtime methods such as quantization, paged memory, IO-aware attention kernels, and request scheduling.

The term **small language model** is used inconsistently, but recent SLM work commonly refers to decoder-only transformer models from roughly hundreds of millions to a few billion parameters. A Hugging Face-indexed SLM survey explicitly studies 100M--5B decoder-only transformers and measures inference latency and memory alongside reasoning, in-context learning, mathematics, and coding capability [13]. This scope is useful because deployment constraints are rarely solved by reducing parameters alone: a 3B model may fit on a device but still fail an interactive latency target, while a 1B model may be fast but insufficiently accurate for the task.

## Model-side routes to efficient SLMs

### Architecture matters more at small scale than simple downscaling suggests

MobileLLM argues that, for sub-billion-parameter on-device models, architecture is a central efficiency lever rather than a secondary detail. It reports that deep-and-thin model shapes, embedding sharing, grouped-query attention (GQA), and related design choices improve compact-model accuracy, and its weight-sharing variant increases accuracy without increasing model size, at the cost of marginal latency overhead [1]. This illustrates a common SLM tradeoff: architectural changes can reduce memory or improve accuracy per parameter, but may introduce less regular compute graphs or additional latency.

GQA is a recurring design choice because it reduces the number of KV heads used at inference while trying to preserve much of multi-head attention's quality. The GQA paper record summarizes the technique as speeding decoder inference while maintaining quality relative to multi-head attention [14]. In practice, the benefit depends on implementation and workload: GQA can reduce KV-cache size and memory bandwidth, but only if kernels and serving stacks exploit the reduced head structure.

### Training tokens and data quality can substitute for raw parameter scale

TinyLlama demonstrates a different strategy: keep a standard LLaMA-style architecture but train a compact 1.1B model on very large token budgets using efficient implementations such as FlashAttention and Lit-GPT [2]. Its framing distinguishes compute-optimal training from inference-optimal deployment: a smaller model can be trained on more tokens than classical scaling-law recommendations if the goal is a better model at a fixed inference budget [2].

Phi-style models emphasize data quality even more strongly. Phi-1.5 uses synthetic, textbook-quality data and reports that a 1.3B model can reach natural-language task performance comparable to substantially larger models in the authors' evaluations [3]. Phi-3-mini scales this recipe with heavily filtered public web data, synthetic data, and alignment; the technical report presents a 3.8B model trained on 3.3T tokens and describes it as small enough for phone deployment while reporting competitive benchmark results [4]. The general lesson is not that synthetic data universally replaces scale, but that careful data curation and training mixture design are central to the SLM efficiency frontier.

### Distillation compresses behavior, but objective design is decisive

Distillation remains one of the clearest mechanisms for producing task-efficient SLMs. Distilling Step-by-Step uses teacher-generated rationales as additional supervision and reports that smaller students can outperform larger few-shot models on some benchmark settings while using less task data [15]. MiniLLM focuses on generative language-model distillation and argues that reverse-KLD/on-policy optimization avoids some failure modes of forward-KLD distillation, such as overestimating low-probability teacher outputs [16]. These papers show that distillation is not merely model-size reduction; it is an objective-design problem involving teacher access, student capacity, response style, calibration, and exposure bias.

For deployment, the choice between pretraining-from-scratch and distillation is workload-specific. Data-centric pretrained SLMs may generalize better across tasks, while distilled or task-specific SLMs can be much more efficient for narrow workflows. The cost is that distilled students inherit teacher biases and may degrade outside the teacher-generated distribution.

## Systems techniques for efficient inference

### Quantization reduces bandwidth and memory, but hardware and outliers dominate outcomes

Quantization is often the first inference optimization because it can shrink weights and accelerate matrix operations. SmoothQuant is a training-free post-training W8A8 method that migrates activation outlier difficulty into weights via per-channel scaling, reporting speed and memory gains with small accuracy loss in its evaluated settings [5]. The key technical insight is that LLM activations can contain difficult outliers, so naive activation quantization is often worse than weight-only quantization unless scaling, calibration, and hardware kernels are co-designed.

KV-cache quantization addresses a different bottleneck: runtime memory that grows with batch size and sequence length. KIVI proposes asymmetric 2-bit KV-cache quantization, using different granularity for keys and values and a small full-precision residual window; it reports reduced peak memory, larger batch size, and throughput gains in its experiments [17]. Unlike weight quantization, KV quantization adds per-token or per-channel metadata and dequantization overhead, and quality must be tested on long-context and reasoning tasks.

### KV-cache allocation, eviction, and reuse are central to serving

PagedAttention reframes KV cache as a virtual-memory-like allocation problem: sequences are split into fixed-size logical blocks mapped to non-contiguous physical blocks, enabling on-demand allocation, lower fragmentation, and sharing via copy-on-write [6]. This is approximately lossless compared with eviction because it changes the memory layout rather than the model's attended content. The tradeoff is systems complexity: kernels must follow block tables and handle non-contiguous cache layouts.

Lossy KV reduction methods make stronger assumptions. H2O observes that a subset of heavy-hitter tokens contributes disproportionately to attention and retains a combination of accumulated-attention heavy hitters and recent tokens [18]. Such policies can substantially reduce memory, but they change the effective context and can fail when a task requires tokens that the heuristic evicts. Attention-sink work similarly cautions that initial tokens may stabilize streaming generation even when they appear semantically unimportant, showing that simple recency windows are not sufficient for long-running streams [19].

### IO-aware attention kernels shift the bottleneck

FlashAttention computes exact attention using tiling and fused operations to reduce high-bandwidth-memory reads/writes and avoid materializing the full attention matrix [7]. FlashAttention-2 improves parallelism and work partitioning and reports substantially better GPU utilization in its benchmark settings [20]. These methods are important because they preserve attention semantics while changing the implementation. However, their benefits depend on sequence length, precision, GPU generation, batch shape, and compatibility with paged or non-contiguous KV cache. They are best viewed as kernel-level building blocks rather than universal speedup constants.

### Scheduling: throughput, TTFT, and fairness are coupled

Autoregressive serving mixes long prefill computations with short decode steps. Sarathi identifies prefill/decode interference and proposes chunked prefills plus decode-maximal batching to make iteration compute more uniform, reporting throughput gains over its evaluated baselines [8]. Modern serving engines such as vLLM document continuous batching, paged attention, prefix caching, chunked prefill, CUDA/HIP graphs, and quantization support, while TensorRT-LLM documents NVIDIA-focused engine building, custom kernels, in-flight batching, and paged KV-cache features [21], [22].

The implication for SLMs is subtle. A smaller model may reduce per-token compute, but under server load the scheduler still controls queueing, tail latency, memory pressure, and batching efficiency. Conversely, a larger quantized model with an optimized scheduler can outperform a smaller model in throughput per dollar for some workloads. Thus, inference efficiency should be measured at the full serving-system level, not only at isolated model-forward latency.

## Inference-time algorithms and the role of SLMs

### Speculative decoding makes SLMs useful as draft models

Speculative decoding is the most direct algorithmic connection between SLMs and efficient inference of larger models. A small or otherwise faster draft model proposes multiple tokens; the target model scores them in parallel; a corrected acceptance/rejection procedure preserves the target distribution under the algorithm's assumptions [23]. Related speculative-sampling work similarly uses a draft model and modified rejection sampling to recover the target distribution within numerical assumptions while reporting acceleration on large-model setups [23].

The critical caveat is that draft-model benchmark quality is not the same as deployment speedup. Decoding Speculative Decoding reports that speedup depends strongly on draft latency and only weakly on general language-modeling capability, and that hardware-efficient draft designs can outperform stronger but slower drafts [24]. This reframes SLM selection: the best draft is not necessarily the best standalone small model, but the one with the best acceptance-rate/latency/memory profile when paired with a target and scheduler.

### Prompt compression, prepacking, and context management trade exactness for savings

Prompt compression reduces prefill cost and API-visible token count by removing or summarizing input content. LLMLingua reports large compression ratios with little performance loss across several tasks, but it is explicitly approximate rather than distribution-preserving [25]. Prepacking attacks a different source of waste: padding in batched prefill. It bin-packs variable-length prompts and adjusts attention masks and positions so that real tokens can be processed more compactly without deleting content [26].

For long contexts, cache and context-management methods become increasingly important. H2O and attention-sink methods show that token importance is structured but nontrivial [18], [19]. More recent Hugging Face daily papers indicate continued exploration of reasoning-trace compression and token-importance pitfalls: LightThinker compresses verbose intermediate reasoning into compact representations, while LLM-Microscope warns that punctuation, stopwords, and other apparently low-information tokens can carry contextual information whose removal degrades benchmarks [27], [28]. The practical lesson is that context reduction should be evaluated task-by-task; semantic-looking heuristics are not reliable safety guarantees.

## Evaluation methodology for efficient SLM inference

### Report phase-specific latency, throughput, memory, quality, and workload

A rigorous survey of efficient SLM inference should not rank systems by tokens per second alone. NVIDIA's NIM benchmarking documentation separates TTFT, end-to-end latency, inter-token latency/TPOT, total output-token throughput, and per-user throughput under concurrency [11]. Etalon argues that conventional TTFT, time-between-tokens, normalized latency, and TPOT do not fully capture streaming user experience, proposing fluidity-oriented metrics that account for token timing and deadline misses [9]. These sources support reporting at least p50/p95/p99 TTFT, TPOT or ITL, end-to-end latency, output throughput, and streaming jitter or deadline violations.

Memory should be decomposed into model weights, activations, KV cache, allocator overhead, fragmentation, and runtime/workspace memory. This is especially important for SLMs because a model that fits at batch size one may fail under long-context or multi-user settings. The SLM survey's pairing of latency and memory footprint with task capabilities is therefore methodologically important [13].

### Use benchmark scenarios, not only microbenchmarks

MLPerf Inference provides a useful benchmark-design template with scenario-specific latency and quality requirements and distinct deployment scenarios such as single-stream, server, offline, and multistream; the accompanying rules define the complete hardware/software system under test [10]. Even if MLPerf is not an SLM-specific benchmark, its principles transfer directly. An edge assistant should be judged by single-user responsiveness and device memory; a batch summarization service by offline throughput and cost; a chatbot service by server arrivals, tail latency, and quality under load.

Energy and cost should be measured under explicit boundaries. ML.ENERGY is described as an inference energy benchmark across architectures and tasks, and the Watt Counts record emphasizes hardware-aware deployment and different energy outcomes in batch versus server scenarios [12]. The important methodological claim is conservative: joules/request or joules/token is only comparable when the hardware, software, telemetry boundary, workload, and quality gate are specified.

## Trends and open problems

First, SLM research is moving from “smaller is cheaper” toward **inference-optimal model design**: architectures, token budgets, and datasets are selected for deployment constraints rather than pretraining compute alone [1], [2], [4]. Second, systems research is converging on **memory-centric inference**. KV cache, allocator fragmentation, attention IO, and batching often dominate real serving performance, especially for long contexts and high concurrency [6], [17], [7], [8]. Third, SLMs increasingly appear as **components inside larger inference algorithms**, especially draft models for speculative decoding, compressors, routers, or task specialists rather than full replacements for frontier models [23], [24], [25].

The open problems are correspondingly cross-layer. We need better methods for jointly optimizing draft model quality, latency, and scheduler interaction; better quality guarantees for prompt and KV-cache compression; portable kernels and quantization methods across non-NVIDIA and edge hardware; and standardized evaluation traces covering prompt-length distributions, output lengths, arrival processes, streaming user experience, and energy measurement. Most importantly, the field needs quality-constrained efficiency reporting: a result should state not only that a system is faster or smaller, but under what task quality, latency SLO, memory budget, hardware stack, and workload distribution the claim holds.

## References
[1] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. arxiv. https://arxiv.org/abs/2402.14905 (2024-02-22)
[2] TinyLlama: An Open-Source Small Language Model. arxiv. https://arxiv.org/abs/2401.02385 (2024-06-04)
[3] Textbooks Are All You Need II: phi-1.5 technical report. arxiv. https://arxiv.org/abs/2309.05463 (2023-09-11)
[4] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arxiv. https://arxiv.org/abs/2404.14219 (2024-04-22)
[5] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arxiv. https://arxiv.org/abs/2211.10438 (2022-11-18)
[6] Efficient Memory Management for Large Language Model Serving with PagedAttention. arxiv. https://arxiv.org/abs/2309.06180 (2023-09-12)
[7] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arxiv. https://arxiv.org/abs/2205.14135 (2022-06-23)
[8] Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. arxiv. https://arxiv.org/abs/2308.16369 (2023-08-31)
[9] Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems. hf-search. https://huggingface.co/papers/2407.07000 (2024-07-09)
[10] MLPerf Inference Benchmarks. web. https://docs.mlcommons.org/inference/ (N/A)
[11] Metrics — NVIDIA NIM LLMs Benchmarking. web. https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html (N/A)
[12] The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization. hf-search. https://huggingface.co/papers/2505.06371 (2025-05-09)
[13] Small Language Models: Survey, Measurements, and Insights. hf-search. https://huggingface.co/papers/2409.15790 (2024-09-24)
[14] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. hf-search. https://huggingface.co/papers/2305.13245 (2023-05-22)
[15] Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. arxiv. https://arxiv.org/abs/2305.02301 (2023-07-05)
[16] MiniLLM: On-Policy Distillation of Large Language Models. arxiv. https://arxiv.org/abs/2306.08543 (2023-06-14)
[17] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arxiv. https://arxiv.org/abs/2402.02750 (2024-02-05)
[18] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. arxiv. https://arxiv.org/abs/2306.14048 (2023-12-18)
[19] Efficient Streaming Language Models with Attention Sinks. arxiv. https://arxiv.org/abs/2309.17453 (2023-09-29)
[20] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arxiv. https://arxiv.org/abs/2307.08691 (2023-07-17)
[21] vLLM Documentation. web. https://docs.vllm.ai/ (N/A)
[22] NVIDIA TensorRT-LLM Documentation. web. https://docs.nvidia.com/tensorrt-llm/index.html (N/A)
[23] Fast Inference from Transformers via Speculative Decoding. arxiv. https://arxiv.org/abs/2211.17192 (2023-05-18)
[24] Decoding Speculative Decoding. arxiv. https://arxiv.org/abs/2402.01528 (2024-02-02)
[25] LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. arxiv. https://arxiv.org/abs/2310.05736 (2023-12-06)
[26] Prepacking: A Simple Method for Fast Prefilling and Increased Throughput in Large Language Models. arxiv. https://arxiv.org/abs/2404.09529 (2024-04-15)
[27] LightThinker: Thinking Step-by-Step Compression. hf-daily. https://huggingface.co/papers/2502.15589 (2025-02-21)
[28] LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers. hf-daily. https://huggingface.co/papers/2502.15007 (2025-02-20)
