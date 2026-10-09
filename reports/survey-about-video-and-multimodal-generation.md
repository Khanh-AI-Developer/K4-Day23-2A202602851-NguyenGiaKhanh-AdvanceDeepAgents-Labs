# Video and Multimodal Generation: Architectures, Conditioning, Evaluation, and Responsible Deployment

## TL;DR

Video generation has moved from frame-level synthesis toward spatiotemporal foundation models that jointly optimize appearance, motion, long-range state, and semantic alignment. The dominant design pattern is latent diffusion or diffusion-transformer generation, often initialized from image models, while autoregressive token models, GANs, and flow matching remain important alternatives for temporal structure and efficient prediction [1][2][3][4][5]. The central bottleneck is no longer merely photorealistic frames: systems must preserve identity and geometry over time, follow compositional prompts, model physical events, and synchronize sound with visible causes.

Multimodal generation is consequently becoming modular and unified at once. Modular systems connect pretrained audio, video, language, pose, depth, or reference encoders through adapters and temporally aligned attention; unified systems use shared or coupled transformer streams and multimodal language-model planners [6][7][8][9][10]. The tradeoff is clear: factorization enables specialization and reuse, whereas joint modeling offers richer cross-modal correspondence but risks interference and synchronization drift.

Evaluation is evolving from single distributional scores to decomposed suites for fidelity, motion, temporal consistency, compositionality, physics, and audio-video alignment [11][12][13][14][15][16][17][18]. Results indicate persistent weaknesses in temporal transitions, physical commonsense, fine-grained synchronization, and benchmark trustworthiness. Deployment therefore requires not just quality improvements but provenance, access controls, safety testing, bias audits, copyright-aware data governance, and environmental accounting [19][20][21][22][23][24].

## Background

A useful abstraction is to represent a video as a sequence of spatial latents $z_{1:T}$ conditioned on text, images, audio, actions, or other signals. Generation must solve three coupled problems: (i) spatial synthesis at each time, (ii) temporal dynamics and persistence, and (iii) semantic grounding of the requested event. Multimodal generation adds a fourth: correspondence between streams whose sampling rates, uncertainty, and semantics differ.

Early caption-conditioned GANs imposed temporal structure directly through 3-D convolutions and separate video, frame, and motion discriminators [1]. Tokenized transformers instead compress visual sequences and model longer dependencies explicitly; TECO combines sequence compression, a temporal transformer, and spatial MaskGit to address revisitation and long-horizon consistency [2]. These approaches clarify a durable design tradeoff: adversarial one-pass generation is attractive for latency, while token prediction supplies explicit context but faces sequence-length and error-accumulation pressure.

Latent diffusion changed the scaling regime by moving denoising into a compact spatiotemporal representation. LVDM uses 3-D latents and hierarchical diffusion for long videos, while Video LDM adapts an image latent-diffusion prior by adding temporal dimensions and temporal alignment [3][4]. The benefit is reuse of strong image priors and lower spatial cost; the cost is iterative sampling and the difficulty of extending a local image prior into coherent motion. Flow matching offers a different transport formulation: RIVER uses VQGAN latents, sparse conditioning on past frames, and fewer ODE integration steps for efficient prediction [5].

## Architectural evolution and long-context generation

### From explicit temporal inductive bias to scalable attention

Architectures increasingly shift temporal reasoning from fixed convolutions toward attention and latent memory. StreamingT2V uses short- and long-term memory blocks plus blending to extend diffusion-generated segments [25]. This is a pragmatic compromise: generate locally, then maintain continuity through explicit state rather than attend over every token in the full history. Modern diffusion transformers generalize the same idea with coarse-to-fine generation, segment-wise synthesis, block sparsity, causal attention, and consistency distillation. Such techniques attack the quadratic cost of dense spatiotemporal attention, but reported speed or quality numbers are not directly comparable across resolutions, durations, hardware, or sampling schedules.

Long-context consistency remains a structural problem rather than a solved scaling property. A model may preserve local texture while changing an object when it leaves and re-enters the scene; repeated extension can accumulate identity, geometry, or motion errors [2][3]. The most promising remedies—hierarchical latents, memory, segment overlap, sparse attention, autoregressive diffusion, and distillation—trade context against detail. Future comparisons should report wall-clock latency, memory, number of denoising/ODE steps, duration, resolution, and temporal/semantic scores under a common protocol.

### Conditioning and controllability

Text is only one control channel. Image references preserve identity and composition; pose, depth, trajectories, and segmentation constrain geometry; audio specifies events, speech, or mood. Dreamix demonstrates the editing pattern of retaining low-resolution source spatiotemporal information while synthesizing high-resolution prompted content, with source-video fine-tuning improving fidelity [26]. DreamBooth provides a few-shot personalization primitive, but video personalization adds the harder requirements of persistence, multi-subject binding, and edit locality [27].

The design space spans lightweight adapters, ControlNet-like branches, feature injection, and full joint attention. Modular conditioning is easier to train from heterogeneous datasets and permits independent control strength. Unified representations can resolve conflicts globally but are more vulnerable to modality interference. This distinction should be evaluated explicitly rather than collapsed into a single “multimodal” label.

## Multimodal and unified generation

### Audio-video synchronization

Audio-video generation makes temporal alignment a first-class objective. AV-Link connects frozen audio and video diffusion models with fusion blocks, aligned rotary embeddings, and feature reinjection, supporting both video-to-audio and audio-to-video while retaining factorized expertise [6]. Ovi takes the opposite route with symmetric twin DiT streams and bidirectional cross-modal attention [7]. MTV decomposes audio into speech, effects, and music so that different tracks can control lip motion, event timing, and visual mood [8]. These approaches expose two distinct notions of synchronization: semantic correspondence (“a drum should sound like a drum”) and event timing (“the sound should begin when the visible impact occurs”).

Joint denoising is not automatically better. Noisy streams can drift apart, global attention can be inefficient for fine temporal cues, and classifier-free guidance may strengthen each modality independently without improving cross-modal agreement. As a result, temporal alignment, local attention, asymmetric interaction, demixed audio controls, and modality-aware guidance are recurring mechanisms. Benchmarks should report semantic and temporal alignment separately [18].

### Unified understanding, editing, and world models

VideoPoet frames multimodal generation as a decoder-only transformer over images, videos, text, and audio [10]. Omni-Video instead couples a multimodal language model to a diffusion decoder through continuous video tokens, supporting understanding, generation, and instruction-based editing [9]. These systems point toward a unified interface in which the model can interpret a scene, plan an edit, and render the result, but they do not eliminate the need for specialized temporal decoders.

World models impose a stronger criterion than prompt-to-video realism: actions should causally affect future observations, and rollouts should remain useful over long horizons. AVID adapts pretrained video diffusion through action-conditioned adapters and masked combinations of pretrained and action-specific noise predictions [28]. Cosmos-style systems unify text-, image-, and video-to-world tasks and add physical-AI grounding [29]. The research challenge is to evaluate causal fidelity, interactivity, state persistence, and sim-to-real utility rather than only visual similarity.

## Data, objectives, and evaluation

Web-scale video-text data provide breadth but are noisy, unstable, and difficult to audit. HD-VILA-100M contains 103M 720p clip-caption pairs, while WebVid and Panda-70M illustrate the scale and source-link fragility of scraped corpora [30][31]. A representative recipe combines video-only pretraining with paired video-text denoising fine-tuning [32]. The resulting scale improves coverage but can also import caption errors, demographic imbalance, copyrighted material, duplicates, and benchmark contamination.

Evaluation should be multidimensional. EvalCrafter combines 700 prompts, objective metrics, and human studies across visual quality, text-video alignment, motion, and temporal consistency [11]. VBench decomposes generation into dimensions such as subject/background consistency, flicker, motion smoothness, dynamic degree, action, spatial relations, and style [12]. T2V-CompBench targets attribute binding, spatial relations, actions, interactions, and numeracy, demonstrating why FID, FVD, and average CLIP similarity do not adequately capture compositionality [13]. TC-Bench further tests state transitions and reports that contemporary systems complete fewer than one-fifth of its compositional-change cases in the cited experiments [14].

Physical plausibility is a separate frontier. VideoPhy evaluates material interactions and reports a best joint semantic/physical result of 39.6% in its setting [15]. Physics-IQ likewise finds that visual realism does not imply physical understanding, with the best model scoring 29.5% of its normalized physical-variance baseline [16]. These results argue for causal and counterfactual tests of conservation, object permanence, contact, and timing—not only image quality.

For audio-video generation, T2AV-Compass combines video, audio, and cross-modal metrics with judge-based instruction following and realism, including CLAP, ImageBind, synchronization, and lip-sync measures [18]. The field should report onset offsets, semantic attribution, acoustic fidelity, lip synchronization, and long-horizon drift separately. Automated judges improve coverage but can hallucinate observations or be manipulated by frame-selection artifacts; blinded human panels, uncertainty, adversarial tests, and inter-rater reliability remain necessary [17].

## Systems, safety, and governance

Scaling occurs in both training and inference. Video-T1 illustrates test-time scaling through adaptive sampling and feedback [33]. Yet quality gains must be weighed against compute, latency, and energy. A Carbon Trust/DIMPACT analysis estimates that short video generation can require roughly two orders of magnitude more energy than a simple text query, while emphasizing that system boundaries and measurement practices vary [23]. Public reporting should include training attribution, retries, editing workflows, hardware, energy mix, and embodied emissions.

Safety is necessarily layered because harmful outcomes can arise from benign text-video combinations and from context rather than content alone. Veo describes filtering, red teaming, model cards, product review, monitoring, and SynthID watermarking [21]. Sora 2 describes controlled rollout, multimodal scanning, reporting, restrictions on real-person generation, C2PA metadata, and visible watermarks, while acknowledging that contextual deception can evade classifiers [22]. NIST recommends documenting data provenance, testing, incidents, bias, intellectual property, and environmental impacts [20]. No watermark or classifier is a complete authenticity guarantee; robustness after cropping, transcoding, screen capture, editing, and adversarial removal must be measured.

Copyright and representation require parallel attention. Training-data provenance affects legal and technical accountability, especially when open weights are fine-tuned downstream; the U.S. Copyright Office emphasizes that the broad label “training” can obscure which data were used, by whom, and for what purpose [24]. Official safety evaluation also reports demographic skew in unspecified prompts [21]. Audits should therefore cover intersectional identity, language, culture, disability, voice, likeness, and temporal stereotypes, not only aggregate image quality.

## Trends and open problems

1. **The field is converging on latent DiT ecosystems, but hybrids matter.** Diffusion transformers offer flexible conditioning and scalable attention; autoregressive diffusion, flow matching, memory, and distillation address latency and long horizons. No family dominates under a common, hardware-controlled comparison.
2. **Temporal intelligence is the key differentiator.** Identity persistence, state revisitation, event timing, camera-motion disentanglement, and physical interactions remain harder than local visual fidelity.
3. **Multimodal control is moving from concatenation to structured interaction.** Aligned rotary embeddings, modality-specific adapters, local cross-attention, semantic planners, and demixed audio make conditioning more interpretable, but composition can still produce interference.
4. **Benchmarks are becoming diagnostic rather than scalar.** Quality, motion, dynamics, composition, physics, semantic alignment, and synchronization should be reported as a profile. Contamination audits, private refresh sets, source-level splits, and training-data disclosure are urgently needed.
5. **World models require causal evaluation.** Action-conditioned rollout, intervention tests, long-horizon state tracking, and sim-to-real transfer should be separated from prompt-following video synthesis.
6. **Responsible deployment is an engineering property.** Access control, red teaming, provenance, watermark robustness, incident response, bias measurement, copyright-compatible datasets, and energy accounting must be evaluated alongside model quality.
7. **Open research questions remain fundamental.** How should models allocate tokens or latent capacity across motion and appearance? Can audio, video, and language share a world state without destructive interference? How can judges be calibrated against humans? What training and inference scale is sufficient for physical reasoning? And how can high-fidelity generation be made auditable, affordable, and environmentally defensible?

## References
[1] To Create What You Tell: Generating Videos from Captions. arxiv. https://arxiv.org/abs/1804.08264 (2018-04-23)
[2] Temporally Consistent Transformers for Video Generation. arxiv. https://arxiv.org/abs/2210.02396 (2022-10-05)
[3] Latent Video Diffusion Models for High-Fidelity Long Video Generation. arxiv. https://arxiv.org/abs/2211.13221 (2022-11-23)
[4] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. arxiv. https://arxiv.org/abs/2304.08818 (2023-04-18)
[5] Efficient Video Prediction via Sparsely Conditioned Flow Matching. arxiv. https://arxiv.org/abs/2211.14575 (2022-11-26)
[6] AV-Link: Temporally-Aligned Diffusion Features for Cross-Modal Audio-Video Generation. arxiv. https://arxiv.org/abs/2412.15191 (2025-03-10)
[7] Ovi: Twin backbone cross-modal fusion for audio-video generation. arxiv. https://arxiv.org/abs/2510.01284 (2025-09-30)
[8] Audio-Sync Video Generation with Multi-Stream Temporal Control. arxiv. https://arxiv.org/abs/2506.08003 (2025-06-09)
[9] Omni-Video: Democratizing Unified Video Understanding and Generation. hf-daily. https://huggingface.co/papers/2507.06119 (2025-07-08)
[10] VideoPoet: A Large Language Model for Zero-Shot Video Generation. hf-daily. https://huggingface.co/papers/2312.14125 (2023-12-21)
[11] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. web. https://arxiv.org/abs/2310.11440v3 (2024-03-23)
[12] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf (2024-06-17)
[13] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.pdf (2025-06-17)
[14] TC-Bench: Benchmarking Temporal Compositionality in Conditional Video Generation. web. https://aclanthology.org/2025.findings-acl.241.pdf (2025-03-27)
[15] Evaluating Physical Commonsense for Video Generation. arxiv. https://arxiv.org/abs/2406.03520 (2024-06-05)
[16] Do Generative Video Models Understand Physical Principles?. arxiv. https://arxiv.org/abs/2501.09038 (2025-02-27)
[17] A Survey of AI-Generated Video Evaluation. web. https://arxiv.org/abs/2410.19884v2 (2024-10-25)
[18] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation. arxiv. https://arxiv.org/abs/2512.21094 (2025-12-26)
[19] Movie Gen: A Cast of Media Foundation Models. hf-search. https://huggingface.co/papers/2410.13720 (2024-10-17)
[20] Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile. web. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf (N/A)
[21] Veo: a text-to-video generation system. web. https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf (N/A)
[22] Sora 2 System Card. web. https://cdn.openai.com/pdf/50d5973c-c4ff-4c2d-986f-c72b5d0ff069/sora_2_system_card.pdf (N/A)
[23] The carbon impact of AI video generation. web. https://www.carbontrust.com/sites/default/files/documents/resource/public/The%20carbon%20impact%20of%20AI%20video%20generation_The%20Carbon%20Trust%20and%20DIMPACT.pdf (N/A)
[24] Part 3: Generative AI Training pre-publication version. web. https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf (N/A)
[25] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text. hf-search. https://huggingface.co/papers/2403.14773 (2024-03-21)
[26] Dreamix: Video Diffusion Models are General Video Editors. arxiv. https://arxiv.org/abs/2302.01329 (2023-02-02)
[27] DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. arxiv. https://arxiv.org/abs/2208.12242 (2023-03-15)
[28] Adapting Video Diffusion Models to World Models. arxiv. https://arxiv.org/abs/2410.12822 (2024-11-24)
[29] World Simulation with Video Foundation Models for Physical AI. hf-daily. https://huggingface.co/papers/2511.00062 (2025-10-28)
[30] Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions. arxiv. https://arxiv.org/abs/2111.10337v2 (2021-11-22)
[31] From Sora What We Can See: A Survey of Text-to-Video Generation. arxiv. https://arxiv.org/abs/2405.10674v1 (2024-05-17)
[32] MagicVideo: Efficient Video Generation With Latent Diffusion Models. arxiv. https://arxiv.org/abs/2211.11018 (2023-05-11)
[33] Video-T1: Test-Time Scaling for Video Generation. hf-search. https://huggingface.co/papers/2503.18942 (2025-03-24)
