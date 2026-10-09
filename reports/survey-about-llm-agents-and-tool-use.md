# LLM Agents and Tool Use: Architectures, Evaluation, Safety, and Emerging Systems

## TL;DR

LLM agents turn language models from passive text generators into systems that choose actions, call tools, observe results, and revise plans. The field has converged on several reusable patterns: interleaved reasoning/action loops, learned API-call tokens, modular routers, search or symbolic planners, reflection with memory, and multi-agent workflows. These patterns improve capability, interpretability, or scalability in different ways, but none eliminates the core difficulty: tool use makes model errors consequential because calls can retrieve private data, mutate external state, spend money, send messages, or execute code. Evaluation is moving from static question answering toward executable and state-based benchmarks such as browser, API, code-repair, safety, and computer-use environments. The best evidence suggests that current agents remain far below human reliability in long-horizon web/computer tasks and are vulnerable to prompt injection, unsafe actions, and brittle self-evaluation. A realistic research agenda therefore combines better agent architectures with least-privilege tooling, outcome-based evaluation, sandboxing, auditability, and calibrated human oversight.

## Background

A tool-using LLM agent couples an LLM with an action interface. Instead of only producing an answer, the model may select an API, search the web, run code, manipulate a browser, retrieve memory, call a planner, or coordinate with other agents. This shift changes the problem from language modeling alone to closed-loop decision making: the system must parse an instruction, infer latent goals and constraints, select permitted tools, construct valid arguments, interpret observations, recover from errors, and decide when the task is complete.

Early and influential work framed this as an interleaving problem. ReAct asks the model to alternate reasoning traces with task-specific actions and observations, so the same trajectory contains a partial plan, evidence gathering, and exception handling [1]. Toolformer instead treats tool use as a learnable token-level behavior: candidate API calls are inserted into text, filtered by whether returned results improve future-token prediction, and then used to fine-tune the model to decide when and how to call tools [2]. Modular approaches such as MRKL decompose the system into a router plus specialist neural or symbolic modules, emphasizing explicit dispatch to calculators, databases, or other experts [3]. Planning approaches place more structure around the model: Tree of Thoughts searches over candidate reasoning states with lookahead and backtracking [4], while LLM+P translates natural-language planning problems into PDDL, invokes a classical planner, and translates the plan back to language [5].

These designs share a common motivation: LLMs have broad linguistic priors but limited intrinsic reliability, state, arithmetic precision, and access to up-to-date or private information. Tools can compensate, but they introduce interface errors, environmental uncertainty, and new threat models.

## Architectural patterns for tool-using agents

### Interleaved reasoning and action

The simplest and still central architecture is a single-agent loop: think, act, observe, and repeat. ReAct made this pattern explicit by prompting models to produce reasoning traces interleaved with actions, allowing an agent to gather external information and update plans as observations arrive [1]. In reported experiments, ReAct was evaluated on question-answering and interactive environments including HotpotQA, Fever, ALFWorld, and WebShop; the paper reports large absolute success-rate gains over imitation and reinforcement-learning baselines on ALFWorld and WebShop with only one or two in-context examples [1].

The strength of this pattern is flexibility. A single model can use context, observations, and scratchpad reasoning without a separate planner or learned controller. The weakness is that control is only as reliable as the model's next-token policy and prompt. ReAct's own limitations include harder scaling to complex tasks with large action spaces because more demonstrations may exceed context limits [1]. In practice, such loops also need parsers, retries, tool schemas, and termination conditions, none of which are solved by the prompting pattern alone.

### Learned tool invocation and API supervision

Toolformer moves some control from inference-time prompting into training. It serializes API calls and results as text, interrupts generation when an API-call marker appears, executes the external API, inserts the result, and continues generation [2]. Its self-supervised pipeline uses a few demonstrations per API to sample calls, filters them by usefulness, and fine-tunes the model on retained call/result traces [2]. This reduces dependence on hand-written prompts, but it assumes that useful calls can be discovered and filtered from text corpora and that the deployment tool interface remains compatible with learned representations.

ToolLLM/ToolBench scales supervision around real API documentation: the notes report 16,464 APIs across 49 categories, filtered into 3,451 tools, with 126,486 instances and 469,585 real API calls, plus a neural API retriever and reasoning/tool-execution traces [6]. This broadens coverage from a handful of APIs to realistic tool ecosystems. The tradeoff is benchmark and data validity: generated instructions and solution paths may inherit teacher-model bias, and real APIs may change, fail, or disappear [6]. API-Bank offers a smaller but runnable evaluation setting with 73 API tools and annotated dialogues, targeting invocation and use failures in tool-augmented LLMs [7].

### Search, symbolic planning, and modular routing

Search-based and symbolic designs impose more structure than free-form loops. Tree of Thoughts generalizes chain-of-thought prompting into a search over coherent text units, supporting self-evaluation, lookahead, and backtracking; the Hugging Face summary reports a large Game of 24 improvement for GPT-4 with Tree of Thoughts over chain-of-thought [4]. The price is computational: more branches require more model calls and a reliable evaluator for partial thoughts.

LLM+P separates linguistic interpretation from planning by translating a task into PDDL, using a classical planner, and rendering the resulting plan back into language [5]. This can provide stronger planning behavior when a correct symbolic domain is available, but failures can arise from missing initial conditions or incorrect generated problem files [5]. MRKL makes a different structural bet: route user inputs to specialized modules such as calculators, databases, and APIs [3]. This improves extensibility and provenance, but creates router and argument-extraction failure modes.

### Memory, reflection, and cross-episode adaptation

Reflection and memory architectures address the fact that many tasks require learning from failed attempts or maintaining state over time. Reflexion converts feedback into verbal reflections stored in episodic memory rather than updating weights; its loop includes an Actor, Evaluator, and Self-Reflection model, and the authors emphasize interpretable memory and no fine-tuning while acknowledging dependence on evaluator quality and absence of formal guarantees [8]. Generative-agent systems similarly use memory streams, reflection, and planning to sustain coherent simulated behavior over time, although long memory creates retrieval and consistency problems.

These methods are attractive because they are lightweight and inspectable. However, verbal self-critique is not equivalent to correct credit assignment. If the evaluator is noisy or the model invents an incorrect explanation for failure, reflection can entrench mistakes. Memory systems also need privacy controls, summarization policies, and relevance filters.

## Learning and evaluation of tool use

The evaluation landscape has shifted from asking whether an answer string is correct to asking whether an agent can change an environment into a desired state. API benchmarks diagnose call selection and argument validity; browser and desktop benchmarks test long-horizon stateful interaction; code benchmarks test executable patches.

WebArena is representative of outcome-based browser evaluation. It provides self-hostable functional sites spanning e-commerce, forums, collaborative development, and content management, with 812 long-horizon tasks and programmatic validators [9]. The reported best GPT-4 agent success rate was 14.41%, compared with 78.24% human performance, and the benchmark evaluates final functional correctness rather than exact action matching [9]. This matters because many valid web trajectories can complete the same task.

SWE-bench evaluates code-oriented tool use by giving a model a real GitHub issue and repository snapshot, applying the generated patch, and running tests in Docker; the original benchmark contains 2,294 problems from 12 Python repositories [10]. This is stronger than string matching, yet it is still test-suite-dependent: passing tests may miss untested behavior, while infrastructure errors can confound results [10]. AgentRewardBench further highlights that automatic judges of web-agent trajectories are themselves unreliable, with Hugging Face's summary noting that rule-based methods may underreport success [11].

A useful survey distinction is therefore between trajectory correctness and outcome correctness. Exact trajectory matching is usually too brittle; final-state checks are better but expensive to author and may miss side effects. LLM-as-judge evaluation scales but needs calibration against executable or human-verified outcomes.

## Safety, security, and governance

Tool use converts prompt-level errors into operational risk. Indirect prompt injection is a central example: malicious instructions can be embedded in external content that the agent reads through tools, causing it to leak data or perform unauthorized actions. InjecAgent studies such attacks in tool-integrated agents, with 1,054 cases across 17 user tools and 62 attacker tools; the notes report that ReAct-prompted GPT-4 followed attacks in 24% of cases, rising to 47% with a reinforced hacking prompt [12].

AgentDojo provides a dynamic, stateful environment for prompt-injection attacks and defenses, with 97 realistic tasks and 629 security test cases in domains such as email, e-banking, and travel [13]. A key lesson is that security must be measured alongside benign utility: an agent that refuses or fails everything is safe only in a vacuous sense [13]. AgentDojo also shows why fixed prompt suites can overstate robustness: adaptive attacks that select among injection phrasings can increase attack success, and defenses such as delimiters, prompt sandwiching, detectors, or tool filters have utility/security tradeoffs [13].

Safety risks are broader than injection. ToolEmu uses LM-emulated sandboxes to test high-stakes tools without real-world effects, covering 36 toolkits and 144 test cases; the notes report that the safest tested agent still exhibited potentially severe failures in 23.9% of cases under its evaluator [14]. AgentHarm targets direct malicious-user requests using synthetic side-effect-free tools and multi-step harmful tasks; the notes report that a universal jailbreak substantially increased harm scores for several frontier models while preserving multi-step competence [15].

The governance implication is architectural, not merely prompt-based. Stronger systems should use least-privilege tool exposure, read/write separation, scoped credentials, dry runs for state-changing calls, argument previews, sandboxing for code and high-impact tools, audit logs, and human confirmation before irreversible, financial, privacy-sensitive, external-communication, or destructive actions. Tool filtering before reading untrusted content is especially important because once the model has ingested hostile instructions, asking the same model to decide whether a call is safe is a weak access-control boundary [13].

## Beyond APIs: browser, computer-use, embodied, and multi-agent systems

Tool use extends naturally from discrete API calls to graphical interfaces, embodied environments, and teams of agents. OSWorld evaluates multimodal agents controlling real web and desktop applications with keyboard and mouse. In the original evaluation, the best model achieved 12.24% success while humans completed over 72.36% of tasks; the authors identify GUI grounding and operational knowledge as major weaknesses, with qualitative failures also arising from unexpected interface states or windows [16].

Embodied environments add physical or simulated grounding. ALFWorld aligns text-based TextWorld environments with visually embodied ALFRED worlds, allowing abstract policies to transfer to low-level embodied actions; the notes report that training first in TextWorld is 7x faster than training from scratch in the embodied world. This illustrates a broader design pattern: use abstract or text environments to learn high-level plans, then map those plans to lower-level controllers.

Multi-agent frameworks distribute tool use across roles. MetaGPT uses role specialization, standardized operating procedures, structured documents and diagrams, publish-subscribe messaging, and executable feedback to reduce cascading inconsistencies in software-generation workflows [17]. AutoGen frames agents as conversable components that can include LLMs, humans, tools, and code execution, with programmable conversation patterns. MultiAgentBench, as summarized on Hugging Face, focuses on collaboration and competition across scenarios and evaluates both task completion and collaboration quality [18].

The promise is decomposition: one agent can plan, another retrieve, another execute, and another critique. The cost is coordination overhead, duplicated errors, higher latency and token cost, and more complex safety boundaries. Parallel agents acting on shared tools can also race, conflict, or amplify unsafe actions unless permissions and state are managed centrally.

## Trends and open problems

First, the field is moving from prompt recipes to systems engineering. Successful agents increasingly combine LLMs with schemas, retrievers, planners, validators, sandboxes, and permissions rather than relying on a single prompt loop. The research challenge is to identify which parts should be learned, which should be symbolic, and which should be enforced by infrastructure.

Second, evaluation is becoming more realistic but also more fragile. Benchmarks such as WebArena, SWE-bench, OSWorld, AgentDojo, and AgentHarm provide executable or state-based signals, but they are expensive to maintain and benchmark-conditional. Reported scores should not be read as deployment incident probabilities. Future evaluations need held-out tools, adaptive attacks, infrastructure-error accounting, privacy side-effect checks, and longitudinal tasks requiring recovery over time.

Third, reliability and safety are inseparable. A model that cannot reliably call the intended tool may be unsafe even without malicious intent, and a model that can execute long-horizon plans becomes more dangerous when jailbroken or injected. Research on tool competence should therefore report benign success, success under attack, harm success, refusal/abort rates, and environment-state changes together.

Fourth, memory and reflection remain under-theorized. Verbal self-reflection is useful and interpretable, but it lacks formal guarantees and depends heavily on evaluators. Long-term memory raises retrieval, poisoning, privacy, and staleness issues. Future systems need memory provenance, deletion policies, trust labels, and mechanisms that distinguish user instructions from untrusted observations.

Fifth, human oversight must be selective and well placed. Asking for approval at every step destroys autonomy and causes fatigue; never asking makes irreversible errors likely. A better design is graduated autonomy: automatic execution for reversible low-risk calls, previews for state changes, and mandatory confirmation for high-impact actions with explicit arguments, destinations, and consequences.

Finally, multi-agent and computer-use systems expose the frontier of the field. Browser and OS agents reveal large human-agent gaps in grounding, recovery, and long-horizon control, while multi-agent systems raise unresolved questions about communication protocols, dynamic team formation, credit assignment, and collective safety. The likely near-term path is not a single fully autonomous general agent, but constrained agents embedded in carefully designed tool, permission, evaluation, and oversight infrastructures.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] MRKL Systems: A Modular, Neuro-Symbolic Architecture. hf-search. https://huggingface.co/papers/2205.00445 (2022-05-01)
[4] Tree of Thoughts: Deliberate Problem Solving with Large Language Models. hf-search. https://huggingface.co/papers/2305.10601 (2023-05-17)
[5] LLM+P: Empowering Large Language Models with Optimal Planning Proficiency. hf-search. https://huggingface.co/papers/2304.11477 (2023-04-22)
[6] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-31)
[7] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arxiv. https://arxiv.org/abs/2304.08244 (2023-04-14)
[8] Reflexion: Language Agents with Verbal Reinforcement Learning. arxiv. https://arxiv.org/abs/2303.11366 (n.d.)
[9] WebArena: A Realistic Web Environment for Building Autonomous Agents. arxiv. https://arxiv.org/abs/2307.13854 (2023-07-26)
[10] SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. arxiv. https://arxiv.org/abs/2310.06770 (2023)
[11] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. hf-daily. https://huggingface.co/papers/2504.08942 (2025-04-11)
[12] InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. web. https://aclanthology.org/2024.findings-acl.624/ (2024)
[13] AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. arxiv. https://arxiv.org/abs/2406.13352 (2024-06-19)
[14] Identifying the Risks of LM Agents with an LM-Emulated Sandbox. arxiv. https://arxiv.org/abs/2309.15817 (2023-09-27)
[15] AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents. arxiv. https://arxiv.org/abs/2410.09024 (2024-10-11)
[16] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. web. https://os-world.github.io/ (2024)
[17] MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. arxiv. https://arxiv.org/abs/2308.00352 (2023-08-01)
[18] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents. hf-search. https://huggingface.co/papers/2503.01935 (2025-03-03)
