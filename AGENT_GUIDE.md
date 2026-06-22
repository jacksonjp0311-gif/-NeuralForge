# 🤖 NeuralForge v2.0 — Agent Guide

This guide explains how any AI agent (LLM-based or multi-agent system) should
use NeuralForge to build, train, optimize, and deploy neural networks.

## Core Concept

NeuralForge exposes a **single tool** that agents can call with natural language:

```
"Build me a state-of-the-art image classifier for 10 classes with < 5M params
 that reaches >92% accuracy on CIFAR-10 within 30 minutes on a single A100."
```

From this single request, NeuralForge should:
1. Parse constraints (5M params, >92% accuracy, 30min budget)
2. Propose candidate architectures
3. Train the best candidate
4. Evaluate and iterate if needed
5. Return the final model + report

## Available Actions

| Action | Input | Output |
|--------|-------|--------|
| `full_pipeline` | `description: str` | End-to-end: create → train → evaluate |
| `create_model` | `description: str` | Model name, params, architecture |
| `train` | `model_name, epochs, batch_size, lr` | Training result with metrics |
| `optimize` | `objective, direction, num_trials` | Best spec, score, trials |
| `evaluate` | `model_name, test_data` | Evaluation report |
| `evolve` | `spec, generations` | Evolved spec |
| `auto_architecture` | `task, data_info, constraints` | Architecture proposals |
| `export` | `model_name, format` | Export path |

## Multi-Agent Mode

For complex tasks, use the multi-agent orchestration:

```python
from neuralforge.tools.multi_agent import ForgeOrchestrator

orch = ForgeOrchestrator()
session = orch.create_session(
    "Build a multimodal model for medical image + report analysis under 8GB VRAM"
)
results = orch.run_pipeline(session)
```

The orchestrator coordinates:
1. **ArchitectAgent** — Proposes architectures
2. **OptimizerAgent** — Searches hyperparameters
3. **EvaluatorAgent** — Evaluates and provides feedback
4. **DeployerAgent** — Handles export and deployment

All agents share a **blackboard** for communication.

## Self-Improvement Loop

NeuralForge includes a meta-optimizer that critiques each training run:

```python
from neuralforge.optimize.meta_optimizer import MetaOptimizer

meta = MetaOptimizer()
critique = meta.critique(spec, training_result)
# Returns: issues, suggestions, score

next_spec = meta.propose_next_spec(spec, training_result)
# Returns: improved spec with suggested changes
```

## Integration Examples

### LangChain ReAct Agent
```python
from langchain.agents import initialize_agent, AgentType
from neuralforge import get_all_langchain_tools

tools = get_all_langchain_tools()
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

agent.run("Build and train a ResNet for CIFAR-10 with 92% accuracy")
```

### CrewAI Crew
```python
from neuralforge import get_crewai_tools
from crewai import Agent, Task, Crew

architect = Agent(
    role="Neural Architect",
    goal="Design optimal neural architectures",
    tools=get_crewai_tools(),
)
```

### AutoGen
```python
from neuralforge import get_autogen_functions

assistant = AssistantAgent(
    name="neural_architect",
    llm_config={
        "functions": get_autogen_functions(),
    },
)
```

## Best Practices for Agents

1. **Use `full_pipeline` for most tasks** — It handles the full flow
2. **Specify constraints clearly** — Include params, memory, accuracy, time budgets
3. **Iterate if needed** — Use the meta-optimizer's suggestions for improvement
4. **Store insights** — Use the InsightsStore for long-term learning
5. **Profile before training** — Check model size and latency before committing
6. **Export for deployment** — Always export the final model

## Error Handling

All tools return structured responses:
```json
{"status": "success", "result": {...}}
{"status": "error", "error": "Description of what went wrong"}
```

Agents should check `status` and handle errors gracefully.
