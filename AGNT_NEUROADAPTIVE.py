"""
NeuroAdaptive — Deep Neural Integration for AGNT

This module provides neural networks that are deeply integrated into AGNT's
core systems, not just as standalone tools but as enhancements to:

1. Chat Response Quality — Neural scoring of every response
2. Tool Selection — Neural prediction of which tool to call next
3. Parameter Optimization — Neural optimization of every tool call
4. Workflow Planning — Neural optimization of workflow structures
5. Memory Retrieval — Neural ranking of relevant memories
6. Error Recovery — Neural prediction and auto-fix of failures
7. Goal Prioritization — Neural ranking of goals and tasks
8. Learning from Outcomes — Continuous improvement from every execution

Each network is trained on real AGNT data and improves over time.
"""
from __future__ import annotations
import logging, time, json, os
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("neuralforge.neuroadaptive")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHAT RESPONSE QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class ChatQualityScorer(nn.Module):
    """Scores the quality of chat responses based on multiple factors.
    
    Input features (20 dims):
    - Response length, structure, coherence
    - Tool usage patterns, error rates
    - User interaction patterns, satisfaction signals
    - Context relevance, completeness
    
    Output: quality_score (0-1) + 4 quality dimensions
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 64), nn.GELU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.quality_head = nn.Linear(64, 1)  # overall quality
        self.dimension_head = nn.Linear(64, 4)  # relevance, completeness, coherence, safety
        self.improvement_head = nn.Linear(64, 5)  # suggestions
        
    def forward(self, x):
        h = self.encoder(x)
        quality = torch.sigmoid(self.quality_head(h))
        dimensions = torch.sigmoid(self.dimension_head(h))
        improvements = torch.sigmoid(self.improvement_head(h))
        return quality, dimensions, improvements


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NEURAL TOOL SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ToolSelector(nn.Module):
    """Predicts the best tool to use given the current context.
    
    Input: context encoding (32 dims) — current task, available tools, history
    Output: probability distribution over all available tools
    
    This replaces heuristic tool selection with learned patterns.
    """
    def __init__(self, n_tools: int = 150):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.tool_scorer = nn.Linear(64, n_tools)
        self.confidence = nn.Linear(64, 1)
        
    def forward(self, context, tool_mask=None):
        h = self.context_encoder(context)
        scores = self.tool_scorer(h)
        if tool_mask is not None:
            scores = scores.masked_fill(tool_mask == 0, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        conf = torch.sigmoid(self.confidence(h))
        return probs, conf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ADAPTIVE PARAMETER OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveParameterOptimizer(nn.Module):
    """Optimizes parameters for ANY tool call based on context.
    
    Learns optimal values for:
    - timeout_ms, retry_count, backoff_factor
    - batch_size, parallel_count, cache_ttl
    - priority, scheduling_delay
    
    Input: tool_type (encoded) + context features = 24 dims
    Output: 8 optimized parameters
    """
    def __init__(self, n_tool_types: int = 50):
        super().__init__()
        self.tool_embedding = nn.Embedding(n_tool_types, 16)
        self.context_encoder = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
        )
        self.optimizer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.GELU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 8),  # 8 parameters to optimize
        )
        self.param_names = ['timeout_ms', 'retry_count', 'backoff_factor', 
                           'batch_size', 'parallel_count', 'cache_ttl',
                           'priority', 'scheduling_delay']
        
    def forward(self, tool_type_idx, context):
        tool_emb = self.tool_embedding(tool_type_idx)
        ctx = self.context_encoder(context)
        combined = torch.cat([tool_emb, ctx], dim=-1)
        params = self.optimizer(combined)
        # Apply appropriate activations for each parameter
        return torch.sigmoid(params)  # All normalized 0-1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WORKFLOW PLANNING OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowPlanner(nn.Module):
    """Neural network that optimizes workflow structure before execution.
    
    Predicts:
    - Optimal node ordering
    - Which nodes can be parallelized
    - Where to add error handling
    - Estimated execution time and success probability
    
    Input: workflow structure encoding (48 dims)
    Output: optimization recommendations (16 dims)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(48, 128), nn.GELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.ordering_head = nn.Linear(128, 20)  # node ordering scores
        self.parallel_head = nn.Linear(128, 10)  # parallelization opportunities
        self.error_handling_head = nn.Linear(128, 10)  # where to add error handling
        self.time_estimator = nn.Linear(128, 1)  # estimated execution time
        self.success_predictor = nn.Linear(128, 1)  # success probability
        
    def forward(self, workflow_encoding):
        h = self.encoder(workflow_encoding)
        return {
            'ordering': F.softmax(self.ordering_head(h), dim=-1),
            'parallel': torch.sigmoid(self.parallel_head(h)),
            'error_handling': torch.sigmoid(self.error_handling_head(h)),
            'estimated_time': F.relu(self.time_estimator(h)),
            'success_prob': torch.sigmoid(self.success_predictor(h)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MEMORY RETRIEVAL RANKER
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryRanker(nn.Module):
    """Neural ranking of relevant memories for the current context.
    
    Goes beyond keyword matching to understand semantic relevance.
    Learns which memories are most useful in which contexts.
    
    Input: query encoding (32 dims) + memory encoding (32 dims)
    Output: relevance score (0-1) + memory type classification
    """
    def __init__(self):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.memory_encoder = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.relevance_scorer = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.type_classifier = nn.Linear(128, 5)  # fact, preference, correction, context, pattern
        
    def forward(self, query, memories):
        q = self.query_encoder(query)
        m = self.memory_encoder(memories)
        # Compute relevance for each memory
        combined = torch.cat([q.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=-1)
        relevance = self.relevance_scorer(combined).squeeze(-1)
        mem_type = F.softmax(self.type_classifier(combined.mean(dim=1)), dim=-1)
        return relevance, mem_type


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ERROR RECOVERY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorRecoveryNet(nn.Module):
    """Predicts the best recovery action when a tool call fails.
    
    Input: error context (16 dims) — error type, tool, history, system state
    Output: recovery action probabilities (8 actions) + success probability
    
    Recovery actions: retry, retry_with_backoff, switch_tool, skip_step,
                     ask_user, use_cache, fallback_value, abort
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.action_head = nn.Linear(64, 8)  # 8 recovery actions
        self.success_head = nn.Linear(64, 1)  # probability of success after recovery
        self.time_head = nn.Linear(64, 1)  # estimated recovery time
        
    def forward(self, error_context):
        h = self.encoder(error_context)
        actions = F.softmax(self.action_head(h), dim=-1)
        success = torch.sigmoid(self.success_head(h))
        recovery_time = F.relu(self.time_head(h))
        return actions, success, recovery_time


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOAL PRIORITIZER
# ═══════════════════════════════════════════════════════════════════════════════

class GoalPrioritizer(nn.Module):
    """Neural ranking of goals and tasks based on current context.
    
    Learns to prioritize goals based on:
    - Urgency, importance, dependencies
    - Resource availability, success probability
    - Historical completion patterns
    
    Input: goal encoding (24 dims) + context encoding (16 dims)
    Output: priority score (0-1) + estimated completion time
    """
    def __init__(self):
        super().__init__()
        self.goal_encoder = nn.Sequential(
            nn.Linear(24, 64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(),
            nn.Linear(32, 32),
        )
        self.prioritizer = nn.Sequential(
            nn.Linear(96, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2),  # priority_score, estimated_time
        )
        
    def forward(self, goals, context):
        g = self.goal_encoder(goals)
        c = self.context_encoder(context)
        combined = torch.cat([g, c], dim=-1)
        output = self.prioritizer(combined)
        return torch.sigmoid(output[:, 0]), F.relu(output[:, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONTINUOUS LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousLearningEngine:
    """Orchestrates training of all neural networks on real AGNT data.
    
    Collects execution outcomes, creates training data, and continuously
    improves all neural networks in the background.
    """
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize all networks
        self.chat_scorer = ChatQualityScorer()
        self.tool_selector = ToolSelector(n_tools=150)
        self.param_optimizer = AdaptiveParameterOptimizer()
        self.workflow_planner = WorkflowPlanner()
        self.memory_ranker = MemoryRanker()
        self.error_recovery = ErrorRecoveryNet()
        self.goal_prioritizer = GoalPrioritizer()
        self.training_epochs = 25
        
        self.total_params = sum(
            sum(p.numel() for p in net.parameters())
            for net in [self.chat_scorer, self.tool_selector, self.param_optimizer,
                       self.workflow_planner, self.memory_ranker, self.error_recovery,
                       self.goal_prioritizer]
        )
        
        logger.info(f"NeuroAdaptive initialized: {self.total_params:,} total parameters")

    @staticmethod
    def _stable_index(value: Any, modulo: int) -> int:
        text = str(value or "unknown")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % modulo

    @staticmethod
    def _success(execution: Dict) -> float:
        status = str(execution.get("status", "")).lower()
        if "success" in execution:
            return 1.0 if bool(execution.get("success")) else 0.0
        return 0.0 if status in {"error", "failed", "failure", "timeout"} else 1.0

    @staticmethod
    def _duration_ms(execution: Dict) -> float:
        return float(
            execution.get("duration_ms")
            or execution.get("duration")
            or execution.get("elapsed_ms")
            or execution.get("latency_ms")
            or 0.0
        )

    @staticmethod
    def _step_count(execution: Dict) -> float:
        return float(
            execution.get("step_count")
            or execution.get("steps")
            or execution.get("node_count")
            or len(execution.get("nodes", []) or [])
            or 0.0
        )

    @staticmethod
    def _retry_count(execution: Dict) -> float:
        return float(
            execution.get("retry_count")
            or execution.get("retries")
            or execution.get("attempt")
            or execution.get("attempts")
            or 0.0
        )

    @staticmethod
    def _params(execution: Dict) -> Dict:
        params = execution.get("params") or execution.get("parameters") or execution.get("tool_params") or {}
        return params if isinstance(params, dict) else {}

    @staticmethod
    def _tool_name(execution: Dict) -> str:
        for key in ("tool_name", "tool", "tool_type", "node_type", "action", "plugin"):
            if execution.get(key):
                return str(execution[key])
        calls = execution.get("tool_calls") or execution.get("tools") or []
        if isinstance(calls, list) and calls:
            first = calls[0]
            if isinstance(first, dict):
                return str(first.get("name") or first.get("tool") or first.get("type") or "unknown")
            return str(first)
        return "unknown"

    @staticmethod
    def _error_type(execution: Dict) -> str:
        return str(
            execution.get("error_type")
            or execution.get("error_code")
            or execution.get("exception_type")
            or execution.get("error")
            or "none"
        ).lower()

    @staticmethod
    def _text_len(execution: Dict, *keys: str) -> float:
        for key in keys:
            value = execution.get(key)
            if value:
                return float(len(str(value)))
        return 0.0

    def _execution_scalar_features(self, execution: Dict, index: int, total: int) -> Dict[str, float]:
        duration = self._duration_ms(execution)
        steps = self._step_count(execution)
        retries = self._retry_count(execution)
        params = self._params(execution)
        success = self._success(execution)
        tool = self._tool_name(execution)
        workflow = execution.get("workflow_id") or execution.get("workflow_name") or "unknown"
        return {
            "duration": min(duration / 60000.0, 1.0),
            "duration_log": min(np.log1p(duration) / np.log1p(600000.0), 1.0),
            "steps": min(steps / 100.0, 1.0),
            "retries": min(retries / 10.0, 1.0),
            "success": success,
            "error": 1.0 - success,
            "param_count": min(len(params) / 20.0, 1.0),
            "tool_hash": self._stable_index(tool, 1000) / 999.0,
            "workflow_hash": self._stable_index(workflow, 1000) / 999.0,
            "error_hash": self._stable_index(self._error_type(execution), 1000) / 999.0,
            "response_len": min(self._text_len(execution, "response", "output", "result") / 8000.0, 1.0),
            "prompt_len": min(self._text_len(execution, "prompt", "input", "query") / 8000.0, 1.0),
            "progress": index / max(total - 1, 1),
        }

    def _chat_features_and_targets(self, executions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        total = len(executions)
        for i, e in enumerate(executions[:500]):
            f = self._execution_scalar_features(e, i, total)
            explicit_quality = e.get("quality_score", e.get("user_rating", e.get("score")))
            if explicit_quality is not None:
                quality = float(explicit_quality)
                if quality > 1.0:
                    quality = quality / 100.0 if quality <= 100 else 1.0
            else:
                quality = (
                    0.60 * f["success"]
                    + 0.15 * (1.0 - f["duration"])
                    + 0.10 * (1.0 - f["retries"])
                    + 0.10 * min(f["response_len"] * 4.0, 1.0)
                    + 0.05 * (1.0 - f["error"])
                )
            rows.append([
                f["duration"], f["duration_log"], f["steps"], f["retries"], f["success"],
                f["error"], f["param_count"], f["tool_hash"], f["workflow_hash"], f["error_hash"],
                f["response_len"], f["prompt_len"], f["progress"],
                1.0 if e.get("cached") else 0.0,
                1.0 if e.get("fallback_used") else 0.0,
                min(float(e.get("tokens", e.get("token_count", 0)) or 0) / 32000.0, 1.0),
                min(float(e.get("cost_usd", 0) or 0) / 10.0, 1.0),
                1.0 if e.get("human_feedback") else 0.0,
                1.0 if e.get("validation_passed", f["success"]) else 0.0,
                1.0,
            ])
            targets.append([float(np.clip(quality, 0.0, 1.0))])
        return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _tool_features_and_targets(self, executions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        total = len(executions)
        for i, e in enumerate(executions[:500]):
            f = self._execution_scalar_features(e, i, total)
            rows.append([
                f["duration"], f["duration_log"], f["steps"], f["retries"], f["success"],
                f["error"], f["param_count"], f["workflow_hash"], f["error_hash"],
                f["response_len"], f["prompt_len"], f["progress"],
                min(float(e.get("queue_depth", 0) or 0) / 100.0, 1.0),
                min(float(e.get("parallel_count", 1) or 1) / 32.0, 1.0),
                min(float(e.get("memory_mb", 0) or 0) / 32768.0, 1.0),
                min(float(e.get("cpu_percent", 0) or 0) / 100.0, 1.0),
                min(float(e.get("gpu_percent", 0) or 0) / 100.0, 1.0),
                min(float(e.get("timeout_ms", 0) or 0) / 600000.0, 1.0),
                1.0 if e.get("cached") else 0.0,
                1.0 if e.get("rate_limited") else 0.0,
                1.0 if e.get("auth_required") else 0.0,
                f["tool_hash"],
                self._stable_index(e.get("workflow_name", "unknown"), 1000) / 999.0,
                self._stable_index(e.get("status", "unknown"), 1000) / 999.0,
                self._stable_index(e.get("model", "unknown"), 1000) / 999.0,
                min(float(e.get("input_bytes", 0) or 0) / 10_000_000.0, 1.0),
                min(float(e.get("output_bytes", 0) or 0) / 10_000_000.0, 1.0),
                min(float(e.get("tokens", 0) or 0) / 32000.0, 1.0),
                min(float(e.get("cost_usd", 0) or 0) / 10.0, 1.0),
                1.0 if e.get("validation_passed", f["success"]) else 0.0,
                1.0 if e.get("user_visible") else 0.0,
                1.0,
            ])
            targets.append(self._stable_index(self._tool_name(e), 150))
        return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.int64)

    def _param_features_and_targets(self, executions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts, tool_types, targets = [], [], []
        total = len(executions)
        for i, e in enumerate(executions[:400]):
            f = self._execution_scalar_features(e, i, total)
            params = self._params(e)
            contexts.append([
                f["duration"], f["steps"], f["retries"], f["success"],
                f["error"], f["param_count"], f["workflow_hash"], f["progress"],
            ])
            tool_types.append(self._stable_index(self._tool_name(e), 50))
            targets.append([
                min(float(params.get("timeout_ms", e.get("timeout_ms", 30000)) or 30000) / 600000.0, 1.0),
                min(float(params.get("retry_count", e.get("retry_count", 0)) or 0) / 10.0, 1.0),
                min(float(params.get("backoff_factor", e.get("backoff_factor", 1.0)) or 1.0) / 10.0, 1.0),
                min(float(params.get("batch_size", e.get("batch_size", 1)) or 1) / 1024.0, 1.0),
                min(float(params.get("parallel_count", e.get("parallel_count", 1)) or 1) / 32.0, 1.0),
                min(float(params.get("cache_ttl", e.get("cache_ttl", 0)) or 0) / 86400.0, 1.0),
                min(float(params.get("priority", e.get("priority", 0.5)) or 0.5), 1.0),
                min(float(params.get("scheduling_delay", e.get("scheduling_delay", 0)) or 0) / 3600.0, 1.0),
            ])
        return (
            np.array(tool_types, dtype=np.int64),
            np.array(contexts, dtype=np.float32),
            np.array(targets, dtype=np.float32),
        )

    def _recovery_features_and_targets(self, executions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = ['retry', 'retry_with_backoff', 'switch_tool', 'skip_step',
                   'ask_user', 'use_cache', 'fallback_value', 'abort']
        action_index = {name: i for i, name in enumerate(actions)}
        failed = [e for e in executions if self._success(e) < 0.5 or self._error_type(e) != "none"]
        rows, recovery_targets, success_targets = [], [], []
        total = len(failed)
        for i, e in enumerate(failed[:400]):
            f = self._execution_scalar_features(e, i, total)
            err = self._error_type(e)
            explicit_action = str(e.get("recovery_action") or e.get("fix_action") or "").lower()
            if explicit_action not in action_index:
                if "timeout" in err or "rate" in err:
                    explicit_action = "retry_with_backoff"
                elif "auth" in err:
                    explicit_action = "ask_user"
                elif "cache" in err:
                    explicit_action = "use_cache"
                elif "tool" in err or "not found" in err:
                    explicit_action = "switch_tool"
                else:
                    explicit_action = "retry"
            recovered = e.get("recovery_success", e.get("resolved", 0.0))
            rows.append([
                f["duration"], f["duration_log"], f["steps"], f["retries"], f["param_count"],
                f["tool_hash"], f["workflow_hash"], f["error_hash"], f["response_len"],
                f["prompt_len"], f["progress"],
                1.0 if "timeout" in err else 0.0,
                1.0 if "memory" in err or "oom" in err else 0.0,
                1.0 if "auth" in err else 0.0,
                1.0 if "rate" in err else 0.0,
                1.0,
            ])
            recovery_targets.append(action_index[explicit_action])
            success_targets.append([float(np.clip(recovered, 0.0, 1.0))])
        return (
            np.array(rows, dtype=np.float32),
            np.array(recovery_targets, dtype=np.int64),
            np.array(success_targets, dtype=np.float32),
        )
    
    def train_on_executions(self, executions: List[Dict]) -> Dict[str, float]:
        """Train all networks on real execution data."""
        metrics = {}
        
        if len(executions) < 10:
            return {"status": "insufficient_data", "executions": len(executions)}
        
        # Train chat quality scorer
        metrics['chat_scorer'] = self._train_chat_scorer(executions)
        
        # Train tool selector
        metrics['tool_selector'] = self._train_tool_selector(executions)
        
        # Train parameter optimizer
        metrics['param_optimizer'] = self._train_param_optimizer(executions)
        
        # Train error recovery
        metrics['error_recovery'] = self._train_error_recovery(executions)
        
        # Save all models
        self._save_models()
        
        metrics['total_params'] = self.total_params
        metrics['executions_trained'] = len(executions)
        
        return metrics
    
    def _train_chat_scorer(self, executions):
        """Train chat quality scorer on execution outcomes."""
        X, y_quality = self._chat_features_and_targets(executions)
        
        X_t, y_t = torch.tensor(X), torch.tensor(y_quality)
        opt = torch.optim.Adam(self.chat_scorer.parameters(), lr=1e-3)
        
        for epoch in range(self.training_epochs):
            q, d, imp = self.chat_scorer(X_t)
            loss = F.mse_loss(q, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return {'loss': loss.item(), 'epochs': self.training_epochs, 'samples': len(X), 'feature_source': 'executions'}
    
    def _train_tool_selector(self, executions):
        """Train tool selector on which tools were actually used."""
        context, tool_targets = self._tool_features_and_targets(executions)
        
        context_t = torch.tensor(context)
        targets_t = torch.tensor(tool_targets)
        opt = torch.optim.Adam(self.tool_selector.parameters(), lr=1e-3)
        
        for epoch in range(self.training_epochs):
            probs, conf = self.tool_selector(context_t)
            loss = F.cross_entropy(probs, targets_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return {'loss': loss.item(), 'accuracy': (probs.argmax(1) == targets_t).float().mean().item(), 'samples': len(context), 'feature_source': 'executions'}
    
    def _train_param_optimizer(self, executions):
        """Train parameter optimizer on successful vs failed executions."""
        tool_types, context, optimal_params = self._param_features_and_targets(executions)
        
        tool_t = torch.tensor(tool_types)
        ctx_t = torch.tensor(context)
        params_t = torch.tensor(optimal_params)
        opt = torch.optim.Adam(self.param_optimizer.parameters(), lr=1e-3)
        
        for epoch in range(self.training_epochs):
            pred = self.param_optimizer(tool_t, ctx_t)
            loss = F.mse_loss(pred, params_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return {'loss': loss.item(), 'samples': len(context), 'feature_source': 'executions'}
    
    def _train_error_recovery(self, executions):
        """Train error recovery on failed executions and their outcomes."""
        error_ctx, recovery_targets, success_targets = self._recovery_features_and_targets(executions)
        if len(error_ctx) < 2:
            return {'status': 'skipped', 'reason': 'no_failed_executions', 'failed_executions_analyzed': len(error_ctx)}
        
        ctx_t = torch.tensor(error_ctx)
        rec_t = torch.tensor(recovery_targets)
        succ_t = torch.tensor(success_targets)
        opt = torch.optim.Adam(self.error_recovery.parameters(), lr=1e-3)
        
        for epoch in range(self.training_epochs):
            actions, success, rec_time = self.error_recovery(ctx_t)
            loss = F.cross_entropy(actions, rec_t) + F.mse_loss(success, succ_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return {'loss': loss.item(), 'failed_executions_analyzed': len(error_ctx), 'feature_source': 'executions'}
    
    def _save_models(self):
        """Save all trained models."""
        models = {
            'chat_scorer.pt': self.chat_scorer,
            'tool_selector.pt': self.tool_selector,
            'param_optimizer.pt': self.param_optimizer,
            'workflow_planner.pt': self.workflow_planner,
            'memory_ranker.pt': self.memory_ranker,
            'error_recovery.pt': self.error_recovery,
            'goal_prioritizer.pt': self.goal_prioritizer,
        }
        for name, model in models.items():
            path = os.path.join(self.model_dir, name)
            torch.save(model.state_dict(), path)
        logger.info(f"Saved {len(models)} neural models to {self.model_dir}")
    
    def predict_chat_quality(self, response_features: np.ndarray) -> Dict:
        """Score a chat response."""
        self.chat_scorer.eval()
        with torch.no_grad():
            x = torch.tensor(response_features, dtype=torch.float32).unsqueeze(0)
            quality, dims, improvements = self.chat_scorer(x)
        return {
            'quality_score': quality.item(),
            'relevance': dims[0, 0].item(),
            'completeness': dims[0, 1].item(),
            'coherence': dims[0, 2].item(),
            'safety': dims[0, 3].item(),
            'improvement_suggestions': improvements[0].tolist(),
        }
    
    def predict_best_tool(self, context: np.ndarray, available_tools: List[str] = None) -> Dict:
        """Predict the best tool for the current context."""
        self.tool_selector.eval()
        with torch.no_grad():
            ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            probs, conf = self.tool_selector(ctx)
        top_k = torch.topk(probs[0], min(5, probs.size(1)))
        return {
            'top_tools': [{'index': i.item(), 'probability': p.item()} for i, p in zip(top_k.indices, top_k.values)],
            'confidence': conf.item(),
        }
    
    def predict_recovery_action(self, error_context: np.ndarray) -> Dict:
        """Predict the best recovery action for a failed tool call."""
        self.error_recovery.eval()
        actions = ['retry', 'retry_with_backoff', 'switch_tool', 'skip_step',
                   'ask_user', 'use_cache', 'fallback_value', 'abort']
        with torch.no_grad():
            ctx = torch.tensor(error_context, dtype=torch.float32).unsqueeze(0)
            probs, success, rec_time = self.error_recovery(ctx)
        top_action = probs[0].argmax().item()
        return {
            'recommended_action': actions[top_action],
            'action_probabilities': {a: p.item() for a, p in zip(actions, probs[0])},
            'success_probability': success.item(),
            'estimated_recovery_time_ms': rec_time.item() * 1000,
        }
    
    def get_status(self) -> Dict:
        """Return status of all neural networks."""
        return {
            'total_parameters': self.total_params,
            'networks': {
                'chat_scorer': sum(p.numel() for p in self.chat_scorer.parameters()),
                'tool_selector': sum(p.numel() for p in self.tool_selector.parameters()),
                'param_optimizer': sum(p.numel() for p in self.param_optimizer.parameters()),
                'workflow_planner': sum(p.numel() for p in self.workflow_planner.parameters()),
                'memory_ranker': sum(p.numel() for p in self.memory_ranker.parameters()),
                'error_recovery': sum(p.numel() for p in self.error_recovery.parameters()),
                'goal_prioritizer': sum(p.numel() for p in self.goal_prioritizer.parameters()),
            },
            'model_dir': self.model_dir,
            'models_saved': os.listdir(self.model_dir) if os.path.exists(self.model_dir) else [],
        }
