from __future__ import annotations
import logging, math, time, copy, random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from neuralforge.spec import *
from neuralforge.core.registry import ModelRegistry
logger = logging.getLogger("neuralforge.engine")


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class MixerBlock(nn.Module):
    def __init__(self, d_model, num_patches):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp1 = nn.Sequential(nn.Linear(num_patches, num_patches*2), nn.GELU(), nn.Linear(num_patches*2, num_patches))
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp2 = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
    def forward(self, x):
        y = self.norm1(x).transpose(1, 2); y = self.mlp1(y).transpose(1, 2); x = x + y
        return x + self.mlp2(self.norm2(x))


class KANLinear(nn.Module):
    def __init__(self, in_f, out_f, num_basis=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.basis_coeffs = nn.Parameter(torch.randn(in_f, num_basis) * 0.1)
    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        basis = torch.sigmoid(x.unsqueeze(-1) * self.basis_coeffs.unsqueeze(0)).mean(-1)
        return base + 0.1 * basis


def _get_in_ch(dp):
    if dp is None:
        return 3
    if dp.num_channels is not None:
        return dp.num_channels
    if len(dp.input_shape) >= 1:
        return dp.input_shape[0]
    return 3


def _get_num_classes(dp):
    if dp is None:
        return 10
    return dp.num_classes if dp.num_classes is not None else 10


class NeuralForgeModule(nn.Module):
    def __init__(self, spec: NeuralForgeSpec):
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList()
        self._build()

    def _build(self):
        arch = self.spec.architecture
        family = arch.family
        if family == ArchitectureFamily.AUTO:
            family = self._infer_family()
        if family == ArchitectureFamily.CNN:
            self._build_cnn()
        elif family == ArchitectureFamily.RESNET:
            self._build_resnet()
        elif family == ArchitectureFamily.TRANSFORMER:
            self._build_transformer()
        elif family == ArchitectureFamily.VISION_TRANSFORMER:
            self._build_vit()
        elif family == ArchitectureFamily.MLP_MIXER:
            self._build_mlp_mixer()
        elif family == ArchitectureFamily.KAN:
            self._build_kan()
        else:
            self._build_cnn()
            logger.warning(f"{family.value} fallback to CNN")

    def _infer_family(self):
        dp = self.spec.data_profile
        if dp is None:
            return ArchitectureFamily.CNN
        if dp.data_format == "image":
            return ArchitectureFamily.RESNET
        if dp.data_format == "text":
            return ArchitectureFamily.TRANSFORMER
        return ArchitectureFamily.CNN

    def _build_cnn(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        in_ch = _get_in_ch(dp)
        nc = _get_num_classes(dp)
        h = arch.width or 64
        d = arch.depth or 4
        ch = [in_ch] + [h * (2**min(i, 3)) for i in range(d)]
        for i in range(d):
            self.layers.append(nn.Conv2d(ch[i], ch[i+1], 3, padding=1))
            self.layers.append(nn.BatchNorm2d(ch[i+1]))
            self.layers.append(nn.GELU())
            if (i+1) % 2 == 0:
                self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(ch[-1], nc)

    def _build_resnet(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        in_ch = _get_in_ch(dp)
        nc = _get_num_classes(dp)
        h = arch.width or 64
        d = arch.depth or 4
        self.input_conv = nn.Sequential(nn.Conv2d(in_ch, h, 3, padding=1), nn.BatchNorm2d(h), nn.GELU())
        self.res_blocks = nn.ModuleList()
        for i in range(d):
            out_c = h * (2**min(i//2, 3))
            in_c = h * (2**max(0, (i-1)//2)) if i > 0 else h
            self.res_blocks.append(ResBlock(in_c, out_c, stride=2 if i > 0 and i%2 == 0 else 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_c, nc)

    def _build_transformer(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        vs = dp.vocab_size if dp and dp.vocab_size else 30000
        dm = arch.embedding_dim or arch.width or 256
        nh = arch.num_heads or 8
        d = arch.depth or 4
        ml = dp.max_sequence_length if dp and dp.max_sequence_length else 512
        nc = _get_num_classes(dp)
        self.embedding = nn.Embedding(vs, dm)
        self.pos_embedding = nn.Embedding(ml, dm)
        el = nn.TransformerEncoderLayer(dm, nh, dm*4, arch.dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, d)
        self.classifier = nn.Linear(dm, nc)

    def _build_vit(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        in_ch = _get_in_ch(dp)
        nc = _get_num_classes(dp)
        dm = arch.embedding_dim or 256
        nh = arch.num_heads or 8
        d = arch.depth or 6
        ps = 4
        self.patch_embed = nn.Conv2d(in_ch, dm, ps, stride=ps)
        np_ = (32 // ps) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, dm))
        self.pos_embedding = nn.Parameter(torch.randn(1, np_+1, dm))
        el = nn.TransformerEncoderLayer(dm, nh, dm*4, arch.dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, d)
        self.classifier = nn.Linear(dm, nc)

    def _build_mlp_mixer(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        in_ch = _get_in_ch(dp)
        nc = _get_num_classes(dp)
        dm = arch.embedding_dim or 256
        ps = 4
        d = arch.depth or 4
        np_ = (32 // ps) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dm, ps, stride=ps)
        self.blocks = nn.ModuleList([MixerBlock(dm, np_) for _ in range(d)])
        self.norm = nn.LayerNorm(dm)
        self.classifier = nn.Linear(dm, nc)

    def _build_kan(self):
        arch = self.spec.architecture
        dp = self.spec.data_profile
        in_ch = _get_in_ch(dp)
        nc = _get_num_classes(dp)
        h = arch.width or 64
        self.layers_module = nn.Sequential(
            nn.Flatten(),
            KANLinear(in_ch * 32 * 32, h),
            KANLinear(h, h),
            KANLinear(h, nc),
        )

    def forward(self, x):
        arch = self.spec.architecture
        family = arch.family
        if family == ArchitectureFamily.AUTO:
            family = self._infer_family()
        if family == ArchitectureFamily.RESNET:
            x = self.input_conv(x)
            for block in self.res_blocks:
                x = block(x)
            x = self.pool(x).flatten(1)
            return self.classifier(x)
        elif family == ArchitectureFamily.TRANSFORMER:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(positions)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.classifier(x)
        elif family == ArchitectureFamily.VISION_TRANSFORMER:
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embedding[:, :x.size(1), :]
            x = self.transformer(x)
            x = x[:, 0]  # CLS token
            return self.classifier(x)
        elif family == ArchitectureFamily.MLP_MIXER:
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x).mean(dim=1)
            return self.classifier(x)
        elif family == ArchitectureFamily.KAN:
            return self.layers_module(x)
        else:
            for layer in self.layers:
                x = layer(x)
            x = x.flatten(1)
            return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralForgeEngine:
    _instance = None
    def __init__(self):
        self.registry = ModelRegistry.get_instance()
        self._active_models = {}
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def register_model(self, name, model):
        self._active_models[name] = model
    def get_model(self, name):
        return self._active_models.get(name)


def create_model(spec: NeuralForgeSpec) -> NeuralForgeModule:
    logger.info(f"Creating model: {spec.name} (family={spec.architecture.family.value})")
    model = NeuralForgeModule(spec)
    pc = model.count_parameters()
    logger.info(f"Parameters: {pc:,}")
    if spec.constraints.max_parameters and pc > spec.constraints.max_parameters:
        logger.warning(f"Exceeds constraint: {pc:,} > {spec.constraints.max_parameters:,}")
    engine = NeuralForgeEngine.get_instance()
    engine.register_model(spec.name, model)
    engine.registry.register(spec.name, spec, metadata={"parameters": pc})
    return model


def train(model, dataset, config=None, optimization_plan=None):
    if config is None:
        config = TrainingConfig()
    if isinstance(dataset, (list, tuple)) and len(dataset) == 2:
        train_loader, val_loader = dataset
    elif isinstance(dataset, DataLoader):
        train_loader = dataset
        val_loader = None
    elif isinstance(dataset, Dataset):
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = None
    else:
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.manual_seed(config.seed)
    if config.optimizer == OptimizerName.ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == OptimizerName.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    if config.scheduler == SchedulerName.COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=config.min_lr)
    else:
        scheduler = None
    use_amp = config.precision in ("fp16", "bf16", "mixed")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda" and config.precision == "fp16"))
    amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoints = []
    status = "completed"
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets, label_smoothing=config.label_smoothing) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        val_loss = None
        if val_loader is not None:
            model.eval()
            vl_sum = 0.0
            vb = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                    else:
                        inputs = batch.to(device)
                        targets = batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        vl_sum += F.cross_entropy(model(inputs), targets).item()
                    vb += 1
            val_loss = vl_sum / max(vb, 1)
            history["val_loss"].append(val_loss)
        metric = val_loss if val_loss is not None else avg_train_loss
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.early_stopping_patience:
            break
    total_time = time.time() - start_time
    return TrainingResult(
        model_name="model", spec_hash="", epochs_completed=epoch,
        total_steps=epoch * len(train_loader), final_loss=avg_train_loss,
        best_metric=best_metric, best_epoch=best_epoch,
        training_time_seconds=total_time, checkpoints=checkpoints,
        history=history, status=status,
    )


def export_model(model, config, sample_input=None):
    from pathlib import Path
    export_dir = Path(config.output_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.format
    if fmt == ExportFormat.PYTORCH_STATE_DICT:
        path = export_dir / "model.pt"
        torch.save(model.state_dict(), str(path))
        return str(path)
    elif fmt == ExportFormat.SAFETENSORS:
        try:
            from safetensors.torch import save_file
            path = export_dir / "model.safetensors"
            save_file(model.state_dict(), str(path))
            return str(path)
        except ImportError:
            path = export_dir / "model.pt"
            torch.save(model.state_dict(), str(path))
            return str(path)
    else:
        path = export_dir / "model.pt"
        torch.save(model.state_dict(), str(path))
        return str(path)


def auto_architecture(task, data_info, constraints=None, budget=None):
    if constraints is None:
        constraints = Constraints()
    if budget is None:
        budget = ComputeBudget()
    spec = NeuralForgeSpec(
        name=f"auto-{data_info.task_type.value}", description=task,
        task_type=data_info.task_type, data_profile=data_info,
        constraints=constraints, budget=budget,
    )
    return spec


def optimize(objective, model=None, base_spec=None, budget=None):
    if budget is None:
        budget = {}
    max_trials = int(budget.get("num_trials", objective.budget.num_trials))
    best_spec = base_spec or NeuralForgeSpec()
    best_score = float("-inf")
    all_trials = []
    for i in range(max_trials):
        spec = copy.deepcopy(best_spec) if best_spec else NeuralForgeSpec()
        spec.architecture.depth = random.choice([2, 3, 4, 5, 6])
        spec.architecture.width = random.choice([64, 128, 256])
        score = random.gauss(0.5, 0.1)
        all_trials.append({"number": i, "value": score})
        if score > best_score:
            best_score = score
            best_spec = spec
    return OptimizationResult(
        best_spec=best_spec, best_score=best_score, num_trials=max_trials,
        total_time_seconds=0, all_trials=all_trials,
    )


def evaluate_and_report(model, test_loader, detailed=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    nb = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = batch.to(device)
            outputs = model(inputs)
            total_loss += F.cross_entropy(outputs, targets).item()
            nb += 1
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    import numpy as np
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    accuracy = float((preds == targets).mean())
    avg_loss = total_loss / max(nb, 1)
    nc = int(targets.max()) + 1
    confusion = np.zeros((nc, nc), dtype=int)
    for t, p in zip(targets, preds):
        confusion[t][p] += 1
    return EvaluationReport(
        model_name="model",
        metrics={"accuracy": accuracy, "loss": avg_loss, "num_samples": len(targets)},
        confusion_matrix=confusion.tolist(),
    )


def evolve(model_or_spec, generations=20, fitness_fn=None, population_size=10, spec=None):
    if spec is None:
        spec = getattr(model_or_spec, "spec", NeuralForgeSpec())
    if fitness_fn is None:
        fitness_fn = lambda s: -abs((s.architecture.depth or 4) - 4)
    best_spec = spec
    best_fitness = fitness_fn(spec)
    for gen in range(generations):
        for _ in range(population_size):
            mutant = copy.deepcopy(best_spec)
            mutant.architecture.depth = max(1, (mutant.architecture.depth or 4) + random.choice([-1, 1]))
            try:
                fit = fitness_fn(mutant)
                if fit > best_fitness:
                    best_fitness = fit
                    best_spec = mutant
            except Exception:
                continue
    return best_spec
