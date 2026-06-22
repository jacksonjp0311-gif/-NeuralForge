from __future__ import annotations
import json, os, shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from neuralforge.spec import NeuralForgeSpec

class ModelRecord:
    def __init__(self, name, spec, model_path=None, metadata=None):
        self.name = name; self.spec = spec; self.model_path = model_path
        self.metadata = metadata or {}; self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow(); self.version = spec.version
        self.config_hash = spec.config_hash()
    def to_dict(self):
        return {"name": self.name, "version": self.version, "config_hash": self.config_hash,
                "model_path": self.model_path, "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(), "metadata": self.metadata}

class ModelRegistry:
    _instance = None; _initialized = False
    def __init__(self, registry_dir="./neuralforge_registry"):
        if ModelRegistry._initialized: return
        self._registry_dir = Path(registry_dir); self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelRecord] = {}
        self._index_file = self._registry_dir / "index.json"; self._load_index()
        ModelRegistry._initialized = True
    @classmethod
    def get_instance(cls, registry_dir="./neuralforge_registry"):
        if cls._instance is None: cls._instance = cls(registry_dir)
        return cls._instance
    @classmethod
    def reset(cls): cls._instance = None; cls._initialized = False
    def register(self, name, spec, model_path=None, metadata=None):
        record = ModelRecord(name, spec, model_path, metadata)
        key = f"{name}:{record.config_hash}"; self._models[key] = record; self._save_index(); return record
    def get(self, name, config_hash=None):
        if config_hash: return self._models.get(f"{name}:{config_hash}")
        matches = [r for k, r in self._models.items() if k.startswith(f"{name}:")]
        return max(matches, key=lambda r: r.updated_at) if matches else None
    def list_models(self): return [r.to_dict() for r in self._models.values()]
    def delete(self, name, config_hash=None):
        if config_hash:
            key = f"{name}:{config_hash}"
            if key in self._models:
                record = self._models.pop(key)
                if record.model_path:
                    p = Path(record.model_path)
                    if p.exists(): shutil.rmtree(p) if p.is_dir() else p.unlink()
                self._save_index(); return True
            return False
        keys = [k for k in self._models if k.startswith(f"{name}:")]
        for key in keys:
            record = self._models.pop(key)
            if record.model_path:
                p = Path(record.model_path)
                if p.exists(): shutil.rmtree(p) if p.is_dir() else p.unlink()
        self._save_index(); return len(keys) > 0
    def _save_index(self):
        data = {k: v.to_dict() for k, v in self._models.items()}
        with open(self._index_file, "w") as f: json.dump(data, f, indent=2, default=str)
    def _load_index(self):
        if self._index_file.exists():
            try:
                with open(self._index_file) as f: self._metadata_cache = json.load(f)
            except: self._metadata_cache = {}
        else: self._metadata_cache = {}
