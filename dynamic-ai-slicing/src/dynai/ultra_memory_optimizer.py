"""
Ultra Memory Optimizer for Dynamic AI Slicing
Implements extreme memory optimization techniques for running huge models on tiny hardware.
"""

from __future__ import annotations

import gc
import json
import mmap
import os
import pickle
import sys
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import weakref

import numpy as np


class MemoryPressureMonitor:
    """Monitors system memory and triggers cleanup when pressure is high."""
    
    def __init__(self, max_memory_mb: int = 512, check_interval: float = 1.0):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.check_interval = check_interval
        self.running = False
        self._thread = None
        self.cleanup_callbacks = []
    
    def add_cleanup_callback(self, callback):
        """Add a callback to be called when memory pressure is high."""
        self.cleanup_callbacks.append(weakref.ref(callback))
    
    def get_memory_usage(self) -> int:
        """Get current process memory usage in bytes."""
        try:
            # On Linux/Android, read from /proc/self/status
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # Extract memory in kB and convert to bytes
                        return int(line.split()[1]) * 1024
        except:
            # Fallback: estimate from gc stats and object count
            return len(gc.get_objects()) * 64  # rough estimate
        return 0
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            memory_usage = self.get_memory_usage()
            if memory_usage > self.max_memory_bytes:
                self._trigger_cleanup(memory_usage)
            time.sleep(self.check_interval)
    
    def _trigger_cleanup(self, memory_usage: int):
        """Trigger cleanup callbacks when memory pressure is high."""
        print(f"[MemoryMonitor] High memory usage: {memory_usage // 1024 // 1024}MB")
        
        # Call cleanup callbacks
        for callback_ref in self.cleanup_callbacks[:]:
            callback = callback_ref()
            if callback is None:
                self.cleanup_callbacks.remove(callback_ref)
            else:
                try:
                    callback()
                except Exception as e:
                    print(f"[MemoryMonitor] Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()


class CompressedLayerCache:
    """LRU cache with compressed storage for neural network layers."""
    
    def __init__(self, max_size: int = 10, compression_level: int = 6):
        self.max_size = max_size
        self.compression_level = compression_level
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = threading.RLock()
    
    def _compress_array(self, arr: np.ndarray) -> bytes:
        """Compress numpy array using zlib + pickle."""
        data = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(data, level=self.compression_level)
    
    def _decompress_array(self, data: bytes) -> np.ndarray:
        """Decompress numpy array."""
        uncompressed = zlib.decompress(data)
        return pickle.loads(uncompressed)
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get array from cache, moving to end (most recent)."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                compressed = self.cache.pop(key)
                self.cache[key] = compressed
                self.stats["hits"] += 1
                return self._decompress_array(compressed)
            else:
                self.stats["misses"] += 1
                return None
    
    def put(self, key: str, arr: np.ndarray) -> None:
        """Store array in cache with compression."""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict least recently used if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove first (oldest)
                self.stats["evictions"] += 1
            
            # Store compressed array
            compressed = self._compress_array(arr)
            self.cache[key] = compressed
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_size
            }


class QuantizedStorage:
    """Store neural network weights with quantization to save memory."""
    
    @staticmethod
    def quantize_weights(W: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
        """Quantize weights to specified bits."""
        if bits == 8:
            dtype = np.int8
            max_val = 127
        elif bits == 4:
            dtype = np.int8  # Store as int8 but only use 4 bits
            max_val = 7
        else:
            raise ValueError(f"Unsupported quantization: {bits} bits")
        
        W_min, W_max = W.min(), W.max()
        scale = (W_max - W_min) / (2 * max_val)
        
        if scale == 0:
            # All values are the same
            return np.zeros(W.shape, dtype=dtype), W_min, 0.0
        
        W_quantized = np.clip(
            np.round((W - W_min) / scale - max_val),
            -max_val, max_val
        ).astype(dtype)
        
        return W_quantized, W_min, scale
    
    @staticmethod
    def dequantize_weights(W_q: np.ndarray, offset: float, scale: float) -> np.ndarray:
        """Dequantize weights back to float32."""
        if scale == 0:
            return np.full(W_q.shape, offset, dtype=np.float32)
        
        return ((W_q.astype(np.float32) + 127) * scale + offset).astype(np.float32)
    
    @staticmethod
    def save_quantized(path: Path, W: np.ndarray, quantize_bits: int = 8) -> None:
        """Save weights in quantized + compressed format."""
        W_q, offset, scale = QuantizedStorage.quantize_weights(W, quantize_bits)
        
        data = {
            'weights': W_q,
            'offset': offset,
            'scale': scale,
            'original_shape': W.shape,
            'bits': quantize_bits
        }
        
        # Compress the entire data structure
        compressed = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        path.write_bytes(compressed)
    
    @staticmethod
    def load_quantized(path: Path) -> np.ndarray:
        """Load and dequantize weights."""
        compressed = path.read_bytes()
        data = pickle.loads(zlib.decompress(compressed))
        
        return QuantizedStorage.dequantize_weights(
            data['weights'], data['offset'], data['scale']
        )


class UltraMemoryOptimizer:
    """Ultra-aggressive memory optimizer for huge models on tiny hardware."""
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        cache_size: int = 5,
        quantize_bits: int = 8,
        compression_level: int = 6
    ):
        self.cache = CompressedLayerCache(cache_size, compression_level)
        self.memory_monitor = MemoryPressureMonitor(max_memory_mb)
        self.quantize_bits = quantize_bits
        
        # Register cleanup callback
        self.memory_monitor.add_cleanup_callback(self._emergency_cleanup)
        self.memory_monitor.start_monitoring()
        
        # Memory-mapped file handles (keep refs to prevent GC)
        self._mmap_handles = {}
    
    def _emergency_cleanup(self):
        """Emergency cleanup when memory pressure is critical."""
        print("[UltraOptimizer] Emergency cleanup triggered!")
        
        # Clear cache
        self.cache.clear()
        
        # Close memory-mapped files
        for handle in self._mmap_handles.values():
            try:
                if hasattr(handle, 'close'):
                    handle.close()
            except:
                pass
        self._mmap_handles.clear()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
    
    def create_optimized_model(
        self,
        outdir: str,
        d_in: int,
        hidden: list[int],
        d_out: int,
        seed: int = 1
    ) -> None:
        """Create model with ultra-optimized storage."""
        rng = np.random.default_rng(seed)
        dims = [d_in] + hidden + [d_out]
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        
        total_params = 0
        for i in range(len(dims) - 1):
            # Generate weights
            W = rng.standard_normal((dims[i], dims[i + 1]), dtype=np.float32)
            W *= (1.0 / np.sqrt(dims[i]))
            b = np.zeros((dims[i + 1],), dtype=np.float32)
            
            total_params += W.size + b.size
            
            # Save with quantization and compression
            QuantizedStorage.save_quantized(
                out / f"layer{i}_w.qz", W, self.quantize_bits
            )
            QuantizedStorage.save_quantized(
                out / f"layer{i}_b.qz", b, self.quantize_bits
            )
            
            # Force cleanup after each layer to prevent memory buildup
            del W, b
            gc.collect()
        
        # Save metadata with optimization info
        meta = {
            "d_in": d_in,
            "hidden": hidden,
            "d_out": d_out,
            "layers": len(dims) - 1,
            "activation": "relu",
            "output": "logits",
            "optimization": {
                "quantize_bits": self.quantize_bits,
                "compressed": True,
                "total_params": total_params
            }
        }
        
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def load_layer_ultra_optimized(
        self,
        model_dir: str,
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load layer with all optimizations: cache, mmap, compression."""
        cache_key = f"{model_dir}:layer{layer_idx}"
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            W, b = cached_data
            return W.copy(), b.copy()  # Return copies to prevent modification
        
        # Load from disk with memory mapping if possible
        base_path = Path(model_dir)
        w_path = base_path / f"layer{layer_idx}_w.qz"
        b_path = base_path / f"layer{layer_idx}_b.qz"
        
        try:
            # Try quantized data first
            if w_path.exists() and b_path.exists():
                W = QuantizedStorage.load_quantized(w_path)
                b = QuantizedStorage.load_quantized(b_path)
            else:
                # Fallback to regular .npy files
                w_path_npy = base_path / f"layer{layer_idx}_w.npy"
                b_path_npy = base_path / f"layer{layer_idx}_b.npy"
                
                if w_path_npy.exists() and b_path_npy.exists():
                    print(f"[UltraOptimizer] Loading regular .npy files for layer {layer_idx}")
                    W = np.load(w_path_npy, mmap_mode="r").copy()
                    b = np.load(b_path_npy, mmap_mode="r").copy()
                else:
                    raise FileNotFoundError(f"No quantized or regular files found for layer {layer_idx}")
            
            # Cache the loaded data (compressed)
            self.cache.put(cache_key, (W, b))
            
            return W, b
            
        except Exception as e:
            print(f"[UltraOptimizer] Failed to load layer {layer_idx}: {e}")
            raise
    
    def streaming_inference_ultra(
        self,
        model_dir: str,
        x: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Ultra-optimized streaming inference with micro-batching."""
        with open(Path(model_dir) / "metadata.json") as f:
            meta = json.load(f)
        
        layers = meta["layers"]
        
        # Handle single sample vs batch
        if x.ndim == 1:
            # Single sample - add batch dimension
            x_batch = x.reshape(1, -1)
            result = self._forward_single_batch(model_dir, x_batch, layers)
            return result.flatten()  # Remove batch dimension
        else:
            # Multiple samples
            if batch_size and x.shape[0] > batch_size:
                results = []
                for i in range(0, x.shape[0], batch_size):
                    batch = x[i:i + batch_size]
                    batch_result = self._forward_single_batch(model_dir, batch, layers)
                    results.append(batch_result)
                    
                    # Cleanup after each batch
                    gc.collect()
                
                return np.concatenate(results, axis=0)
            else:
                return self._forward_single_batch(model_dir, x, layers)
    
    def _forward_single_batch(
        self,
        model_dir: str,
        x: np.ndarray,
        layers: int
    ) -> np.ndarray:
        """Forward pass for a single batch with minimal memory footprint."""
        h = x.astype(np.float32)
        
        # Gradient checkpointing simulation: only keep every Nth activation
        checkpoint_interval = max(1, layers // 3)  # Keep ~3 checkpoints max
        checkpoints = {}
        
        for i in range(layers):
            # Load layer (from cache if available)
            W, b = self.load_layer_ultra_optimized(model_dir, i)
            
            # Compute layer output
            h_new = np.dot(h, W).astype(np.float32)
            h_new += b
            
            # Apply activation (in-place to save memory)
            if i < layers - 1:  # Not the last layer
                np.maximum(h_new, 0.0, out=h_new)  # ReLU in-place
            
            # Gradient checkpointing: save intermediate results strategically
            if i % checkpoint_interval == 0 and i > 0:
                # Only keep a compressed version
                checkpoints[i] = self.cache._compress_array(h.copy())
                
                # Clear old checkpoints to save memory
                keys_to_remove = [k for k in checkpoints.keys() if k < i - checkpoint_interval]
                for k in keys_to_remove:
                    del checkpoints[k]
            
            # Replace h with new output (frees old h memory)
            h = h_new
            
            # Explicitly delete large arrays to hint GC
            del W, b
            
            # Micro-cleanup to prevent buildup
            if i % 2 == 0:  # Every other layer
                gc.collect()
        
        # Clear any remaining checkpoints
        checkpoints.clear()
        
        return h
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and performance statistics."""
        memory_usage = self.memory_monitor.get_memory_usage()
        cache_stats = self.cache.get_stats()
        
        return {
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "memory_limit_mb": self.memory_monitor.max_memory_bytes / (1024 * 1024),
            "cache": cache_stats,
            "optimizations": {
                "quantize_bits": self.quantize_bits,
                "cache_compression": True,
                "memory_monitoring": True,
                "garbage_collection": True
            }
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.memory_monitor.stop_monitoring()
            self._emergency_cleanup()
        except:
            pass