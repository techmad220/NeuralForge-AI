"""
Neural Architecture Search for Dynamic Model Slicing
Automatically finds optimal model configurations that fit in memory while preserving quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
from pathlib import Path


@dataclass
class ModelArchitecture:
    """Represents a discovered model architecture."""
    layers: List[Dict[str, Any]]
    memory_usage: float  # GB
    theoretical_flops: float
    quality_score: float
    compression_techniques: List[str]


class NeuralArchitectureSearch:
    """
    Evolutionary algorithm to find optimal architectures for memory-constrained inference.
    """

    def __init__(self, target_memory_gb: float = 8.0, target_params: int = 405_000_000_000):
        self.target_memory = target_memory_gb
        self.target_params = target_params
        self.population = []
        self.best_architecture = None
        self.generation = 0

    def create_initial_population(self, size: int = 50) -> List[ModelArchitecture]:
        """Create initial random architectures."""
        population = []

        for _ in range(size):
            arch = self._random_architecture()
            population.append(arch)

        return population

    def _random_architecture(self) -> ModelArchitecture:
        """Generate random architecture that might fit in memory."""
        layers = []
        remaining_params = self.target_params
        layer_count = np.random.randint(80, 120)  # Variable depth

        for i in range(layer_count):
            layer_type = np.random.choice(['attention', 'ffn', 'moe', 'sparse'])

            if layer_type == 'attention':
                layer = {
                    'type': 'attention',
                    'heads': int(np.random.choice([8, 16, 32, 64])),
                    'dim': int(np.random.choice([2048, 4096, 8192])),
                    'compression': np.random.choice(['none', 'linear', 'kernel', 'flash']),
                    'precision': np.random.choice(['fp32', 'fp16', 'int8', 'mixed'])
                }
            elif layer_type == 'ffn':
                layer = {
                    'type': 'ffn',
                    'hidden_dim': int(np.random.choice([8192, 16384, 32768])),
                    'activation': np.random.choice(['relu', 'gelu', 'swish', 'silu']),
                    'compression': np.random.choice(['none', 'factorized', 'sparse']),
                    'precision': np.random.choice(['fp32', 'fp16', 'int8', 'int4'])
                }
            elif layer_type == 'moe':  # Mixture of Experts
                layer = {
                    'type': 'moe',
                    'num_experts': int(np.random.choice([8, 16, 32])),
                    'top_k': int(np.random.choice([1, 2, 4])),
                    'expert_dim': int(np.random.choice([1024, 2048, 4096])),
                    'routing': np.random.choice(['learned', 'random', 'hash'])
                }
            else:  # sparse
                layer = {
                    'type': 'sparse',
                    'sparsity': float(np.random.uniform(0.8, 0.99)),
                    'pattern': np.random.choice(['random', 'structured', 'learned']),
                    'block_size': int(np.random.choice([8, 16, 32, 64]))
                }

            layers.append(layer)

        # Determine compression techniques
        techniques = []
        if np.random.random() > 0.5:
            techniques.append('weight_sharing')
        if np.random.random() > 0.5:
            techniques.append('knowledge_distillation')
        if np.random.random() > 0.5:
            techniques.append('pruning')
        if np.random.random() > 0.5:
            techniques.append('quantization')

        arch = ModelArchitecture(
            layers=layers,
            memory_usage=self._estimate_memory(layers, techniques),
            theoretical_flops=self._estimate_flops(layers),
            quality_score=self._estimate_quality(layers, techniques),
            compression_techniques=techniques
        )

        return arch

    def _estimate_memory(self, layers: List[Dict], techniques: List[str]) -> float:
        """Estimate memory usage in GB."""
        total_params = 0

        for layer in layers:
            if layer['type'] == 'attention':
                # Attention has Q, K, V, O projections
                dim = layer['dim']
                params = 4 * dim * dim
                if layer['compression'] == 'linear':
                    params *= 0.5
                elif layer['compression'] == 'kernel':
                    params *= 0.1
                total_params += params

            elif layer['type'] == 'ffn':
                hidden = layer['hidden_dim']
                params = 2 * 8192 * hidden  # Assuming base dim of 8192
                if layer['compression'] == 'factorized':
                    params *= 0.3
                elif layer['compression'] == 'sparse':
                    params *= 0.1
                total_params += params

            elif layer['type'] == 'moe':
                # Only active experts count toward memory
                active_params = layer['top_k'] * layer['expert_dim'] * 8192
                total_params += active_params

            elif layer['type'] == 'sparse':
                # Sparse layers use less memory
                base_params = 8192 * 8192
                total_params += base_params * (1 - layer['sparsity'])

        # Apply compression techniques
        compression_factor = 1.0
        if 'weight_sharing' in techniques:
            compression_factor *= 0.7
        if 'pruning' in techniques:
            compression_factor *= 0.5
        if 'quantization' in techniques:
            compression_factor *= 0.25

        # Calculate memory in GB (4 bytes per param for FP32)
        memory_gb = (total_params * compression_factor * 4) / (1024**3)
        return memory_gb

    def _estimate_flops(self, layers: List[Dict]) -> float:
        """Estimate theoretical FLOPS."""
        total_flops = 0

        for layer in layers:
            if layer['type'] == 'attention':
                # O(n^2 * d) for attention
                seq_len = 2048  # Assume sequence length
                flops = seq_len * seq_len * layer['dim']
                if layer['compression'] == 'flash':
                    flops *= 0.5  # Flash attention is more efficient
                total_flops += flops

            elif layer['type'] == 'ffn':
                flops = 2 * 8192 * layer['hidden_dim']
                total_flops += flops

            elif layer['type'] == 'moe':
                # Only active experts compute
                flops = layer['top_k'] * layer['expert_dim'] * 8192 * 2
                total_flops += flops

            elif layer['type'] == 'sparse':
                base_flops = 2 * 8192 * 8192
                total_flops += base_flops * (1 - layer['sparsity'])

        return total_flops

    def _estimate_quality(self, layers: List[Dict], techniques: List[str]) -> float:
        """Estimate model quality (0-1 score)."""
        quality = 1.0

        # Penalize aggressive compression
        for layer in layers:
            if layer.get('precision') == 'int4':
                quality *= 0.95
            elif layer.get('precision') == 'int8':
                quality *= 0.98

            if layer.get('type') == 'sparse':
                quality *= (1 - layer['sparsity'] * 0.5)  # High sparsity hurts quality

        # Compression techniques impact quality
        if 'pruning' in techniques:
            quality *= 0.9
        if 'quantization' in techniques:
            quality *= 0.95
        if 'knowledge_distillation' in techniques:
            quality *= 1.05  # Distillation can improve quality

        # Bonus for good architectures
        moe_layers = sum(1 for l in layers if l['type'] == 'moe')
        if moe_layers > 10:
            quality *= 1.1  # MoE is good for large models

        return min(1.0, quality)

    def evolve_population(self, population: List[ModelArchitecture]) -> List[ModelArchitecture]:
        """Evolve population using genetic algorithm."""
        # Sort by fitness (quality / memory ratio)
        population.sort(key=lambda x: x.quality_score / max(1, x.memory_usage / self.target_memory), reverse=True)

        # Keep top 20%
        elite_size = len(population) // 5
        new_population = population[:elite_size]

        # Generate offspring
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            if np.random.random() < 0.3:
                child = self._mutate(child)

            new_population.append(child)

        return new_population

    def _tournament_select(self, population: List[ModelArchitecture], k: int = 5) -> ModelArchitecture:
        """Tournament selection."""
        tournament = np.random.choice(population, k)
        return max(tournament, key=lambda x: x.quality_score / max(1, x.memory_usage / self.target_memory))

    def _crossover(self, parent1: ModelArchitecture, parent2: ModelArchitecture) -> ModelArchitecture:
        """Crossover two architectures."""
        # Mix layers from both parents
        crossover_point = np.random.randint(0, min(len(parent1.layers), len(parent2.layers)))

        child_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]

        # Mix compression techniques
        child_techniques = list(set(parent1.compression_techniques + parent2.compression_techniques))
        if len(child_techniques) > 3:
            child_techniques = np.random.choice(child_techniques, 3, replace=False).tolist()

        child = ModelArchitecture(
            layers=child_layers,
            memory_usage=self._estimate_memory(child_layers, child_techniques),
            theoretical_flops=self._estimate_flops(child_layers),
            quality_score=self._estimate_quality(child_layers, child_techniques),
            compression_techniques=child_techniques
        )

        return child

    def _mutate(self, architecture: ModelArchitecture) -> ModelArchitecture:
        """Mutate an architecture."""
        mutated_layers = architecture.layers.copy()

        # Randomly mutate a layer
        if mutated_layers:
            idx = np.random.randint(0, len(mutated_layers))
            layer = mutated_layers[idx].copy()

            # Mutate a random property
            if layer['type'] == 'attention':
                if np.random.random() < 0.5:
                    layer['heads'] = int(np.random.choice([8, 16, 32, 64]))
                else:
                    layer['compression'] = np.random.choice(['none', 'linear', 'kernel', 'flash'])
            elif layer['type'] == 'sparse':
                layer['sparsity'] = float(np.clip(layer['sparsity'] + np.random.uniform(-0.1, 0.1), 0.5, 0.99))

            mutated_layers[idx] = layer

        # Maybe add/remove compression technique
        techniques = architecture.compression_techniques.copy()
        if np.random.random() < 0.3:
            new_technique = np.random.choice(['weight_sharing', 'knowledge_distillation', 'pruning', 'quantization'])
            if new_technique not in techniques:
                techniques.append(new_technique)

        mutated = ModelArchitecture(
            layers=mutated_layers,
            memory_usage=self._estimate_memory(mutated_layers, techniques),
            theoretical_flops=self._estimate_flops(mutated_layers),
            quality_score=self._estimate_quality(mutated_layers, techniques),
            compression_techniques=techniques
        )

        return mutated

    def search(self, generations: int = 100) -> ModelArchitecture:
        """Run neural architecture search."""
        print(f"[NAS] Starting search for {self.target_params/1e9:.0f}B model in {self.target_memory}GB")

        # Initialize population
        self.population = self.create_initial_population(50)

        for gen in range(generations):
            # Evolve
            self.population = self.evolve_population(self.population)

            # Track best
            best = max(self.population, key=lambda x: x.quality_score if x.memory_usage <= self.target_memory else 0)

            if self.best_architecture is None or best.quality_score > self.best_architecture.quality_score:
                if best.memory_usage <= self.target_memory:
                    self.best_architecture = best

            if gen % 10 == 0:
                print(f"[NAS] Generation {gen}: Best memory={best.memory_usage:.2f}GB, "
                      f"quality={best.quality_score:.3f}")

            self.generation = gen

        # Ensure we have a result even if nothing fits perfectly
        if self.best_architecture is None:
            # Return the smallest architecture as fallback
            self.best_architecture = min(self.population, key=lambda x: x.memory_usage)
            print(f"[NAS] No architecture fits target, using smallest: {self.best_architecture.memory_usage:.2f}GB")
        else:
            print(f"[NAS] Search complete! Found architecture with {self.best_architecture.memory_usage:.2f}GB "
                  f"and quality score {self.best_architecture.quality_score:.3f}")

        return self.best_architecture

    def save_architecture(self, architecture: ModelArchitecture, path: Path):
        """Save discovered architecture."""
        data = {
            'layers': architecture.layers,
            'memory_usage_gb': float(architecture.memory_usage),
            'theoretical_flops': float(architecture.theoretical_flops),
            'quality_score': float(architecture.quality_score),
            'compression_techniques': architecture.compression_techniques,
            'target_params': int(self.target_params),
            'target_memory_gb': float(self.target_memory)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[NAS] Architecture saved to {path}")


class DynamicModelSlicer:
    """
    Slice model dynamically based on NAS-discovered architecture.
    """

    def __init__(self, architecture: ModelArchitecture):
        self.architecture = architecture
        self.active_slices = {}
        self.slice_schedule = self._create_schedule()

    def _create_schedule(self) -> Dict[int, List[int]]:
        """Create execution schedule for model slices."""
        schedule = {}
        time_step = 0

        for i, layer in enumerate(self.architecture.layers):
            # Determine when this layer should be active
            if layer['type'] == 'moe':
                # MoE layers are activated conditionally
                schedule[time_step] = [i]
                time_step += 1
            elif layer.get('compression') == 'flash':
                # Flash attention layers can share time slots
                if time_step not in schedule:
                    schedule[time_step] = []
                schedule[time_step].append(i)
                if len(schedule[time_step]) >= 4:
                    time_step += 1
            else:
                # Regular layers get their own time slot
                schedule[time_step] = [i]
                time_step += 1

        return schedule

    def get_active_slice(self, input_data: np.ndarray, time_step: int) -> List[Dict]:
        """Get layers active at current time step."""
        layer_indices = self.slice_schedule.get(time_step % len(self.slice_schedule), [])
        return [self.architecture.layers[i] for i in layer_indices]

    def route_computation(self, input_data: np.ndarray) -> List[int]:
        """Determine which layers to execute for given input."""
        active_layers = []

        for i, layer in enumerate(self.architecture.layers):
            if layer['type'] == 'moe':
                # Route to specific experts based on input
                input_hash = hash(input_data.tobytes())
                if input_hash % 3 == 0:  # Simple routing logic
                    active_layers.append(i)
            elif layer.get('type') == 'sparse':
                # Skip sparse layers if input is dense
                if np.mean(np.abs(input_data) > 0.01) > layer['sparsity']:
                    active_layers.append(i)
            else:
                active_layers.append(i)

        return active_layers