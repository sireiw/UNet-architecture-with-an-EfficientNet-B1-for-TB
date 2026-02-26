import random
import math
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Iterator, TypeVar
from .config import Config

T_co = TypeVar('T_co', covariant=True)

class TwoStreamBatchSampler(Sampler[List[int]]):
    """Sampler that guarantees a specific ratio of synthetic vs real samples per batch."""
    
    def __init__(self, idx_syn: List[int], idx_real: List[int], batch_size: int, ratio_syn: float = 0.5, drop_last: bool = False): 
        """
        Initialize the two-stream balanced random sampler.
        
        Args:
            idx_syn (List[int]): Complete pool of synthetic generator inference indices.
            idx_real (List[int]): Pool of real hospital/clinical data indices.
            batch_size (int): Expected step length parameter.
            ratio_syn (float, optional): Exact distribution weighting of synthetic samples. Defaults to 0.5.
            drop_last (bool, optional): Whether to cull incomplete batches. Defaults to False.
        """
        self.syn = idx_syn
        self.real = idx_real
        self.bs = batch_size
        self.k_syn = int(round(self.bs * ratio_syn))
        self.k_real = self.bs - self.k_syn
        self.drop_last = drop_last
        
        # Ensure we have at least one of each, if possible
        if self.k_syn == 0 and len(self.real) > 0:
            self.k_syn = 1
            self.k_real = self.bs - 1
        if self.k_real == 0 and len(self.syn) > 0:
            self.k_real = 1
            self.k_syn = self.bs - 1

        # Handle very small real-world sets
        if len(self.real) < self.k_real:
            self.k_real = len(self.real)
            self.k_syn = self.bs - self.k_real
        
        print((f"\n📊 TwoStreamBatchSampler initialized:\n"
               f"  Batch size: {self.bs}\n"
               f"  Synthetic per batch: {self.k_syn}\n"
               f"  Real per batch: {self.k_real}\n"
               f"  Ratio: {self.k_syn / self.bs:.1%} synthetic"))
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yields balanced mixed indices randomly."""
        syn = random.sample(self.syn, len(self.syn))
        real = random.sample(self.real, len(self.real))
        
        # Handle case where real set is too small to ever fill a batch
        if len(self.real) == 0:
             real_indices = []
        else:
            real_indices = (real * math.ceil(len(syn) / len(real)))[:len(syn)]
            random.shuffle(real_indices)

        syn_indices = syn
        random.shuffle(syn_indices)

        i_syn = i_real = 0
        
        while True:
            batch_syn_indices = []
            for _ in range(self.k_syn):
                batch_syn_indices.append(syn_indices[i_syn % len(syn_indices)])
                i_syn += 1

            batch_real_indices = []
            if len(real_indices) > 0:
                for _ in range(self.k_real):
                    batch_real_indices.append(real_indices[i_real % len(real_indices)])
                    i_real += 1
            
            batch = batch_syn_indices + batch_real_indices
            random.shuffle(batch)
            yield batch
            
            # Stop condition
            if i_syn >= len(syn_indices) and i_real >= len(real_indices):
                break
        
    def __len__(self) -> int:
        if self.k_syn == 0 or len(self.syn) == 0:
             return math.floor(len(self.real) / self.k_real) if self.k_real > 0 else 0
        if self.k_real == 0 or len(self.real) == 0:
            return math.floor(len(self.syn) / self.k_syn) if self.k_syn > 0 else 0
            
        return math.floor(min(len(self.syn)/self.k_syn, len(self.real)/self.k_real))

def create_balanced_real_indices(mixed_dataset: Dataset, real_indices_base: List[int], config: Config) -> List[int]:
    """Create truly balanced real indices using statistical oversampling over the base pool.
    
    Args:
        mixed_dataset (Dataset): Complete data loader source.
        real_indices_base (List[int]): Sub-pool identifying true actuals vs synthetics.
        config (Config): System parameters mapping num_classes and target labels.
        
    Returns:
        List[int]: An oversampled, fully flat balanced distribution of valid indices.
    """
    real_buckets = {c: [] for c in range(config.NUM_CLASSES)}
    for idx in real_indices_base:
        label = mixed_dataset.labels[idx]
        if isinstance(label, str):
            label = mixed_dataset.label_map.get(label.lower().replace(' ', '_'), 0)
        real_buckets[label].append(idx)
    
    # Find max count
    max_count = max(len(bucket) for bucket in real_buckets.values())
    
    # Oversample ALL classes to max_count (not just balance)
    # Use higher factor for TB specifically
    tb_idx = config.CLASS_NAMES.index('tuberculosis')
    target_counts = {}
    for class_idx in range(config.NUM_CLASSES):
        if class_idx == tb_idx:
            target_counts[class_idx] = int(max_count * 1.5)  # 50% more TB
        else:
            target_counts[class_idx] = max_count
    
    balanced_indices = []
    for class_idx in range(config.NUM_CLASSES):
        bucket = real_buckets[class_idx]
        if len(bucket) == 0:
            continue
        
        target = target_counts[class_idx]
        repeats = target // len(bucket)
        remainder = target % len(bucket)
        
        balanced_indices.extend(bucket * repeats)
        if remainder > 0:
            balanced_indices.extend(np.random.choice(bucket, remainder, replace=False).tolist())
    
    print(f"\n📊 Balanced real stream (with TB boost):")
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        count = sum(1 for idx in balanced_indices 
                   if mixed_dataset.labels[idx] == class_name or 
                   mixed_dataset.label_map.get(str(mixed_dataset.labels[idx]).lower().replace(' ', '_'), -1) == class_idx)
        print(f"  {class_name}: {count}")
    
    return balanced_indices
