"""data.py

Dataset class for collecting spectrograms.

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Dataset for spectrogram `.pt` files with integer labels (0,1,...)."""

    def __init__(
        self,
        csv_file: str | Path,
        transform=None,
        preload: bool = False,
    ) -> None:
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        df = pd.read_csv(self.csv_file)
        if 'spec_dir' not in df.columns or 'class' not in df.columns:
            raise ValueError("CSV must contain columns 'spec_dir' and 'class'")

        self.paths: List[Path] = [Path(p) for p in df['spec_dir'].tolist()]
        self.labels: List[int] = df['class'].astype(int).tolist()

        self.transform = transform
        self.preload = preload

        # Optionally preload tensors into memory
        self._cache: Optional[List[torch.Tensor]] = None
        if self.preload:
            self._cache = []
            for p in self.paths:
                if not p.exists():
                    raise FileNotFoundError(f"Spectrogram file not found: {p}")
                t = torch.load(p)
                if not torch.is_tensor(t):
                    raise TypeError(f"Loaded object is not a tensor: {p}")
                t = t.unsqueeze(0).float()
                self._cache.append(t)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            spec = self._cache[idx]
        else:
            p = self.paths[idx]
            if not p.exists():
                raise FileNotFoundError(f"Spectrogram file not found: {p}")
            spec = torch.load(p)
            if not torch.is_tensor(spec):
                raise TypeError(f"Loaded object is not a tensor: {p}")
            spec = spec.unsqueeze(0).float()  # [1, H, W]

        if self.transform is not None:
            spec = self.transform(spec)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


# --- simple CLI ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build SpectrogramDataset')
    parser.add_argument('--csv', type=str, default=r'data/labels/final_labels.csv')
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()

    print(f'Loading dataset from: {args.csv}')
    ds = SpectrogramDataset(csv_file=args.csv, preload=args.preload)
    print(f'Num samples: {len(ds)}')
    print(f'Num classes: {ds.num_classes}')
