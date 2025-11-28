"""
PyTorch Dataset for Macrostrat geological data.
Fetches real stratigraphic data and converts to training format.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import json

from ..data_generation.macrostrat_fetcher import (
    MacrostratFetcher, StratColumn, StratUnit,
    BASIN_COORDINATES, TARGET_FORMATIONS,
    LITHOLOGY_RESISTIVITY, LITHOLOGY_POROSITY
)


class MacrostratDataset(Dataset):
    """
    PyTorch Dataset that fetches and processes Macrostrat geological data.
    
    Generates training samples from real stratigraphic columns with:
    - Cross-section grids with petrophysical properties
    - Sparse sampling masks simulating borehole data
    - Multiple realizations from the same geological structure
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: Tuple[int, int] = (32, 64),
        basins: Optional[List[str]] = None,
        formations: Optional[List[str]] = None,
        property_type: str = 'resistivity',
        sparse_rate: float = 0.05,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        num_realizations: int = 5,
        augment: bool = True,
        max_depth_m: float = 3000,
        seed: int = 42,
        offline_mode: bool = False
    ):
        """
        Args:
            num_samples: Number of training samples to generate
            grid_size: (height, width) of output grids
            basins: List of basin names to query (uses all if None)
            formations: List of target formations (uses all if None)
            property_type: 'resistivity' or 'porosity'
            sparse_rate: Fraction of grid points to sample
            cache_dir: Directory to cache fetched data
            use_cache: Whether to use cached data
            num_realizations: Number of stochastic realizations per column
            augment: Apply data augmentation
            max_depth_m: Maximum depth in meters
            seed: Random seed
            offline_mode: Use only cached data, don't fetch
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.property_type = property_type
        self.sparse_rate = sparse_rate
        self.cache_dir = cache_dir or "macrostrat_cache"
        self.use_cache = use_cache
        self.num_realizations = num_realizations
        self.augment = augment
        self.max_depth_m = max_depth_m
        self.offline_mode = offline_mode
        
        self.basins = basins or list(BASIN_COORDINATES.keys())
        self.formations = formations or TARGET_FORMATIONS
        
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # Setup cache
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize fetcher
        self.fetcher = MacrostratFetcher()
        
        # Fetch/load geological data
        self.columns = self._load_or_fetch_data()
        
        # Generate samples
        self.samples = self._generate_samples()
        
        print(f"MacrostratDataset initialized with {len(self.samples)} samples")
    
    def _load_or_fetch_data(self) -> List[StratColumn]:
        """Load from cache or fetch from API."""
        cache_file = os.path.join(self.cache_dir, "columns_cache.pkl")
        
        if self.use_cache and os.path.exists(cache_file):
            print("Loading cached Macrostrat data...")
            with open(cache_file, 'rb') as f:
                columns = pickle.load(f)
            print(f"Loaded {len(columns)} columns from cache")
            return columns
        
        if self.offline_mode:
            print("Offline mode - no cached data available, generating synthetic fallback")
            return []
        
        print("Fetching data from Macrostrat API...")
        columns = []
        
        # Fetch from each basin
        for basin_key in tqdm(self.basins, desc="Fetching basins"):
            if basin_key not in BASIN_COORDINATES:
                continue
            
            coords = BASIN_COORDINATES[basin_key]
            try:
                basin_columns = self.fetcher.query_columns_by_location(
                    coords["lat"], coords["lng"], radius_km=100
                )
                
                # Get detailed units for each column
                for col in basin_columns[:10]:  # Limit per basin
                    units = self.fetcher.query_units_by_column(col.col_id)
                    if units:
                        col.units = units
                        columns.append(col)
                
            except Exception as e:
                print(f"Error fetching {basin_key}: {e}")
        
        # Fetch by formation
        for formation in tqdm(self.formations[:5], desc="Fetching formations"):
            try:
                units = self.fetcher.query_units_by_formation(formation)
                # Group units by column
                col_units = {}
                for unit in units[:50]:  # Limit
                    if unit.col_id:
                        if unit.col_id not in col_units:
                            col_units[unit.col_id] = []
                        col_units[unit.col_id].append(unit)
                
                # Create synthetic columns
                for col_id, unit_list in col_units.items():
                    if unit_list:
                        col = StratColumn(
                            col_id=col_id,
                            col_name=f"{formation}_col_{col_id}",
                            lat=unit_list[0].lat or 0,
                            lng=unit_list[0].lng or 0,
                            col_area=0,
                            max_thick=sum(u.thickness or 100 for u in unit_list),
                            units=unit_list
                        )
                        columns.append(col)
            except Exception as e:
                print(f"Error fetching formation {formation}: {e}")
        
        print(f"Fetched {len(columns)} total columns")
        
        # Cache the data
        if self.use_cache and columns:
            with open(cache_file, 'wb') as f:
                pickle.dump(columns, f)
            print(f"Cached data to {cache_file}")
        
        return columns
    
    def _generate_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate training samples from columns."""
        samples = []
        
        if not self.columns:
            print("No Macrostrat data - generating synthetic geological samples")
            return self._generate_synthetic_samples()
        
        # Filter columns with sufficient units
        valid_columns = [c for c in self.columns if len(c.units) >= 2]
        
        if not valid_columns:
            print("No valid columns - generating synthetic samples")
            return self._generate_synthetic_samples()
        
        print(f"Generating {self.num_samples} samples from {len(valid_columns)} columns...")
        
        samples_per_column = max(1, self.num_samples // len(valid_columns))
        
        for col in tqdm(valid_columns, desc="Building samples"):
            for realization in range(min(samples_per_column, self.num_realizations)):
                try:
                    sample = self._build_sample_from_column(col, realization)
                    if sample is not None:
                        samples.append(sample)
                        
                        # Add augmented versions
                        if self.augment and len(samples) < self.num_samples:
                            aug_sample = self._augment_sample(sample)
                            samples.append(aug_sample)
                    
                    if len(samples) >= self.num_samples:
                        break
                except Exception as e:
                    pass
            
            if len(samples) >= self.num_samples:
                break
        
        # Fill remaining with synthetic if needed
        while len(samples) < self.num_samples:
            synthetic = self._generate_synthetic_sample(len(samples))
            samples.append(synthetic)
        
        return samples[:self.num_samples]
    
    def _build_sample_from_column(self, column: StratColumn, 
                                   realization: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Build a training sample from a single column."""
        height, width = self.grid_size
        
        # Create grid
        target = np.zeros((height, width), dtype=np.float32)
        
        # Sort units by age (youngest/shallowest first)
        units_sorted = sorted(column.units, key=lambda u: u.t_age)
        
        # Build 1D column profile first
        profile = []
        current_depth = 0
        
        for unit in units_sorted:
            thickness = unit.thickness or unit.max_thick or 100
            
            # Stochastic variation for different realizations
            seed = hash(f"{column.col_id}_{unit.unit_id}_{realization}") % (2**31)
            
            if self.property_type == 'resistivity':
                value = unit.get_resistivity(seed)
                value = np.log10(max(value, 0.1))  # Log scale, 0-5 range
            else:
                value = unit.get_porosity(seed) * 10  # 0-3.5 range
            
            profile.append({
                'depth_start': current_depth,
                'depth_end': current_depth + thickness,
                'value': value,
                'thickness': thickness
            })
            current_depth += thickness
        
        if not profile:
            return None
        
        total_depth = current_depth
        
        # Fill the 2D grid
        # Add lateral variation across width
        for layer in profile:
            y_start = int((layer['depth_start'] / self.max_depth_m) * height)
            y_end = int((layer['depth_end'] / self.max_depth_m) * height)
            
            y_start = max(0, min(y_start, height - 1))
            y_end = max(y_start + 1, min(y_end, height))
            
            base_value = layer['value']
            
            # Create lateral variation
            lateral_trend = np.sin(np.linspace(0, np.pi * self.rng.uniform(1, 3), width))
            lateral_variation = lateral_trend * 0.2 * self.rng.random()
            
            for y in range(y_start, y_end):
                # Add depth-dependent noise
                depth_noise = self.rng.randn(width) * 0.05
                target[y, :] = base_value + lateral_variation + depth_noise
        
        # Normalize to [0, 9] range
        if target.max() > target.min():
            target = 9.0 * (target - target.min()) / (target.max() - target.min() + 1e-6)
        
        # Create sparse sampling mask (simulating boreholes)
        mask = self._create_borehole_mask()
        
        # Convert to tensors
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)  # (1, H, W)
        sparse_input = target_tensor.clone()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
        
        return target_tensor, sparse_input, mask_tensor
    
    def _create_borehole_mask(self) -> np.ndarray:
        """Create a mask simulating borehole sampling pattern."""
        height, width = self.grid_size
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Number of boreholes based on sparse_rate
        num_boreholes = max(3, int(width * self.sparse_rate * 2))
        
        for _ in range(num_boreholes):
            # Random borehole x position
            x = self.rng.randint(0, width)
            
            # Borehole samples at various depths
            num_samples = self.rng.randint(3, min(15, height))
            y_positions = self.rng.choice(height, size=num_samples, replace=False)
            
            for y in y_positions:
                mask[y, x] = 1.0
        
        return mask
    
    def _augment_sample(self, sample: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Apply data augmentation."""
        target, sparse_input, mask = sample
        
        # Random horizontal flip
        if self.rng.random() > 0.5:
            target = torch.flip(target, dims=[2])
            sparse_input = torch.flip(sparse_input, dims=[2])
            mask = torch.flip(mask, dims=[2])
        
        # Random vertical shift
        if self.rng.random() > 0.5:
            shift = self.rng.randint(-3, 4)
            target = torch.roll(target, shifts=shift, dims=1)
            sparse_input = torch.roll(sparse_input, shifts=shift, dims=1)
            mask = torch.roll(mask, shifts=shift, dims=1)
        
        # Value perturbation
        noise = torch.randn_like(target) * 0.1
        target = torch.clamp(target + noise, 0, 9)
        sparse_input = torch.clamp(sparse_input + noise, 0, 9)
        
        # Regenerate mask
        new_mask = torch.from_numpy(self._create_borehole_mask()).float().unsqueeze(0)
        
        return target, sparse_input, new_mask
    
    def _generate_synthetic_samples(self) -> List[Tuple[torch.Tensor, ...]]:
        """Generate synthetic geological samples as fallback."""
        samples = []
        for i in range(self.num_samples):
            sample = self._generate_synthetic_sample(i)
            samples.append(sample)
        return samples
    
    def _generate_synthetic_sample(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Generate one synthetic geological sample based on typical geology."""
        np.random.seed(idx + 54321)
        height, width = self.grid_size
        
        # Create layered geology
        target = np.zeros((height, width), dtype=np.float32)
        
        # Random number of layers (3-7)
        num_layers = np.random.randint(3, 8)
        layer_boundaries = np.sort(np.random.randint(0, height, num_layers - 1))
        layer_boundaries = np.concatenate([[0], layer_boundaries, [height]])
        
        # Pick lithologies and assign properties
        lithology_keys = list(LITHOLOGY_RESISTIVITY.keys())
        
        for i in range(len(layer_boundaries) - 1):
            y_start = layer_boundaries[i]
            y_end = layer_boundaries[i + 1]
            
            # Pick random lithology
            lith = lithology_keys[np.random.randint(0, len(lithology_keys))]
            rho_min, rho_max = LITHOLOGY_RESISTIVITY[lith]
            
            # Base resistivity
            if self.property_type == 'resistivity':
                base_value = 10 ** np.random.uniform(np.log10(rho_min), np.log10(rho_max))
                base_value = np.log10(base_value)
            else:
                phi_min, phi_max = LITHOLOGY_POROSITY.get(lith, (0.05, 0.25))
                base_value = np.random.uniform(phi_min, phi_max) * 10
            
            # Add lateral variation
            for y in range(y_start, y_end):
                x_trend = np.sin(np.linspace(0, np.random.uniform(2, 6) * np.pi, width))
                variation = x_trend * 0.2 + np.random.randn(width) * 0.1
                target[y, :] = base_value + variation
        
        # Normalize to [0, 9]
        target = 9.0 * (target - target.min()) / (target.max() - target.min() + 1e-6)
        
        # Create mask
        mask = self._create_borehole_mask()
        
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)
        sparse_input = target_tensor.clone()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return target_tensor, sparse_input, mask_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return {
            'num_samples': len(self.samples),
            'grid_size': self.grid_size,
            'property_type': self.property_type,
            'num_columns': len(self.columns),
            'basins': self.basins,
            'formations': self.formations,
            'sparse_rate': self.sparse_rate,
        }


class MacrostratCrossSectionDataset(Dataset):
    """
    Dataset generating cross-sections through multiple columns.
    More realistic for training subsurface models.
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        grid_size: Tuple[int, int] = (32, 64),
        columns_per_section: int = 8,
        cache_dir: str = "macrostrat_cache",
        property_type: str = 'resistivity',
        sparse_rate: float = 0.05,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.columns_per_section = columns_per_section
        self.cache_dir = cache_dir
        self.property_type = property_type
        self.sparse_rate = sparse_rate
        
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.fetcher = MacrostratFetcher()
        
        # Fetch cross-section data
        self.sections = self._fetch_cross_sections()
        
        # Generate samples
        self.samples = self._generate_samples()
        
        print(f"CrossSectionDataset: {len(self.samples)} samples from {len(self.sections)} sections")
    
    def _fetch_cross_sections(self) -> List[List[StratColumn]]:
        """Fetch multiple cross-sections from different basins."""
        cache_file = os.path.join(self.cache_dir, "sections_cache.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                sections = pickle.load(f)
            print(f"Loaded {len(sections)} sections from cache")
            return sections
        
        print("Fetching cross-sections from Macrostrat...")
        sections = []
        
        for basin_key, coords in tqdm(BASIN_COORDINATES.items(), desc="Fetching sections"):
            try:
                # Get cross-section with multiple azimuths
                for azimuth in [0, 45, 90, 135]:
                    columns = self.fetcher.get_cross_section_data(
                        coords["lat"], coords["lng"],
                        azimuth=azimuth,
                        length_km=80,
                        num_columns=self.columns_per_section
                    )
                    
                    # Get detailed units
                    for col in columns:
                        if col.col_id:
                            units = self.fetcher.query_units_by_column(col.col_id)
                            if units:
                                col.units = units
                    
                    # Only keep if we have enough valid columns
                    valid_cols = [c for c in columns if len(c.units) >= 2]
                    if len(valid_cols) >= 3:
                        sections.append(valid_cols)
                        
            except Exception as e:
                print(f"Error fetching {basin_key}: {e}")
        
        print(f"Fetched {len(sections)} valid sections")
        
        if sections:
            with open(cache_file, 'wb') as f:
                pickle.dump(sections, f)
        
        return sections
    
    def _generate_samples(self) -> List[Tuple[torch.Tensor, ...]]:
        """Generate samples from cross-sections."""
        samples = []
        
        if not self.sections:
            # Fallback to synthetic
            for i in range(self.num_samples):
                samples.append(self._generate_synthetic_section(i))
            return samples
        
        samples_per_section = max(1, self.num_samples // len(self.sections))
        
        for section in self.sections:
            for real in range(samples_per_section):
                try:
                    sample = self._build_section_sample(section, real)
                    samples.append(sample)
                    
                    if len(samples) >= self.num_samples:
                        break
                except Exception:
                    pass
            
            if len(samples) >= self.num_samples:
                break
        
        # Fill with synthetic if needed
        while len(samples) < self.num_samples:
            samples.append(self._generate_synthetic_section(len(samples)))
        
        return samples[:self.num_samples]
    
    def _build_section_sample(self, columns: List[StratColumn], 
                               realization: int) -> Tuple[torch.Tensor, ...]:
        """Build sample from a cross-section of columns."""
        target, sparse_input, mask = self.fetcher.build_cross_section_grid(
            columns,
            grid_height=self.height,
            grid_width=self.width,
            property_type=self.property_type
        )
        
        # Override mask with our sparse pattern
        mask = self._create_sparse_mask()
        
        # Add stochastic variation
        noise = self.rng.randn(*target.shape).astype(np.float32) * 0.2 * realization
        target = np.clip(target + noise, 0, 9)
        
        return (
            torch.from_numpy(target).float().unsqueeze(0),
            torch.from_numpy(target.copy()).float().unsqueeze(0),
            torch.from_numpy(mask).float().unsqueeze(0)
        )
    
    def _create_sparse_mask(self) -> np.ndarray:
        """Create sparse sampling mask."""
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        num_wells = max(3, int(self.width * self.sparse_rate * 3))
        
        for _ in range(num_wells):
            x = self.rng.randint(0, self.width)
            num_samples = self.rng.randint(5, min(20, self.height))
            y_pos = self.rng.choice(self.height, size=num_samples, replace=False)
            mask[y_pos, x] = 1.0
        
        return mask
    
    def _generate_synthetic_section(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Generate synthetic cross-section."""
        np.random.seed(idx + 98765)
        
        target = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Create undulating layers
        num_layers = np.random.randint(4, 8)
        
        x_coords = np.arange(self.width)
        
        base_depths = np.linspace(0, self.height, num_layers + 1).astype(int)
        
        lithologies = list(LITHOLOGY_RESISTIVITY.keys())
        
        for i in range(num_layers):
            # Undulating layer boundary
            freq = np.random.uniform(0.5, 2)
            amp = np.random.uniform(2, 6)
            phase = np.random.uniform(0, 2 * np.pi)
            
            top_boundary = base_depths[i] + amp * np.sin(2 * np.pi * freq * x_coords / self.width + phase)
            bot_boundary = base_depths[i + 1] + amp * np.sin(2 * np.pi * freq * x_coords / self.width + phase + 0.5)
            
            top_boundary = np.clip(top_boundary, 0, self.height - 1).astype(int)
            bot_boundary = np.clip(bot_boundary, 0, self.height).astype(int)
            
            # Layer property
            lith = lithologies[np.random.randint(len(lithologies))]
            rho_min, rho_max = LITHOLOGY_RESISTIVITY[lith]
            
            if self.property_type == 'resistivity':
                base_val = np.log10(10 ** np.random.uniform(np.log10(rho_min), np.log10(rho_max)))
            else:
                phi_range = LITHOLOGY_POROSITY.get(lith, (0.05, 0.25))
                base_val = np.random.uniform(*phi_range) * 10
            
            for x in range(self.width):
                y_top = top_boundary[x]
                y_bot = bot_boundary[x]
                
                if y_bot > y_top:
                    noise = np.random.randn(y_bot - y_top) * 0.1
                    target[y_top:y_bot, x] = base_val + noise
        
        # Normalize
        target = 9.0 * (target - target.min()) / (target.max() - target.min() + 1e-6)
        
        mask = self._create_sparse_mask()
        
        return (
            torch.from_numpy(target).float().unsqueeze(0),
            torch.from_numpy(target.copy()).float().unsqueeze(0),
            torch.from_numpy(mask).float().unsqueeze(0)
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx]


if __name__ == "__main__":
    print("Testing MacrostratDataset...")
    
    # Test basic dataset
    dataset = MacrostratDataset(
        num_samples=50,
        grid_size=(32, 64),
        basins=["Permian_Delaware", "Williston_Bakken"],
        use_cache=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Metadata: {dataset.get_metadata()}")
    
    # Check sample
    target, sparse_input, mask = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Target: {target.shape}")
    print(f"  Sparse input: {sparse_input.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Target range: [{target.min():.2f}, {target.max():.2f}]")
    print(f"  Sparse points: {mask.sum().item():.0f}")

