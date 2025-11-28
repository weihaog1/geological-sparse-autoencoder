"""
Macrostrat API Data Fetcher for Geological Model Training
Fetches real stratigraphic data and converts to training format.
"""

import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import time
from functools import lru_cache

BASE_URL = "https://macrostrat.org/api"

# Lithology to resistivity mapping (Ohm-m) based on typical values
LITHOLOGY_RESISTIVITY = {
    # Sedimentary
    'shale': (5, 50),
    'mudstone': (5, 50),
    'claystone': (1, 20),
    'siltstone': (10, 100),
    'sandstone': (50, 500),
    'conglomerate': (50, 300),
    'limestone': (100, 10000),
    'doloite': (100, 5000),
    'dolomite': (100, 5000),
    'chalk': (50, 500),
    'carbonate': (100, 5000),
    'evaporite': (10, 10000),
    'gypsum': (100, 10000),
    'salt': (10, 100000),
    'coal': (10, 1000),
    'chert': (1000, 100000),
    # Igneous
    'granite': (1000, 100000),
    'basalt': (100, 50000),
    'rhyolite': (500, 50000),
    'andesite': (100, 10000),
    'gabbro': (1000, 100000),
    'diorite': (1000, 50000),
    'volcanic': (100, 10000),
    # Metamorphic
    'schist': (100, 10000),
    'gneiss': (1000, 100000),
    'slate': (100, 10000),
    'marble': (1000, 100000),
    'quartzite': (1000, 100000),
    # General
    'crystalline': (1000, 100000),
    'sedimentary': (10, 1000),
    'metamorphic': (100, 10000),
    'ignite': (100, 50000),
    'siliciite': (50, 500),
    'mixed': (10, 1000),
}

# Lithology to porosity mapping (fraction)
LITHOLOGY_POROSITY = {
    'shale': (0.02, 0.10),
    'mudstone': (0.02, 0.10),
    'claystone': (0.01, 0.08),
    'siltstone': (0.05, 0.20),
    'sandstone': (0.10, 0.35),
    'conglomerate': (0.05, 0.25),
    'limestone': (0.01, 0.20),
    'doloite': (0.01, 0.15),
    'dolomite': (0.01, 0.15),
    'chalk': (0.15, 0.40),
    'carbonate': (0.01, 0.20),
    'evaporite': (0.01, 0.05),
    'gypsum': (0.01, 0.05),
    'salt': (0.001, 0.02),
    'coal': (0.01, 0.15),
    'chert': (0.01, 0.10),
    'granite': (0.001, 0.02),
    'basalt': (0.01, 0.15),
    'volcanic': (0.05, 0.20),
    'schist': (0.001, 0.05),
    'gneiss': (0.001, 0.02),
    'crystalline': (0.001, 0.02),
    'sedimentary': (0.05, 0.25),
    'metamorphic': (0.001, 0.05),
    'mixed': (0.05, 0.20),
}


@dataclass
class StratUnit:
    """Represents a stratigraphic unit from Macrostrat."""
    unit_id: int
    unit_name: str
    strat_name: str
    lith: List[str]
    lith_type: str
    environ: str
    t_age: float  # Top age (Ma)
    b_age: float  # Bottom age (Ma)
    thickness: Optional[float]  # meters
    min_thick: Optional[float]
    max_thick: Optional[float]
    col_id: Optional[int] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    
    @property
    def primary_lithology(self) -> str:
        """Get the primary lithology."""
        if self.lith:
            return self.lith[0].lower() if isinstance(self.lith[0], str) else 'mixed'
        return self.lith_type.lower() if self.lith_type else 'mixed'
    
    def get_resistivity(self, seed: Optional[int] = None) -> float:
        """Get a resistivity value based on lithology."""
        lith = self.primary_lithology
        rng = np.random.RandomState(seed)
        
        for key, (rho_min, rho_max) in LITHOLOGY_RESISTIVITY.items():
            if key in lith:
                # Log-uniform distribution for resistivity
                return 10 ** rng.uniform(np.log10(rho_min), np.log10(rho_max))
        
        # Default
        return 10 ** rng.uniform(1, 3)
    
    def get_porosity(self, seed: Optional[int] = None) -> float:
        """Get a porosity value based on lithology."""
        lith = self.primary_lithology
        rng = np.random.RandomState(seed)
        
        for key, (phi_min, phi_max) in LITHOLOGY_POROSITY.items():
            if key in lith:
                return rng.uniform(phi_min, phi_max)
        
        return rng.uniform(0.05, 0.20)


@dataclass
class StratColumn:
    """Represents a stratigraphic column from Macrostrat."""
    col_id: int
    col_name: str
    lat: float
    lng: float
    col_area: float
    max_thick: float
    units: List[StratUnit]
    
    @property
    def total_thickness(self) -> float:
        """Calculate total thickness from units."""
        return sum(u.thickness or u.max_thick or 100 for u in self.units)


class MacrostratFetcher:
    """Fetches and processes data from Macrostrat API."""
    
    def __init__(self, cache_dir: Optional[str] = None, rate_limit: float = 0.5):
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeoSparseAutoencoder/1.0 (Research)'
        })
    
    def _rate_limited_request(self, url: str, params: Dict) -> Dict:
        """Make a rate-limited API request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {'success': {'data': []}}
    
    def query_columns_by_location(self, lat: float, lng: float, 
                                   radius_km: float = 50) -> List[StratColumn]:
        """Query stratigraphic columns near a location."""
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0
        
        params = {
            'lat': lat,
            'lng': lng,
            'adjacents': 'true',
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/columns", params)
        columns = []
        
        if 'success' in result and 'data' in result['success']:
            for col_data in result['success']['data']:
                try:
                    col = self._parse_column(col_data)
                    if col:
                        columns.append(col)
                except Exception as e:
                    print(f"Error parsing column: {e}")
        
        return columns
    
    def query_columns_in_bbox(self, lat_min: float, lat_max: float,
                               lng_min: float, lng_max: float) -> List[StratColumn]:
        """Query columns within a bounding box."""
        params = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lng_min': lng_min,
            'lng_max': lng_max,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/columns", params)
        columns = []
        
        if 'success' in result and 'data' in result['success']:
            for col_data in result['success']['data']:
                try:
                    col = self._parse_column(col_data)
                    if col:
                        columns.append(col)
                except Exception as e:
                    pass
        
        return columns
    
    def query_units_by_column(self, col_id: int) -> List[StratUnit]:
        """Query all units in a column."""
        params = {
            'col_id': col_id,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/units", params)
        units = []
        
        if 'success' in result and 'data' in result['success']:
            for unit_data in result['success']['data']:
                try:
                    unit = self._parse_unit(unit_data, col_id)
                    if unit:
                        units.append(unit)
                except Exception as e:
                    pass
        
        return units
    
    def query_units_by_formation(self, formation_name: str, 
                                  project_id: int = 1) -> List[StratUnit]:
        """Query units by formation name."""
        params = {
            'strat_name_like': formation_name,
            'project_id': project_id,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/units", params)
        units = []
        
        if 'success' in result and 'data' in result['success']:
            for unit_data in result['success']['data']:
                try:
                    unit = self._parse_unit(unit_data)
                    if unit:
                        units.append(unit)
                except Exception as e:
                    pass
        
        return units
    
    def query_units_by_age(self, interval_name: str, 
                           project_id: int = 1) -> List[StratUnit]:
        """Query units by geological age interval."""
        params = {
            'interval_name': interval_name,
            'project_id': project_id,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/units", params)
        units = []
        
        if 'success' in result and 'data' in result['success']:
            for unit_data in result['success']['data']:
                try:
                    unit = self._parse_unit(unit_data)
                    if unit:
                        units.append(unit)
                except Exception as e:
                    pass
        
        return units
    
    def query_units_by_lithology(self, lith_type: str = "sedimentary",
                                  project_id: int = 1) -> List[StratUnit]:
        """Query units by lithology type."""
        params = {
            'lith_type': lith_type,
            'project_id': project_id,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/units", params)
        units = []
        
        if 'success' in result and 'data' in result['success']:
            for unit_data in result['success']['data']:
                try:
                    unit = self._parse_unit(unit_data)
                    if unit:
                        units.append(unit)
                except Exception as e:
                    pass
        
        return units
    
    def query_economic_columns(self, resource: str = "oil,gas") -> List[StratColumn]:
        """Query columns with economic resources (oil, gas, coal)."""
        params = {
            'econ': resource,
            'project_id': 1,
            'format': 'json',
            'response': 'long'
        }
        
        result = self._rate_limited_request(f"{BASE_URL}/v2/columns", params)
        columns = []
        
        if 'success' in result and 'data' in result['success']:
            for col_data in result['success']['data']:
                try:
                    col = self._parse_column(col_data)
                    if col:
                        columns.append(col)
                except Exception as e:
                    pass
        
        return columns
    
    def _parse_column(self, data: Dict) -> Optional[StratColumn]:
        """Parse column data from API response."""
        try:
            units = []
            if 'units' in data:
                for unit_data in data['units']:
                    unit = self._parse_unit(unit_data, data.get('col_id'))
                    if unit:
                        units.append(unit)
            
            return StratColumn(
                col_id=data.get('col_id', 0),
                col_name=data.get('col_name', 'Unknown'),
                lat=data.get('lat', 0),
                lng=data.get('lng', 0),
                col_area=data.get('col_area', 0),
                max_thick=data.get('max_thick', 0),
                units=units
            )
        except Exception:
            return None
    
    def _parse_unit(self, data: Dict, col_id: Optional[int] = None) -> Optional[StratUnit]:
        """Parse unit data from API response."""
        try:
            # Handle lithology - could be string or list
            lith = data.get('lith', [])
            if isinstance(lith, str):
                lith = [lith]
            elif lith is None:
                lith = []
            
            return StratUnit(
                unit_id=data.get('unit_id', 0),
                unit_name=data.get('unit_name', 'Unknown'),
                strat_name=data.get('strat_name', data.get('strat_name_long', 'Unknown')),
                lith=lith,
                lith_type=data.get('lith_type', 'mixed'),
                environ=data.get('environ', ''),
                t_age=data.get('t_age', 0),
                b_age=data.get('b_age', 0),
                thickness=data.get('max_thick'),
                min_thick=data.get('min_thick'),
                max_thick=data.get('max_thick'),
                col_id=col_id or data.get('col_id'),
                lat=data.get('clat'),
                lng=data.get('clng')
            )
        except Exception:
            return None
    
    def get_cross_section_data(self, lat: float, lng: float, 
                                 azimuth: float = 90.0,
                                 length_km: float = 100.0,
                                 num_columns: int = 10) -> List[StratColumn]:
        """
        Get data for a cross-section line.
        
        Args:
            lat, lng: Starting point
            azimuth: Direction in degrees (0=N, 90=E)
            length_km: Total length in km
            num_columns: Number of columns to sample
        """
        columns = []
        
        # Calculate endpoint
        azimuth_rad = np.radians(azimuth)
        
        for i in range(num_columns):
            # Calculate position along line
            distance = length_km * i / (num_columns - 1)
            
            # Approximate lat/lng change
            dlat = (distance * np.cos(azimuth_rad)) / 111.0
            dlng = (distance * np.sin(azimuth_rad)) / (111.0 * np.cos(np.radians(lat)))
            
            point_lat = lat + dlat
            point_lng = lng + dlng
            
            # Query columns at this point
            local_cols = self.query_columns_by_location(point_lat, point_lng, radius_km=10)
            
            if local_cols:
                # Take the closest one
                columns.append(local_cols[0])
        
        return columns
    
    def build_cross_section_grid(self, columns: List[StratColumn],
                                   grid_height: int = 32,
                                   grid_width: int = 64,
                                   max_depth_m: float = 3000,
                                   property_type: str = 'resistivity'
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a 2D grid from a series of stratigraphic columns.
        
        Returns:
            target: Full ground truth grid (H, W)
            sparse_input: Sparse sampled input (H, W) 
            mask: Sampling mask (H, W)
        """
        if len(columns) == 0:
            raise ValueError("No columns provided")
        
        # Initialize grids
        target = np.zeros((grid_height, grid_width), dtype=np.float32)
        mask = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # Interpolate columns to grid width
        col_positions = np.linspace(0, grid_width - 1, len(columns)).astype(int)
        
        for col_idx, col in enumerate(columns):
            x = col_positions[col_idx]
            
            if not col.units:
                continue
            
            # Sort units by age (top to bottom)
            units_sorted = sorted(col.units, key=lambda u: u.t_age)
            
            # Calculate cumulative depth
            current_depth = 0
            for unit in units_sorted:
                thickness = unit.thickness or unit.max_thick or 100
                
                # Convert depth to grid position
                y_start = int((current_depth / max_depth_m) * grid_height)
                y_end = int(((current_depth + thickness) / max_depth_m) * grid_height)
                
                y_start = min(y_start, grid_height - 1)
                y_end = min(y_end, grid_height)
                
                if y_start >= grid_height:
                    break
                
                # Get property value
                seed = hash(f"{col.col_id}_{unit.unit_id}") % (2**31)
                
                if property_type == 'resistivity':
                    value = unit.get_resistivity(seed)
                    # Convert to log scale and normalize
                    value = np.log10(value)  # Typically 0-5 range
                elif property_type == 'porosity':
                    value = unit.get_porosity(seed)
                    value = value * 10  # Scale to 0-3.5 range
                else:
                    value = unit.get_resistivity(seed)
                    value = np.log10(value)
                
                # Fill the column segment
                for y in range(y_start, y_end):
                    target[y, x] = value
                    
                    # Sparse sampling - mark some points
                    if np.random.random() < 0.3:  # 30% sample rate
                        mask[y, x] = 1.0
                
                current_depth += thickness
        
        # Interpolate between columns for full grid
        from scipy.interpolate import griddata
        
        # Get known points
        known_y, known_x = np.where(target > 0)
        known_values = target[known_y, known_x]
        
        if len(known_values) > 0:
            # Create full grid
            grid_y, grid_x = np.mgrid[0:grid_height, 0:grid_width]
            
            # Interpolate
            target_full = griddata(
                (known_y, known_x), known_values,
                (grid_y, grid_x), method='linear', fill_value=np.mean(known_values)
            )
            target = target_full.astype(np.float32)
        
        # Normalize to [0, 9] range for model
        if target.max() > target.min():
            target = 9.0 * (target - target.min()) / (target.max() - target.min())
        
        sparse_input = target.copy()
        
        return target, sparse_input, mask


# Major oil & gas basin coordinates for training data
BASIN_COORDINATES = {
    "Permian_Delaware": {"lat": 31.8, "lng": -103.9, "name": "Delaware Basin"},
    "Permian_Midland": {"lat": 31.9, "lng": -101.5, "name": "Midland Basin"},
    "Williston_Bakken": {"lat": 48.0, "lng": -103.0, "name": "Williston Basin"},
    "DJ_Niobrara": {"lat": 40.0, "lng": -104.5, "name": "DJ Basin"},
    "Appalachian": {"lat": 40.0, "lng": -80.0, "name": "Appalachian Basin"},
    "Gulf_Coast": {"lat": 29.5, "lng": -95.0, "name": "Gulf Coast Basin"},
    "Anadarko": {"lat": 35.5, "lng": -99.0, "name": "Anadarko Basin"},
    "Powder_River": {"lat": 44.0, "lng": -106.0, "name": "Powder River Basin"},
    "Uinta": {"lat": 40.0, "lng": -110.0, "name": "Uinta Basin"},
    "San_Juan": {"lat": 36.5, "lng": -108.0, "name": "San Juan Basin"},
    "Michigan": {"lat": 43.5, "lng": -84.5, "name": "Michigan Basin"},
    "Illinois": {"lat": 38.5, "lng": -88.5, "name": "Illinois Basin"},
}

TARGET_FORMATIONS = [
    "Wolfcamp", "Bone Spring", "Spraberry", "Delaware",
    "Bakken", "Three Forks", "Niobrara", "Codell",
    "Marcellus", "Utica", "Eagle Ford", "Austin Chalk",
    "Woodford", "Barnett", "Fayetteville", "Haynesville"
]


if __name__ == "__main__":
    # Test the fetcher
    print("Testing Macrostrat Fetcher...")
    
    fetcher = MacrostratFetcher()
    
    # Test: Query Permian Basin columns
    print("\n1. Querying Permian Delaware Basin...")
    coords = BASIN_COORDINATES["Permian_Delaware"]
    columns = fetcher.query_columns_by_location(coords["lat"], coords["lng"])
    print(f"   Found {len(columns)} columns")
    
    if columns:
        col = columns[0]
        print(f"   First column: {col.col_name} ({len(col.units)} units)")
        
        # Get units
        units = fetcher.query_units_by_column(col.col_id)
        print(f"   Detailed units: {len(units)}")
        
        for unit in units[:3]:
            print(f"     - {unit.strat_name}: {unit.primary_lithology}, "
                  f"rho={unit.get_resistivity():.1f} Ohm-m")
    
    # Test: Query by formation
    print("\n2. Querying Wolfcamp formation...")
    wolfcamp_units = fetcher.query_units_by_formation("Wolfcamp")
    print(f"   Found {len(wolfcamp_units)} Wolfcamp units")
    
    # Test: Build cross-section
    print("\n3. Building cross-section grid...")
    if columns:
        try:
            target, sparse_input, mask = fetcher.build_cross_section_grid(
                columns[:5], grid_height=32, grid_width=64
            )
            print(f"   Grid shape: {target.shape}")
            print(f"   Value range: [{target.min():.2f}, {target.max():.2f}]")
            print(f"   Sparse samples: {mask.sum():.0f}")
        except Exception as e:
            print(f"   Error building grid: {e}")
    
    print("\nFetcher test complete!")

