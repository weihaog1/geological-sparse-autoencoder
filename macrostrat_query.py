#!/usr/bin/env python3
"""
Macrostrat API Query Tool for Geological Model Training

This script provides:
1. CLI interface for querying Macrostrat API
2. Data fetching for training the sparse autoencoder
3. Integration with the training pipeline

Usage:
    # Test API connection
    python macrostrat_query.py --test
    
    # Fetch data for specific basin
    python macrostrat_query.py --basin Permian_Delaware --save-cache
    
    # Query formation
    python macrostrat_query.py --formation Wolfcamp --verbose
    
    # Prefetch all training data
    python macrostrat_query.py --prefetch-all
"""

import argparse
import json
import sys
from typing import Optional

# Import from our data generation module
from sparseconvae.data_generation.macrostrat_fetcher import (
    MacrostratFetcher,
    BASIN_COORDINATES,
    TARGET_FORMATIONS,
    LITHOLOGY_RESISTIVITY,
    LITHOLOGY_POROSITY
)


def test_api():
    """Test API connectivity and basic queries."""
    print("Testing Macrostrat API connectivity...\n")
    
    fetcher = MacrostratFetcher()
    
    # Test 1: Query a known location
    print("1. Testing location query (Permian Basin)...")
    coords = BASIN_COORDINATES["Permian_Delaware"]
    columns = fetcher.query_columns_by_location(coords["lat"], coords["lng"])
    
    if columns:
        print(f"   ✓ Found {len(columns)} columns")
        col = columns[0]
        print(f"   First column: {col.col_name} (ID: {col.col_id})")
    else:
        print("   ✗ No columns found - check API status")
        return False
    
    # Test 2: Query units
    print("\n2. Testing unit query...")
    if columns:
        units = fetcher.query_units_by_column(columns[0].col_id)
        if units:
            print(f"   ✓ Found {len(units)} units")
            for unit in units[:3]:
                rho = unit.get_resistivity()
                print(f"     - {unit.strat_name}: {unit.primary_lithology}, ρ={rho:.1f} Ω·m")
        else:
            print("   ✗ No units found")
    
    # Test 3: Formation query
    print("\n3. Testing formation query (Wolfcamp)...")
    wolfcamp = fetcher.query_units_by_formation("Wolfcamp")
    print(f"   ✓ Found {len(wolfcamp)} Wolfcamp units")
    
    # Test 4: Age query
    print("\n4. Testing age query (Permian)...")
    permian = fetcher.query_units_by_age("Permian")
    print(f"   ✓ Found {len(permian)} Permian-aged units")
    
    print("\n" + "=" * 50)
    print("API TEST PASSED - Ready for training")
    print("=" * 50)
    return True


def query_basin(basin_key: str, verbose: bool = False, save_cache: bool = False):
    """Query a specific basin."""
    if basin_key not in BASIN_COORDINATES:
        print(f"Unknown basin: {basin_key}")
        print(f"Available: {list(BASIN_COORDINATES.keys())}")
        return
    
    coords = BASIN_COORDINATES[basin_key]
    print(f"\nQuerying {coords['name']} ({coords['lat']}, {coords['lng']})...")
    
    fetcher = MacrostratFetcher()
    columns = fetcher.query_columns_by_location(coords["lat"], coords["lng"], radius_km=100)
    
    print(f"Found {len(columns)} stratigraphic columns\n")
    
    for col in columns[:10]:
        units = fetcher.query_units_by_column(col.col_id)
        col.units = units
        
        if verbose:
            print(f"\n{col.col_name} (ID: {col.col_id})")
            print(f"  Location: ({col.lat:.2f}, {col.lng:.2f})")
            print(f"  Units: {len(units)}")
            
            for unit in units[:5]:
                print(f"    - {unit.strat_name}")
                print(f"      Lithology: {unit.primary_lithology}")
                print(f"      Age: {unit.t_age:.1f} - {unit.b_age:.1f} Ma")
                print(f"      Resistivity: {unit.get_resistivity():.1f} Ω·m")
        else:
            print(f"  {col.col_name}: {len(units)} units")
    
    if save_cache:
        import pickle
        import os
        os.makedirs("macrostrat_cache", exist_ok=True)
        cache_file = f"macrostrat_cache/{basin_key}_columns.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(columns, f)
        print(f"\nCached to {cache_file}")


def query_formation(formation_name: str, verbose: bool = False):
    """Query a specific formation."""
    print(f"\nQuerying formation: {formation_name}...")
    
    fetcher = MacrostratFetcher()
    units = fetcher.query_units_by_formation(formation_name)
    
    print(f"Found {len(units)} units\n")
    
    if verbose:
        # Group by lithology
        lith_counts = {}
        for unit in units:
            lith = unit.primary_lithology
            lith_counts[lith] = lith_counts.get(lith, 0) + 1
        
        print("Lithology distribution:")
        for lith, count in sorted(lith_counts.items(), key=lambda x: -x[1]):
            print(f"  {lith}: {count}")
        
        print("\nSample units:")
        for unit in units[:10]:
            print(f"  - {unit.strat_name} ({unit.primary_lithology})")
            print(f"    Age: {unit.t_age:.1f} - {unit.b_age:.1f} Ma")
    else:
        for unit in units[:20]:
            print(f"  {unit.strat_name}: {unit.primary_lithology}")


def prefetch_all():
    """Prefetch all data for training."""
    print("Prefetching all Macrostrat data for training...\n")
    
    from sparseconvae.datasets.macrostrat_dataset import MacrostratDataset
    
    # This will fetch and cache all data
    dataset = MacrostratDataset(
        num_samples=100,  # Just need to trigger fetch
        basins=list(BASIN_COORDINATES.keys()),
        formations=TARGET_FORMATIONS,
        use_cache=True,
        offline_mode=False
    )
    
    print(f"\nPrefetch complete!")
    print(f"  Columns cached: {len(dataset.columns)}")
    print(f"  Cache location: macrostrat_cache/")
    print("\nYou can now train with --offline flag:")
    print("  python train_macrostrat.py --offline")


def show_lithology_mappings():
    """Show lithology to resistivity/porosity mappings."""
    print("\nLithology to Resistivity Mapping (Ω·m):")
    print("-" * 45)
    for lith, (rho_min, rho_max) in sorted(LITHOLOGY_RESISTIVITY.items()):
        print(f"  {lith:20s}: {rho_min:8.0f} - {rho_max:8.0f}")
    
    print("\n\nLithology to Porosity Mapping (fraction):")
    print("-" * 45)
    for lith, (phi_min, phi_max) in sorted(LITHOLOGY_POROSITY.items()):
        print(f"  {lith:20s}: {phi_min:6.3f} - {phi_max:6.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Macrostrat API Query Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                           Test API connection
  %(prog)s --basin Permian_Delaware -v      Query basin with details
  %(prog)s --formation Wolfcamp -v          Query formation
  %(prog)s --prefetch-all                   Prefetch all training data
  %(prog)s --show-mappings                  Show lithology mappings

Available basins:
  """ + ", ".join(BASIN_COORDINATES.keys())
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Test API connectivity')
    parser.add_argument('--basin', type=str,
                        help='Query specific basin')
    parser.add_argument('--formation', type=str,
                        help='Query specific formation')
    parser.add_argument('--prefetch-all', action='store_true',
                        help='Prefetch all data for training')
    parser.add_argument('--show-mappings', action='store_true',
                        help='Show lithology to petrophysical mappings')
    parser.add_argument('--save-cache', action='store_true',
                        help='Save results to cache')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.test:
        test_api()
    elif args.basin:
        query_basin(args.basin, args.verbose, args.save_cache)
    elif args.formation:
        query_formation(args.formation, args.verbose)
    elif args.prefetch_all:
        prefetch_all()
    elif args.show_mappings:
        show_lithology_mappings()
    else:
        # Default: run test
        print("Running API test (use --help for options)\n")
        test_api()


if __name__ == "__main__":
    main()
