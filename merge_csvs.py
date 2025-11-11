#!/usr/bin/env python3
"""
Script to list all CSV files in a directory and merge those with matching prefixes.
"""

import os
import pandas as pd
import argparse
import glob
from pathlib import Path


def list_csvs_in_directory(directory_path):
    """
    List all CSV files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory to scan
        
    Returns:
        list: List of CSV file paths
    """
    csv_files = []
    
    # Use glob to find all CSV files
    pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(pattern)
    
    return sorted(csv_files)


def get_files_with_prefix(csv_files, prefix):
    """
    Filter CSV files that start with the given prefix.
    
    Args:
        csv_files (list): List of CSV file paths
        prefix (str): Prefix to match
        
    Returns:
        list: List of CSV files with matching prefix
    """
    matching_files = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith(prefix):
            matching_files.append(csv_file)
    
    return matching_files


def merge_csv_files(csv_files, output_file=None):
    """
    Merge multiple CSV files into one.
    
    Args:
        csv_files (list): List of CSV file paths to merge
        output_file (str, optional): Output file path. If None, returns DataFrame
        
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    if not csv_files:
        print("No CSV files to merge.")
        return pd.DataFrame()
    
    print(f"Merging {len(csv_files)} CSV files...")
    
    dataframes = []
    
    for csv_file in csv_files:
        print(f"Reading: {os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file)
            # Add a column to track source file
            df['source_file'] = os.path.basename(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not dataframes:
        print("No valid CSV files could be read.")
        return pd.DataFrame()
    
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to: {output_file}")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="List and merge CSV files with matching prefix")
    parser.add_argument("directory", help="Directory path to scan for CSV files")
    parser.add_argument("--prefix", "-p", help="Prefix to match for merging CSV files")
    parser.add_argument("--output", "-o", help="Output file path for merged CSV")
    parser.add_argument("--list-only", "-l", action="store_true", 
                       help="Only list CSV files, don't merge")
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return 1
    
    # List all CSV files
    print(f"Scanning directory: {args.directory}")
    csv_files = list_csvs_in_directory(args.directory)
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return 0
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # If only listing, exit here
    if args.list_only:
        return 0
    
    # If prefix is provided, filter and merge
    if args.prefix:
        matching_files = get_files_with_prefix(csv_files, args.prefix)
        
        if not matching_files:
            print(f"\nNo CSV files found with prefix '{args.prefix}'")
            return 0
        
        print(f"\nFiles matching prefix '{args.prefix}':")
        for file in matching_files:
            print(f"  - {os.path.basename(file)}")
        
        # Set default output file if not provided
        output_file = args.output
        if not output_file:
            output_file = os.path.join(args.directory, f"merged_{args.prefix}.csv")
        
        # Merge the files
        print(f"\nMerging files...")
        merged_df = merge_csv_files(matching_files, output_file)
        
        if not merged_df.empty:
            print(f"Successfully merged {len(matching_files)} files.")
            print(f"Final dataset shape: {merged_df.shape}")
            print(f"Columns: {list(merged_df.columns)}")
        
    else:
        print("\nNo prefix provided. Use --prefix to specify which files to merge.")
        print("Example usage:")
        print(f"  python {os.path.basename(__file__)} {args.directory} --prefix adsb_flightpings")
    
    return 0


if __name__ == "__main__":
    exit(main())
