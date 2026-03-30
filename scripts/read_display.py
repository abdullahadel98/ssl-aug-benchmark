#!/usr/bin/env python3
"""
Script to read and display file content with various options (including numpy arrays).

Usage:
    python read_display.py [OPTIONS] <file>

Options:
    -H, --head N       Display first N lines (default: 20)
    -t, --tail N       Display last N lines (default: 20)
    -l, --lines N      Display specific line range (format: start-end, e.g., 10-50)
    -c, --cat          Display entire file content
    -n, --number       Show line numbers
    -s, --search TERM  Search and highlight specific term
    --help             Show this help message

Supported File Types:
    - Text files (.txt, .log, .yaml, .json, etc.)
    - Numpy arrays (.npy)

Examples:
    python read_display.py -H 50 file.txt              # Show first 50 lines
    python read_display.py -t 30 file.txt              # Show last 30 lines
    python read_display.py -l 10-50 file.txt           # Show lines 10-50
    python read_display.py -n -c file.txt              # Show entire file with line numbers
    python read_display.py -s "error" file.txt         # Search and highlight "error"
    python read_display.py file.txt                    # Show first 20 lines (default)
    python read_display.py data.npy                    # Display numpy array content
    python read_display.py -n data.npy                 # Display numpy array with indices

"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def read_file(filepath):
    """Read file and return lines."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def read_numpy_file(filepath):
    """Read and return numpy array."""
    if not NUMPY_AVAILABLE:
        print("Error: numpy is not installed. Install it with: pip install numpy", file=sys.stderr)
        sys.exit(1)
    
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"Error reading numpy file: {e}", file=sys.stderr)
        sys.exit(1)


def display_head(lines, num_lines, show_numbers):
    """Display first N lines."""
    head_lines = lines[:num_lines]
    num_to_show = len(head_lines)
    print(f"=== First {num_to_show} lines ===")
    for i, line in enumerate(head_lines, 1):
        if show_numbers:
            print(f"{i:6d}: {line}", end='')
        else:
            print(line, end='')


def display_tail(lines, num_lines, show_numbers, total):
    """Display last N lines."""
    tail_lines = lines[-num_lines:]
    num_to_show = len(tail_lines)
    start_line = total - num_to_show + 1
    print(f"=== Last {num_to_show} lines (Total: {total} lines) ===")
    for i, line in enumerate(tail_lines, start_line):
        if show_numbers:
            print(f"{i:6d}: {line}", end='')
        else:
            print(line, end='')


def display_lines(lines, line_range, show_numbers, total):
    """Display specific line range."""
    try:
        start, end = map(int, line_range.split('-'))
        if start < 1 or end > total or start > end:
            print(f"Error: Invalid line range. File has {total} lines.", file=sys.stderr)
            sys.exit(1)
        selected_lines = lines[start-1:end]
        print(f"=== Lines {start}-{end} (Total: {total} lines) ===")
        for i, line in enumerate(selected_lines, start):
            if show_numbers:
                print(f"{i:6d}: {line}", end='')
            else:
                print(line, end='')
    except ValueError:
        print(f"Error: Invalid line range format. Use 'start-end' (e.g., 10-50).", file=sys.stderr)
        sys.exit(1)


def display_all(lines, show_numbers, total):
    """Display entire file."""
    print(f"=== Full content (Total: {total} lines) ===")
    for i, line in enumerate(lines, 1):
        if show_numbers:
            print(f"{i:6d}: {line}", end='')
        else:
            print(line, end='')


def search_in_file(lines, search_term, show_numbers, total):
    """Search and display matching lines."""
    matches = []
    for i, line in enumerate(lines, 1):
        if search_term.lower() in line.lower():
            matches.append((i, line))
    
    if not matches:
        print(f"=== No matches found for '{search_term}' ===")
        return
    
    print(f"=== Found {len(matches)} match(es) for '{search_term}' ===")
    for line_num, line in matches:
        if show_numbers:
            # Highlight the search term
            highlighted_line = line.replace(
                search_term, 
                f"\033[92m{search_term}\033[0m"
            )
            print(f"{line_num:6d}: {highlighted_line}", end='')
        else:
            highlighted_line = line.replace(
                search_term, 
                f"\033[92m{search_term}\033[0m"
            )
            print(highlighted_line, end='')


def display_numpy_array(array, show_numbers=False, max_rows=None, max_cols=None):
    """Display numpy array content."""
    print(f"=== Numpy Array ===")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Size: {array.size}")
    
    # Set print options
    np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=np.inf)
    
    # For very large arrays, only show a slice
    if max_rows is None:
        max_rows = 100
    if max_cols is None:
        max_cols = 10
    
    if array.ndim == 0:
        # Scalar
        print(f"Value: {array}")
    elif array.ndim == 1:
        # 1D array
        if array.size > max_rows:
            print(f"\n[Showing first {max_rows} of {array.size} elements]")
            display_array = array[:max_rows]
        else:
            display_array = array
        
        if show_numbers:
            for i, val in enumerate(display_array):
                print(f"{i:6d}: {val}")
        else:
            print(display_array)
        
        if array.size > max_rows:
            print(f"... ({array.size - max_rows} more elements)")
    
    elif array.ndim == 2:
        # 2D array (matrix)
        rows, cols = array.shape
        if rows > max_rows or cols > max_cols:
            print(f"\n[Showing first {min(max_rows, rows)}x{min(max_cols, cols)} of {rows}x{cols}]")
            display_array = array[:max_rows, :max_cols]
        else:
            display_array = array
        
        if show_numbers:
            for i, row in enumerate(display_array):
                print(f"{i:6d}: {row}")
        else:
            print(display_array)
        
        if rows > max_rows or cols > max_cols:
            print(f"... (truncated to first {min(max_rows, rows)} rows and {min(max_cols, cols)} cols)")
    
    else:
        # Higher dimensional arrays
        print(f"\n[Array has {array.ndim} dimensions]")
        if array.size > max_rows:
            print(f"[Showing first {max_rows} elements (flattened)]")
            display_array = array.flatten()[:max_rows]
        else:
            display_array = array.flatten()
        
        if show_numbers:
            for i, val in enumerate(display_array):
                print(f"{i:6d}: {val}")
        else:
            print(display_array)
        
        if array.size > max_rows:
            print(f"... ({array.size - max_rows} more elements)")


def main():
    parser = argparse.ArgumentParser(
        description='Read and display file content with various options.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('file', help='File to read')
    parser.add_argument('-H', '--head', type=int, default=20, 
                        help='Display first N lines (default: 20)')
    parser.add_argument('-t', '--tail', type=int, default=20,
                        help='Display last N lines (default: 20)')
    parser.add_argument('-l', '--lines', type=str,
                        help='Display line range (format: start-end, e.g., 10-50)')
    parser.add_argument('-c', '--cat', action='store_true',
                        help='Display entire file content')
    parser.add_argument('-n', '--number', action='store_true',
                        help='Show line numbers')
    parser.add_argument('-s', '--search', type=str,
                        help='Search and highlight specific term')
    
    args = parser.parse_args()
    
    # Check if file exists
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Handle numpy files
    if filepath.suffix == '.npy':
        print(f"Loading numpy file: {filepath}")
        array = read_numpy_file(filepath)
        display_numpy_array(array, show_numbers=args.number)
        print(f"\nFile: {filepath}")
        return
    
    # Read text file
    lines = read_file(filepath)
    total_lines = len(lines)
    
    # Display based on mode
    if args.search:
        search_in_file(lines, args.search, args.number, total_lines)
    elif args.lines:
        display_lines(lines, args.lines, args.number, total_lines)
    elif args.cat:
        display_all(lines, args.number, total_lines)
    elif args.tail != 20:  # If tail was explicitly set
        display_tail(lines, args.tail, args.number, total_lines)
    else:
        # Default: head
        display_head(lines, args.head, args.number)
    
    print(f"\nFile: {filepath} | Total lines: {total_lines}")


if __name__ == '__main__':
    main()
