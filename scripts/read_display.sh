#!/bin/bash

# Script to read and display file content with various options
# Usage: ./read_display.sh [OPTIONS] <file>
# Options:
#   -h, --head N       Display first N lines (default: 20)
#   -t, --tail N       Display last N lines (default: 20)
#   -l, --lines N      Display specific line range (format: start-end, e.g., 10-50)
#   -c, --cat          Display entire file content
#   -n, --number       Show line numbers
#   -s, --search TERM  Search and highlight specific term
#   --help             Show this help message

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] <file>

Read and display file content with various options.

Options:
    -h, --head N          Display first N lines (default: 20)
    -t, --tail N          Display last N lines (default: 20)
    -l, --lines N         Display specific line range (format: start-end, e.g., 10-50)
    -c, --cat             Display entire file content
    -n, --number          Show line numbers
    -s, --search TERM     Search and highlight specific term
    --help                Show this help message

Examples:
    $(basename "$0") -h 50 file.txt              # Show first 50 lines
    $(basename "$0") -t 30 file.txt              # Show last 30 lines
    $(basename "$0") -l 10-50 file.txt           # Show lines 10-50
    $(basename "$0") -n -c file.txt              # Show entire file with line numbers
    $(basename "$0") -s "error" file.txt         # Search and highlight "error"
    $(basename "$0") file.txt                    # Show first 20 lines (default)

EOF
}

# Default values
HEAD_LINES=20
TAIL_LINES=20
SHOW_NUMBERS=false
DISPLAY_MODE="head"
SEARCH_TERM=""
FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--head)
            HEAD_LINES="$2"
            DISPLAY_MODE="head"
            shift 2
            ;;
        -t|--tail)
            TAIL_LINES="$2"
            DISPLAY_MODE="tail"
            shift 2
            ;;
        -l|--lines)
            LINE_RANGE="$2"
            DISPLAY_MODE="lines"
            shift 2
            ;;
        -c|--cat)
            DISPLAY_MODE="cat"
            shift
            ;;
        -n|--number)
            SHOW_NUMBERS=true
            shift
            ;;
        -s|--search)
            SEARCH_TERM="$2"
            DISPLAY_MODE="search"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            FILE="$1"
            shift
            ;;
    esac
done

# Check if file is provided
if [[ -z "$FILE" ]]; then
    echo "Error: No file specified."
    usage
    exit 1
fi

# Check if file exists
if [[ ! -f "$FILE" ]]; then
    echo "Error: File '$FILE' not found."
    exit 1
fi

# Get total lines in file
TOTAL_LINES=$(wc -l < "$FILE")

# Display based on mode
case $DISPLAY_MODE in
    head)
        echo "=== First $HEAD_LINES lines of $FILE ==="
        if [[ "$SHOW_NUMBERS" == true ]]; then
            head -n "$HEAD_LINES" "$FILE" | nl -v 1
        else
            head -n "$HEAD_LINES" "$FILE"
        fi
        ;;
    tail)
        echo "=== Last $TAIL_LINES lines of $FILE (Total: $TOTAL_LINES lines) ==="
        if [[ "$SHOW_NUMBERS" == true ]]; then
            tail -n "$TAIL_LINES" "$FILE" | nl -v $((TOTAL_LINES - TAIL_LINES + 1))
        else
            tail -n "$TAIL_LINES" "$FILE"
        fi
        ;;
    lines)
        START=$(echo "$LINE_RANGE" | cut -d'-' -f1)
        END=$(echo "$LINE_RANGE" | cut -d'-' -f2)
        echo "=== Lines $START-$END of $FILE (Total: $TOTAL_LINES lines) ==="
        if [[ "$SHOW_NUMBERS" == true ]]; then
            sed -n "${START},${END}p" "$FILE" | nl -v "$START"
        else
            sed -n "${START},${END}p" "$FILE"
        fi
        ;;
    cat)
        echo "=== Full content of $FILE (Total: $TOTAL_LINES lines) ==="
        if [[ "$SHOW_NUMBERS" == true ]]; then
            nl -v 1 "$FILE"
        else
            cat "$FILE"
        fi
        ;;
    search)
        echo "=== Searching for '$SEARCH_TERM' in $FILE ==="
        if [[ "$SHOW_NUMBERS" == true ]]; then
            grep -n --color=always "$SEARCH_TERM" "$FILE" || echo "No matches found."
        else
            grep --color=always "$SEARCH_TERM" "$FILE" || echo "No matches found."
        fi
        ;;
esac

echo ""
echo "File: $FILE | Total lines: $TOTAL_LINES"
