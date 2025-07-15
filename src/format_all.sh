#!/bin/bash

# Set the base directory (current directory by default)
BASE_DIR=${1:-.}

# Find and format all .cpp and .hpp files
find "$BASE_DIR" \( -name '*.cpp' -o -name '*.hpp' \) -exec clang-format -i {} +

echo "Formatting complete."
