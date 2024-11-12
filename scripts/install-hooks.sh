#!/bin/sh

# Install pre-commit hooks
pre-commit install

# Make sure the script is executable
chmod +x .git/hooks/pre-commit

echo "Pre-commit hooks installed successfully!"
