#!/bin/bash

# Define project name
PROJECT_NAME="ai-fitness-tracker"

# Create the root project directory
mkdir -p $PROJECT_NAME

# Navigate into the directory
cd $PROJECT_NAME

# Create sub-directories
mkdir -p src
mkdir -p models

# Create the files in the root
touch app.py
touch config.py
touch requirements.txt
touch packages.txt
touch environment.yml
touch README.md
touch .gitignore

# Create the files in src directory
touch src/__init__.py
touch src/processor.py
touch src/utils.py

echo "Structure for '$PROJECT_NAME' created successfully!"