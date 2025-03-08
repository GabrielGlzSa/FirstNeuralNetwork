#!/bin/bash

# Copy the notebook from the source to the target location
cp /app/exercise_template.ipynb /app/notebooks/exercise.ipynb

# Start the Jupyter server
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/notebooks/ --NotebookApp.default_url=/app/notebooks/exercise.ipynb