FROM python:3.11-slim

# Install rsync
RUN apt-get update && apt-get install -y rsync

# Set the working directory
WORKDIR /app

# Set the PYTHONUNBUFFERED environment variable
ENV PYTHONBUFFERED=1

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r /app/requirements.txt

# Copy unittest into container
COPY test_answer.py /app/test_answer.py

# Copy monitor into container
COPY monitor.py /app/monitor.py

# Create the notebooks directory
RUN mkdir -p /app/notebooks
RUN mkdir -p /app/notebooks/images

# Copy images into the container
COPY ./images/ /app/notebooks/images/

# Copy blank exercise template into a location that won't be overwritten
COPY exercise_template.ipynb /app/exercise_template.ipynb

# Copy the start script into the container
COPY start_fresh_jupyter.sh /app/start_fresh_jupyter.sh

# Make the start script executable
RUN chmod +x /app/start_fresh_jupyter.sh

# Expose the port for Jupyter notebook
EXPOSE 8888
CMD ["/app/start_fresh_jupyter.sh"]