# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install gcloud CLI
RUN apt-get update && apt-get install -y curl \
    && curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-367.0.0-linux-x86_64.tar.gz \
    && tar -xf google-cloud-sdk-367.0.0-linux-x86_64.tar.gz \
    && ./google-cloud-sdk/install.sh

# Authenticate gcloud (this step might need to be done manually after container is running)
# RUN gcloud auth login
# RUN gcloud auth application-default login

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py", "-c", "download.ini"]