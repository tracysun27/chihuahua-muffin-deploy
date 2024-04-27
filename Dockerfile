# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for gunicorn
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --workers=3"

# Run gunicorn when the container launches
CMD ["gunicorn", "app:app"]
