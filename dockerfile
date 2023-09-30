# Use the base Python image
FROM python:3.11.4-slim

# Set the working directory inside the container
WORKDIR /dckr

# Install libGL for graphics support
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install GTK-related library
RUN apt-get update && apt-get install -y libglib2.0-0

# Install Git if needed
RUN apt-get update && apt-get install -y git

# Clean up after installation
RUN rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY TheCosmoBot/requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Create a directory for YOLOv5
RUN mkdir -p ./yolov5
RUN mkdir -p ./datasets
RUN mkdir -p ./datasets/clean

# Copy your YOLOv5 weights into the container
COPY TheCosmoBot/datasets/weights/best_weights.pt ./datasets/weights/

# Copy your source code and models to the working directory
COPY TheCosmoBot/bot.py .
COPY TheCosmoBot/config.json .
COPY TheCosmoBot/botik.png .

# Copy specific files from the host to the container
COPY TheCosmoBot/datasets/clean/camedons_clean_with_categories.csv ./datasets/clean/
COPY TheCosmoBot/datasets/clean/cuperoz_clean_with_categories.csv ./datasets/clean/
COPY TheCosmoBot/datasets/clean/acne_clean_with_categories.csv ./datasets/clean/

# Clone YOLOv5 repository and install it
RUN git clone https://github.com/ultralytics/yolov5.git ./yolov5 && \
    cd ./yolov5 && \
    pip install -U -r requirements.txt

# Set the command to run your bot when the container starts
CMD ["python", "bot.py"]
