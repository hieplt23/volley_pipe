FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxcb1 \
    libxrender1 \
    libxi6 \
    libqt5x11extras5 \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# copy project files
COPY config/ /workspace/config/
COPY data/ /workspace/data/
COPY models/ /workspace/models/
COPY inputs/ /workspace/inputs/
COPY src/ /workspace/src/
COPY requirements.txt /workspace/
COPY main.py /workspace/


# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set the working directory
WORKDIR /workspace

# run the application -> results will be saved in outputs/videos
CMD ["python", "main.py"]