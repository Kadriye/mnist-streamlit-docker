FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev

# Install lightweight Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch and torchvision directly from PyTorch URL
RUN pip install --no-cache-dir torch torchvision \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html


# Copy the rest of the app
COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
