FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights into the image layer
# Avoids ~4GB download on every cold start
RUN python -c "\
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor; \
ColQwen2_5.from_pretrained('vidore/colqwen2.5-v0.1'); \
ColQwen2_5_Processor.from_pretrained('vidore/colqwen2.5-v0.1')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
