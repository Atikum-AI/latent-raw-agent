# Latent Raw agent, sample mode: no GPU, no Nightshift, no database needed.
#   docker build -t latent-raw .
#   docker run --rm latent-raw                                  # runs all 3 eval cases
#   docker run --rm -p 8000:8000 latent-raw python run.py --serve  # POST /run
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SAMPLE_MODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# The sample clips, masks, EXRs and measurements live on Hugging Face (public); fetch them at build time
# so a clean clone runs: https://huggingface.co/datasets/caiotbr/latent-raw-agent-samples
RUN python scripts/fetch_samples.py

EXPOSE 8000
CMD ["python", "run.py", "--all"]
