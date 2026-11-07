FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/closedloop_oa

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY htc_twin ./htc_twin
COPY recipes ./recipes
COPY procedures ./procedures
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["closedloop-oa"]
CMD ["--help"]
