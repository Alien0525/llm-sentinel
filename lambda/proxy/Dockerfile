FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

RUN yum install -y gcc gcc-c++ make && yum clean all

# Install onnxruntime 1.16.3 (latest available pre-built wheel for x86_64)
# Then install fastembed with --no-deps and install its other deps manually
RUN pip install --no-cache-dir \
    boto3 \
    requests \
    "numpy==1.26.4" \
    "onnxruntime==1.16.3" \
    "tokenizers>=0.15,<0.16" \
    "huggingface-hub>=0.20,<0.21" \
    "loguru>=0.7,<0.8" \
    tqdm

# Install fastembed without deps so it uses our pinned onnxruntime
RUN pip install --no-cache-dir --no-deps "fastembed==0.2.7"

# Pre-download model at build time
RUN python -c "from fastembed import TextEmbedding; model = TextEmbedding('BAAI/bge-small-en-v1.5'); print('Model loaded OK')"

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]