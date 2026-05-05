FROM python:3.11-slim

WORKDIR /var/task

COPY requirements-lambda.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: -r requirements-lambda.txt

COPY app.py .
COPY model/ ./model/

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]

CMD ["app.lambda_handler"]