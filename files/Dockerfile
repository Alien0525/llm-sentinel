FROM public.ecr.aws/lambda/python:3.11

COPY requirements-runtime.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY model/model.pkl /var/task/model.pkl
COPY handler.py /var/task/handler.py

CMD ["handler.lambda_handler"]