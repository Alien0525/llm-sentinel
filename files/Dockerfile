FROM public.ecr.aws/lambda/python:3.11
 
# Install Python dependencies
COPY requirements-runtime.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy model artifact and Lambda handler
COPY model/model.pkl   /var/task/model.pkl
COPY handler.py        /var/task/handler.py
 
# Runtime environment variables
ENV MODEL_PATH=/var/task/model.pkl
ENV ATTACK_THRESHOLD=0.5
 
# Lambda entrypoint: filename.functionname
CMD ["handler.handler"]