FROM python:3.12-slim
 
WORKDIR /workspace

copy requirements.txt requirements.txt

# Activate the virtual environment and install requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]