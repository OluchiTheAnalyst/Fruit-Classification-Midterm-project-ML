# 1. Base image: a slim Python environment
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code into the container
COPY . .

# 5. Expose the port your API will run on
EXPOSE 9696

# 6. Specify the command to run your API
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]
