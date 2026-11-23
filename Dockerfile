FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads && sed -i 's/\r$//' start.sh && chmod +x start.sh
EXPOSE 8000
CMD ["./start.sh"]
