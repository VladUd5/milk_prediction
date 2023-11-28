FROM python:3.10
WORKDIR /app_2

COPY requirements.txt /app_2/requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install --upgrade pyarrow
RUN pip install --upgrade scikit-learn
COPY . /app_2

VOLUME /app_2

RUN chmod +x /app_2/baseline.py
CMD ["python3","/app_2/baseline.py"]
