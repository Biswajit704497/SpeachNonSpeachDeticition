FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirments.txt
EXPOSE $PORT
CMD gunicon --workers=4 --bind 0.0.0.0:$PORT app:app
