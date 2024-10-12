FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./main.py /code/
COPY ./desc2024_v2.csv /code/

CMD ["fastapi", "run", "main.py", "--port", "8000"]