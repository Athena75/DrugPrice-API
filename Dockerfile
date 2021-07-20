FROM python:3.8

RUN useradd -ms /bin/bash api
USER api

ENV PYTHONPATH=/api
WORKDIR /api
EXPOSE 8000

COPY ./requirements.txt /api/requirements.txt

COPY ./api /api
COPY ./src /api/src

RUN pip install -r ./requirements.txt
ENV PATH="/home/api/.local/bin:${PATH}"

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]