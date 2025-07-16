## Projektbeskrivelse og formål


## Installation og opsætning (fx miljø, dependencies)

Tjek blandt andet requirements.txt

## Instruktioner til hvordan man kører koden eller notebooks

**start mlflow server:**

mlflow server `
  --backend-store-uri "sqlite:///mlflow.db" `
  --default-artifact-root ".\mlruns" `
  --host 127.0.0.1 `
  --port 5000


**model skal gemmes i roden af projektet sammen med main.py og så kør nedenstående for at starte appen**

uvicorn main:app --reload

**gå til** 

http://127.0.0.1:8000/docs


## Forklaring af datasæt og kilde

Link til datasæt:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Resultater og konklusioner