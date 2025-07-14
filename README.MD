**Projektbeskrivelse og formål**


**Installation og opsætning (fx miljø, dependencies)**

Tjek blandt andet requirements.txt

**Instruktioner til hvordan man kører koden eller notebooks**


**Forklaring af datasæt og kilde**


**Resultater og konklusioner**


cd "C:\Job_og_eksamensbevis\Github\projekter\RF_project"

git init

git add .

git commit -m "initial commit"

git remote add origin https://github.com/JeppeFons/RF_FASTAPI_MLflow_project.git

git push -u origin main

git branch -m master main

git push -u origin main



mlflow server `
  --backend-store-uri "sqlite:///mlflow.db" `
  --default-artifact-root ".\mlruns" `
  --host 127.0.0.1 `
  --port 5000
