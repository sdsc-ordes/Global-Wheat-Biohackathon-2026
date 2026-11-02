## PROMPTS.md

Initial prompt to the LLM:

```{text}
Create a working Streamlit app in app.py in the current directory.

Build a "Wheat Image Explorer" using streamlit, requests, and Pillow,
all of which are already installed.

Requirements:
- A dropdown selecting between these two public images:
  GWHD: https://data.global-wheat.com/gwhd/images/00000000.jpg
  GWFSS: https://data.global-wheat.com/gwfss/images/00000000.png
- Fetch images server-side with a request timeout and cache downloads.
- Display the selected image and its original pixel dimensions.
- Add a contrast slider that updates the displayed image, using Pillow.
- Add a button to download the original image.
- Show errors if downloading or decoding fails.

Do not read secrets, do not change environment files. 
```

Then, start the server (from repo root):

```{sh}
python -m streamlit run examples/app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  --server.runOnSave=true \
  --server.baseUrlPath="${RENKU_BASE_URL_PATH%/}/proxy/absolute/8501" \
  --browser.serverAddress="$(python -c 'import os; from urllib.parse import urlsplit; print(urlsplit(os.environ["RENKU_BASE_URL"]).hostname)')" \
  --browser.serverPort=443
```

Stop the server with Ctrl+C from the Renku terminal.

