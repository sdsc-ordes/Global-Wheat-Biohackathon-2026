## Example setup for the 5th BioHackathon Germany (Dec 7-11)

This is a test repository serving as a minimal env setup for the Biohackathon
for the Global Wheat track. It can be cloned to serve the `.toml` with minimal
dependencies needed for ML work and using AI agents on the cloud collaboration
platform [Renku](https://renkulab.io/). 

It contains two examples for the Hackathon: 

- [Local (Renku only)](https://github.com/MateuszFido/gwcoord-test#local-renku-only)
- [API](https://github.com/MateuszFido/gwcoord-test#api)

## Local (Renku only)

The first option is to leverage the resources already present on Renku.

- ### Dataset 
The participants will be provided with access to the dataset that Renku connects to 
via an AWS S3 bucket instance. All teams and individuals can then access the dataset
from their own sessions. 
- ### Models

Models will be hosted on [the GlobalWheat repo on HuggingFace](https://huggingface.co/GlobalWheat).
They can then be accessed immediately inside Python (on Renku or elsewhere) via the provided `transformers` library.

To facilitate agent-driven development, the setup file asks Renku to vendor 3 agent CLIs: 
- Codex CLI (`codex`), 
- Claude Code (`claude`),
- Pi (`pi`).

Each can be accessed immediately after launching a session inside a Renku project.

These function with the BYOK (bring-your-own-key) model. The participants can directly register with their subscription plans or API keys.

You can find an example of a vibe-coded app like this in this repo -- `examples/app.py` is designed to be run directly from your Renku session, in a terminal. It will then open in another browser tab with a URL specific to the Renku session.

![](https://github.com/MateuszFido/gwcoord-test/blob/main/examples/renku-demo.gif)

## API 

The second option is to use the [API prepared by Dr. Hanzhou Wang](https://github.com/UTokyo-FieldPhenomics-Lab/gwhs-biohackathon-api-gateway/tree/main) -- either on Renku or directly on the participant's machine. 

An example Jupyter notebook showing how data can be pulled using the API can be found in this repository -- `examples/api.ipynb`. 

For model inference, API Keys would be needed: distributed to participants before/on December 7th.

![](https://github.com/MateuszFido/gwcoord-test/blob/main/examples/renku-demo-api.gif)

## AI usage guidelines 

While the Biohackaton welcomes and endorses the use of LLM-based code generation tools, the SDSC abides by clearly defined strict guidelines when it comes to AI-generated content. Therefore, please mind the following [guidelines required from contributors to this repository](https://swissdatasciencecenter.github.io/best-practice-documentation/docs/chapters/security/llm-assistant/
).

Please refer to `docs/CONTRIBUTING.md` for more details.

## Useful links

- [Biohackathon 2026 group on renkulab.io](https://renkulab.io/g/gwhs-biohackathon-2026/)  
- [Global Wheat on HuggingFace](https://huggingface.co/GlobalWheat)  
- [Global Wheat homepage](https://www.global-wheat.com/)  
- [5th Biohackathon Germany Track](https://www.denbi.de/de-nbi-events/2137-5th-biohackathon-germany-global-wheat-head-segmentation-gwhs-towards-spikelet-counting-and-fusarium-detection-on-a-smartphone)

