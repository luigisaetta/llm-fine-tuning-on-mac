# Demo 03: OCI Dedicated Endpoint Inference

## Problem

After importing and deploying the locally fine-tuned Qwen3 model on an OCI Generative AI Dedicated AI Cluster, the project needs a minimal, reproducible client that validates the deployed endpoint with LangChain.

## Scope

Create `demo03/` with a Python script and usage README. The script invokes one prompt through `langchain_oci.ChatOCIGenAI` against an OCI Generative AI dedicated endpoint and prints the response. It uses the endpoint OCID as the model ID.

## Assumptions

* The imported fine-tuned model has an active OCI Generative AI endpoint on a Dedicated AI Cluster.
* OCI API-key credentials are configured locally in the selected OCI profile.
* The caller has permission to invoke the endpoint and knows its compartment OCID, endpoint OCID, and regional Generative AI inference URL.
* `langchain` and `langchain-oci` are installed from `requirements.txt` in the project Conda environment.

## Functional requirements

* Read `OCI_COMPARTMENT_OCID`, `OCI_GENAI_SERVICE_ENDPOINT`, and `OCI_GENAI_ENDPOINT_OCID` from the environment. Treat them as required and never write their values to source control.
* Support optional `OCI_PROFILE`, defaulting to `DEFAULT`, and use `API_KEY` authentication with the selected OCI profile.
* Accept an optional `--prompt`; use a safe, non-sensitive default prompt when omitted.
* Create `ChatOCIGenAI` with the endpoint OCID as `model_id`, deterministic temperature `0`, and a visible maximum-output-token configuration.
* Print only the model response and actionable errors. Do not upload data, create or modify OCI resources, or persist prompts and responses.

## Out of scope

Streaming, chat history, tool calling, retrieval, endpoint creation, model import, and alternate OCI authentication modes are out of scope.

## Acceptance criteria

* The demo has English usage documentation with environment-variable names and a command example that does not contain credentials or OCIDs.
* Unit tests validate required environment variables and the `ChatOCIGenAI` construction parameters without invoking OCI.
* The script imports `ChatOCIGenAI` from `langchain_oci` and uses the endpoint OCID as `model_id`.

## Verification

* Install dependencies with `python -m pip install -r requirements.txt` in the project Conda environment.
* Run `pytest -q` without OCI network calls.
* Run the script against the active endpoint after setting the required environment variables and inspect the returned response.
