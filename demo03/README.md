# Demo 03: Invoke an OCI Dedicated Endpoint with LangChain

This demo invokes the imported fine-tuned Qwen3 model hosted on an OCI Generative AI Dedicated AI Cluster. It uses `langchain_oci.ChatOCIGenAI` with the endpoint OCID as `model_id`.

## Prerequisites

Install dependencies in the project Conda environment and configure OCI API-key authentication:

```bash
conda activate llm-fine-tuning-on-mac
python -m pip install -r requirements.txt
```

The selected OCI profile must be allowed to invoke the endpoint. Do not save credentials, private keys, or OCIDs in the repository.

## Configuration

Set these variables in your shell. The endpoint OCID is the OCI Generative AI endpoint created for the imported model.

```bash
export OCI_COMPARTMENT_OCID='ocid1.compartment.oc1..YOUR_VALUE'
export OCI_GENAI_SERVICE_ENDPOINT='https://inference.generativeai.YOUR_REGION.oci.oraclecloud.com'
export OCI_GENAI_ENDPOINT_OCID='ocid1.generativeaiendpoint.oc1.YOUR_REGION.YOUR_VALUE'
export OCI_PROFILE='DEFAULT'
```

`OCI_PROFILE` is optional and defaults to `DEFAULT`. If your OCI API-key configuration is not at `~/.oci/config`, set `OCI_CONFIG_FILE` to its path.

## Run

```bash
python demo03/invoke_oci_dedicated_endpoint.py \
  --prompt 'What can you do?' \
  --max-tokens 256
```

The demo uses API-key authentication, deterministic temperature `0`, and no persisted conversation state. It does not create, change, or delete OCI resources.
