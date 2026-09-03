"""
Author: L. Saetta
Date last modified: 2026-09-03
License: MIT
Description: Invoke an OCI Generative AI Dedicated AI Cluster endpoint with LangChain ChatOCIGenAI.
"""

import argparse
import os
import sys
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_oci import ChatOCIGenAI


DEFAULT_PROMPT = "In one sentence, introduce yourself and describe how you can help."
DEFAULT_MAX_TOKENS = 256
ENDPOINT_OCID_PREFIX = "ocid1.generativeaiendpoint"


@dataclass(frozen=True)
class EndpointSettings:
    """Configuration needed to invoke one OCI Generative AI dedicated endpoint."""

    compartment_ocid: str
    service_endpoint: str
    endpoint_ocid: str
    profile: str
    config_file: str | None


def read_settings(environment: dict[str, str]) -> EndpointSettings:
    """Read and validate OCI dedicated-endpoint settings from environment variables.

    Args:
        environment: Environment-variable mapping, injectable for tests.

    Returns:
        Validated endpoint settings.

    Raises:
        RuntimeError: If a required setting is missing or the endpoint OCID is invalid.
    """
    required_names = ("OCI_COMPARTMENT_OCID", "OCI_GENAI_SERVICE_ENDPOINT", "OCI_GENAI_ENDPOINT_OCID")
    missing_names = [name for name in required_names if not environment.get(name)]
    if missing_names:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_names)}.")

    endpoint_ocid = environment["OCI_GENAI_ENDPOINT_OCID"]
    if not endpoint_ocid.startswith(ENDPOINT_OCID_PREFIX):
        raise RuntimeError(
            "OCI_GENAI_ENDPOINT_OCID must be an OCI Generative AI endpoint OCID "
            f"starting with '{ENDPOINT_OCID_PREFIX}'."
        )

    return EndpointSettings(
        compartment_ocid=environment["OCI_COMPARTMENT_OCID"],
        service_endpoint=environment["OCI_GENAI_SERVICE_ENDPOINT"],
        endpoint_ocid=endpoint_ocid,
        profile=environment.get("OCI_PROFILE", "DEFAULT"),
        config_file=environment.get("OCI_CONFIG_FILE"),
    )


def create_chat_model(settings: EndpointSettings, max_tokens: int) -> ChatOCIGenAI:
    """Create a deterministic LangChain client for the dedicated Qwen endpoint.

    Args:
        settings: Validated OCI endpoint settings.
        max_tokens: Maximum number of tokens generated in the response.

    Returns:
        Configured ChatOCIGenAI instance.
    """
    client_kwargs: dict[str, object] = {
        "auth_type": "API_KEY",
        "auth_profile": settings.profile,
        "model_id": settings.endpoint_ocid,
        "provider": "generic",
        "service_endpoint": settings.service_endpoint,
        "compartment_id": settings.compartment_ocid,
        "model_kwargs": {"temperature": 0, "max_tokens": max_tokens},
    }
    if settings.config_file:
        client_kwargs["auth_file_location"] = settings.config_file
    return ChatOCIGenAI(**client_kwargs)


def invoke_prompt(chat_model: ChatOCIGenAI, prompt: str) -> str:
    """Invoke a prompt and return the text content from the endpoint response.

    Args:
        chat_model: Configured LangChain OCI chat model.
        prompt: User prompt to send to the endpoint.

    Returns:
        Text response returned by the model.
    """
    response = chat_model.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)


def parse_arguments() -> argparse.Namespace:
    """Parse prompt and response-length options for the endpoint demo."""
    parser = argparse.ArgumentParser(description="Invoke an OCI Generative AI Dedicated AI Cluster endpoint with LangChain.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the deployed model.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    return parser.parse_args()


def main() -> None:
    """Load settings, invoke the endpoint once, and print the model response."""
    args = parse_arguments()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be a positive integer.")
    settings = read_settings(dict(os.environ))
    chat_model = create_chat_model(settings, args.max_tokens)
    print(invoke_prompt(chat_model, args.prompt))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Endpoint invocation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
