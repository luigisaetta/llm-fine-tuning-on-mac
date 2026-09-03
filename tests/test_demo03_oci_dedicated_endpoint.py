"""
Author: L. Saetta
Date last modified: 2026-09-03
License: MIT
Description: Unit tests for OCI Dedicated AI Cluster endpoint demo configuration and LangChain client setup.
"""

import pytest

import demo03.invoke_oci_dedicated_endpoint as endpoint_demo


def valid_environment() -> dict[str, str]:
    """Return non-sensitive valid OCI endpoint settings for tests."""
    return {
        "OCI_COMPARTMENT_OCID": "ocid1.compartment.oc1..example",
        "OCI_GENAI_SERVICE_ENDPOINT": "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
        "OCI_GENAI_ENDPOINT_OCID": "ocid1.generativeaiendpoint.oc1.eu-frankfurt-1.example",
    }


def test_read_settings_requires_all_endpoint_environment_variables() -> None:
    """Missing OCI settings produce an actionable error before endpoint invocation."""
    with pytest.raises(RuntimeError, match="OCI_COMPARTMENT_OCID"):
        endpoint_demo.read_settings({})


def test_read_settings_rejects_non_endpoint_model_ocid() -> None:
    """Only endpoint OCIDs may be supplied as the dedicated model ID."""
    environment = valid_environment()
    environment["OCI_GENAI_ENDPOINT_OCID"] = "ocid1.generativeaimodel.oc1.eu-frankfurt-1.example"

    with pytest.raises(RuntimeError, match="endpoint OCID"):
        endpoint_demo.read_settings(environment)


def test_create_chat_model_uses_endpoint_ocid_and_generic_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured LangChain client targets the dedicated endpoint in generic mode."""
    captured_kwargs: dict[str, object] = {}

    class FakeChatOCIGenAI:
        """Minimal ChatOCIGenAI replacement that captures constructor arguments."""

        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(endpoint_demo, "ChatOCIGenAI", FakeChatOCIGenAI)
    settings = endpoint_demo.read_settings(valid_environment())

    endpoint_demo.create_chat_model(settings, max_tokens=128)

    assert captured_kwargs["model_id"] == settings.endpoint_ocid
    assert captured_kwargs["provider"] == "generic"
    assert captured_kwargs["auth_type"] == "API_KEY"
    assert captured_kwargs["model_kwargs"] == {"temperature": 0, "max_tokens": 128}


def test_invoke_prompt_returns_response_content() -> None:
    """The demo returns the text content from the LangChain endpoint response."""

    class FakeChatModel:
        """Minimal chat model response double."""

        def invoke(self, _messages: object):
            """Return a simple response object."""
            return type("Response", (), {"content": "Endpoint response"})()

    assert endpoint_demo.invoke_prompt(FakeChatModel(), "Hello") == "Endpoint response"
