"""Tests for provider response normalization."""

from types import SimpleNamespace

from structagent.providers import _openai_message_content


def test_openai_message_content_handles_list_parts():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "text", "text": "Hotspot analysis"},
                        SimpleNamespace(text="completed."),
                    ]
                )
            )
        ]
    )

    assert _openai_message_content(response) == "Hotspot analysis\ncompleted."
