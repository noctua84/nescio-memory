"""Trace metadata known only once the work is done.

What is known up front goes through langfuse's propagate_attributes, which is a
silent no-op when tracing is off. This is the counterpart for values such as a new
note's id, and it needs the guard below to be equally silent.
"""
from langfuse import get_client
from opentelemetry import trace


def annotate_span(**metadata: object) -> None:
    """Add metadata to the current span, if one is being recorded.

    Measured against langfuse 5.0.4 with no LANGFUSE_* keys: @observe produces no
    recording span, and update_current_span then logs two WARNINGs (a client without
    a public key, and "No active span") on every request. Deployments that leave
    tracing off are normal, so that must not be what their logs fill up with.

    Never pass path_or_uri, a query or note content: traces may leave the machine.
    """
    if trace.get_current_span().is_recording():
        get_client().update_current_span(metadata=metadata)
