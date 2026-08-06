"""Configured NSDA-compatible topic lookup tool."""

from __future__ import annotations

from datetime import date as calendar_date
from typing import Any

from pydantic import ValidationError

from casefile.agents.errors import ErrorCode
from casefile.providers.nsda import (
    NSDANotFound,
    NSDAProviderDisabled,
    NSDAProviderError,
    build_nsda_provider,
)

from .contracts import CurrentTopicArgs
from .context import ToolContext, ToolRuntime


class TopicTools:
    def __init__(self, runtime: ToolRuntime) -> None:
        self.runtime = runtime

    def get_current_topic(
        self,
        context: ToolContext,
        *,
        event: str = "Public Forum",
        as_of: str | None = None,
    ) -> dict[str, Any] | str:
        self.runtime.authorize(
            context,
            tool="get_current_topic",
            action="look up the current topic",
            roles=frozenset({"student", "coach"}),
            agents=frozenset({"evidence_librarian"}),
        )
        try:
            args = CurrentTopicArgs(event=event, as_of=as_of)
            requested_date = (
                calendar_date.fromisoformat(args.as_of) if args.as_of else None
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "get_current_topic", exc)
        except ValueError as exc:
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "as_of must be a valid date in YYYY-MM-DD format.",
                tool="get_current_topic",
                cause=exc,
            )

        provider = None
        try:
            provider = build_nsda_provider(self.runtime.settings)
            topic = provider.current_topic(args.event, as_of=requested_date)
            metadata = provider.metadata()
        except NSDANotFound:
            result: dict[str, Any] | str = (
                f"No NSDA-compatible {args.event} topic is available"
                + (f" for {args.as_of}." if args.as_of else ".")
            )
        except NSDAProviderDisabled as exc:
            self.runtime.fail(
                context,
                ErrorCode.CAPABILITY_DISABLED,
                "The NSDA provider capability is disabled.",
                tool="get_current_topic",
                cause=exc,
            )
        except NSDAProviderError as exc:
            self.runtime.fail(
                context,
                ErrorCode.NSDA_UPSTREAM_ERROR,
                "The configured NSDA-compatible provider could not return a topic.",
                tool="get_current_topic",
                cause=exc,
                retryable=True,
            )
        except ValueError as exc:
            self.runtime.fail(
                context,
                ErrorCode.CONFIGURATION_ERROR,
                "The configured NSDA-compatible provider is invalid.",
                tool="get_current_topic",
                cause=exc,
            )
        else:
            result = {
                "topic": topic,
                "provider": metadata.get("provider", "NSDA-compatible provider"),
                "backend": provider.backend,
                "dataset_version": metadata.get("dataset_version"),
                "fixture": bool(metadata.get("fixture")),
                "synthetic": bool(metadata.get("synthetic")),
                "disclaimer": metadata.get("disclaimer", ""),
            }
        finally:
            close = getattr(provider, "close", None) if provider is not None else None
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    self.runtime.fail(
                        context,
                        ErrorCode.NSDA_UPSTREAM_ERROR,
                        "The configured NSDA-compatible provider could not close cleanly.",
                        tool="get_current_topic",
                        cause=exc,
                        retryable=True,
                    )

        self.runtime.audit(
            context,
            "get_current_topic",
            {"event": args.event, "as_of": args.as_of},
            result,
        )
        return result
