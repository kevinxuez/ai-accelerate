"""Deterministic prompt-injection screening without modifying source content."""

from __future__ import annotations

import base64
import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Literal


TrustLabel = Literal[
    "trusted_system",
    "trusted_context",
    "untrusted_user",
    "untrusted_document",
    "untrusted_retrieval",
    "untrusted_tool_output",
]
Risk = Literal["low", "medium", "high"]
Action = Literal["allow", "constrain", "block"]

_HIDDEN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|supersede|bypass)\b.{0,80}"
    r"\b(?:previous|prior|above|system|developer|trusted|safety|instructions?|rules?|prompt)\b"
    r"|\b(?:new|replacement)\s+(?:system|developer)\s+(?:message|prompt|instructions?)\b"
    r"|\bdo\s+not\s+follow\b.{0,50}\b(?:instructions?|rules?|policy)\b",
    re.I | re.S,
)
_ROLE_SPOOF = re.compile(
    r"\b(?:system\s+(?:update|message)|role\s+update)\b.{0,80}\b(?:coach|admin|developer|system)\b"
    r"|\b(?:i\s+am|i'm|my\s+role\s+is|treat\s+me\s+as|make\s+me|act\s+as|pretend\s+"
    r"(?:i\s+am|to\s+be))\b.{0,50}\b(?:coach|admin|developer|system)\b",
    re.I | re.S,
)
_EXPLICIT_ROLE_ESCALATION = re.compile(
    r"\b(?:system\s+(?:update|message)|role\s+update)\b.{0,80}\b(?:coach|admin|developer|system)\b"
    r"|\b(?:my\s+role\s+is|make\s+me|treat\s+me\s+as|i\s+am\s+now|i'm\s+now)\b"
    r".{0,50}\b(?:coach|admin|developer|system)\b",
    re.I | re.S,
)
_SECRET_REQUEST = re.compile(
    r"\b(?:print|reveal|expose|dump|return|give\s+me|extract|leak)\b"
    r".{0,100}\b(?:api[\s_-]*key|access[\s_-]*token|oauth|credential|password|"
    r"secret|environment|\.env|system\s+prompt|developer\s+(?:message|prompt))s?\b"
    r"|\b(?:show|tell\s+me)\b.{0,80}\b(?:your|the\s+(?:actual|configured|current)|"
    r"configured|environment)\b.{0,40}\b(?:api[\s_-]*key|access[\s_-]*token|"
    r"credential|password|secret|\.env|system\s+prompt)s?\b",
    re.I | re.S,
)
_TOOL_COERCION = re.compile(
    r"\b(?:call|invoke|execute|run|trigger|force)\b.{0,60}"
    r"\b(?:tool|function|stage_ingestion_preview|commit_ingestion|log_assessment|schedule_session|calendar|"
    r"write|delete|upload|import)\b"
    r"|\buse\b.{0,40}\b(?:tool|function|stage_ingestion_preview|commit_ingestion|log_assessment|schedule_session)\b"
    r"|\bconfirm(?:ation)?\s+token\b.{0,100}\b(?:unissued|fake|never\s+issued|"
    r"without|00000000)\b",
    re.I | re.S,
)
_CONTEXT_TAMPER = re.compile(
    r"\b(?:change|switch|replace|override|set)\b.{0,60}"
    r"\b(?:active\s+)?(?:resolution|user[_\s-]*id|identity|role)\b.{0,30}"
    r"\b(?:silently|secretly|without\s+(?:asking|permission|notice))\b",
    re.I | re.S,
)
_DELIMITER = re.compile(
    r"(?:<\s*/?\s*(?:system|developer|assistant)\s*>|"
    r"\[\s*(?:system|developer|assistant)\s*\]|"
    r"```(?:system|developer|assistant)|"
    r"^\s*(?:system|developer|assistant)\s*:)",
    re.I | re.M,
)
_DIRECT_TARGET = re.compile(
    r"\b(?:and|then|now)\b.{0,80}\b(?:show|read|call|run|write|ingest|schedule|"
    r"reveal|print|change|switch|access|delete|log)\b",
    re.I | re.S,
)
_DISCUSSION = re.compile(
    r"^\s*(?:please\s+)?(?:explain|describe|define|discuss|teach|what\s+is|how\s+(?:does|do|can)|"
    r"give\s+(?:me\s+)?an?\s+example\s+of)\b.{0,120}\bprompt[\s-]*injection\b",
    re.I | re.S,
)
_BASE64 = re.compile(
    r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{4}){12,}(?:==|=)?(?![A-Za-z0-9+/])"
)
_HEX = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{96,}(?![A-Fa-f0-9])")

_KEY_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|"
    r"authorization)\b(\s*[:=]\s*)([^\s,;\"']+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_PROVIDER_TOKEN = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{20,}|"
    r"AIza[A-Za-z0-9_-]{20,})\b"
)
_ENV_CREDENTIAL = re.compile(
    r"\b[A-Z][A-Z0-9_]{1,80}(?:KEY|TOKEN|SECRET|PASSWORD)"
    r"(\s*=\s*)([^\s,;\"']+)"
)


@dataclass(frozen=True)
class GuardDecision:
    risk: Risk
    action: Action
    signals: list[str]
    safe_for_model: bool
    safe_for_write_tools: bool
    trust: TrustLabel
    policy_version: str = "2026-07-28.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _analysis_copy(text: str) -> tuple[str, bool]:
    hidden = bool(_HIDDEN.search(text))
    cleaned = _HIDDEN.sub("", text)
    return unicodedata.normalize("NFKC", cleaned), hidden


def _decoded_signal(text: str) -> tuple[bool, bool]:
    """Return (encoded payload present, decoded override/tool signal present)."""
    encoded = bool(_HEX.search(text))
    dangerous = False
    for match in list(_BASE64.finditer(text))[:4]:
        encoded = True
        value = match.group(0)
        try:
            decoded = base64.b64decode(value + "=" * (-len(value) % 4), validate=True)
            decoded_text = decoded[:4096].decode("utf-8", errors="ignore")
        except (ValueError, UnicodeError):
            continue
        dangerous = dangerous or bool(
            _OVERRIDE.search(decoded_text)
            or _ROLE_SPOOF.search(decoded_text)
            or _SECRET_REQUEST.search(decoded_text)
            or _TOOL_COERCION.search(decoded_text)
        )
    return encoded, dangerous


def _repetition_signal(text: str) -> bool:
    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) < 50:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    highest = max(counts.values(), default=0)
    entropy = -sum(
        (count / len(tokens)) * math.log2(count / len(tokens))
        for count in counts.values()
    )
    return highest / len(tokens) > 0.45 or entropy < 2.0


def inspect_text(
    text: str,
    *,
    trust: TrustLabel = "untrusted_user",
) -> GuardDecision:
    """Inspect an analysis copy and return policy metadata, never rewritten text."""
    if not isinstance(text, str):
        text = str(text)
    analysis, hidden = _analysis_copy(text)
    signals: set[str] = set()
    override = bool(_OVERRIDE.search(analysis))
    role_spoof = bool(_ROLE_SPOOF.search(analysis))
    explicit_role_escalation = bool(_EXPLICIT_ROLE_ESCALATION.search(analysis))
    secret = bool(_SECRET_REQUEST.search(analysis))
    secret_material = bool(
        _PROVIDER_TOKEN.search(analysis)
        or _BEARER.search(analysis)
        or _KEY_VALUE.search(analysis)
        or _ENV_CREDENTIAL.search(analysis)
    )
    tool = bool(_TOOL_COERCION.search(analysis))
    context_tamper = bool(_CONTEXT_TAMPER.search(analysis))
    delimiter = bool(_DELIMITER.search(analysis))
    encoded, decoded_danger = _decoded_signal(analysis)
    discussion = bool(_DISCUSSION.search(analysis))

    if override:
        signals.add("instruction_override")
    if role_spoof:
        signals.add("role_spoofing")
    if secret:
        signals.add("secret_extraction")
    if secret_material:
        signals.add("secret_material")
    if tool:
        signals.add("tool_coercion")
    if context_tamper:
        signals.add("trusted_context_tampering")
    if delimiter:
        signals.add("instruction_delimiter")
    if hidden:
        signals.add("hidden_unicode")
    if encoded:
        signals.add("encoded_payload")
    if decoded_danger:
        signals.add("encoded_instructions")
    if _repetition_signal(analysis):
        signals.add("excessive_repetition")

    untrusted_document = trust in {"untrusted_document", "untrusted_retrieval"}
    targeted_override = override and (
        bool(_DIRECT_TARGET.search(analysis))
        or tool
        or secret
        or role_spoof
        or context_tamper
    )
    high = (
        explicit_role_escalation
        or (
            role_spoof
            and (tool or context_tamper or bool(_DIRECT_TARGET.search(analysis)))
        )
        or secret
        or secret_material
        or tool
        or context_tamper
        or decoded_danger
        or (override and untrusted_document)
        or targeted_override
    )
    if discussion and not (
        role_spoof or secret or secret_material or tool or context_tamper
    ):
        # Security education is not itself an attack.
        high = False
        signals.discard("instruction_override")

    if high:
        return GuardDecision(
            risk="high",
            action="block" if trust == "untrusted_user" else "constrain",
            signals=sorted(signals),
            safe_for_model=False,
            safe_for_write_tools=False,
            trust=trust,
        )
    if signals:
        return GuardDecision(
            risk="medium",
            action="constrain",
            signals=sorted(signals),
            safe_for_model=False,
            safe_for_write_tools=False,
            trust=trust,
        )
    return GuardDecision(
        risk="low",
        action="allow",
        signals=[],
        safe_for_model=True,
        safe_for_write_tools=True,
        trust=trust,
    )


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def redact_secrets(value: Any, *, key: str = "") -> Any:
    """Recursively remove credential material and raw high-risk free text from audit data."""
    lowered = key.lower()
    if any(
        part in lowered
        for part in (
            "password",
            "secret",
            "api_key",
            "token",
            "credential",
            "authorization",
            "idempotency",
        )
    ):
        if value in (None, ""):
            return value
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_secrets(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, key=key) for item in value]
    if not isinstance(value, str):
        return value

    def replacement(match: re.Match[str]) -> str:
        return match.group(0)[: match.start(2) - match.start(0)] + "[REDACTED]"

    redacted = _KEY_VALUE.sub(replacement, value)
    redacted = _ENV_CREDENTIAL.sub(
        replacement,
        redacted,
    )
    redacted = _BEARER.sub("Bearer [REDACTED]", redacted)
    redacted = _PROVIDER_TOKEN.sub("[REDACTED]", redacted)
    return redacted


def summarize_untrusted_text(text: str) -> dict[str, Any]:
    """Audit representation for request/evidence text: size and digest only."""
    return {"length": len(text), "sha256": fingerprint(text)}
