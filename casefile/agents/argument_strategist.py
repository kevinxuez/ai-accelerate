"""Evidence-bounded structured argument generation and revision."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from casefile.tools.context import ToolContext

from .contracts import (
    ArgumentDraft,
    ArgumentRequest,
    ArgumentSection,
    EvidencePacket,
    Side,
)
from .errors import CaseFileError, ErrorCode
from .prompt_registry import load_prompt


STRATEGIST = "argument_strategist"
ARGUMENT_SECTION_NAMES = (
    "claim",
    "warrant",
    "evidence",
    "impact",
    "resolution_link",
    "likely_response",
)
_SPEECH_OR_SCRIPT = re.compile(
    r"(?im)^\s*(?:"
    r"good\s+(?:morning|afternoon|evening)[,!]?\s+(?:judge|judges|everyone)"
    r"|judge(?:s)?\s*[,!:]"
    r"|ladies\s+and\s+gentlemen\b"
    r"|my\s+partner\s+and\s+i\b"
    r"|(?:first|second|third)?\s*(?:speaker|constructive|rebuttal)\s*:"
    r"|(?:vote|please\s+vote)\s+(?:pro|con|affirmative|negative)\b"
    r"|we\s+urge\s+(?:a\s+)?(?:pro|con|affirmative|negative)\s+ballot\b"
    r")"
)


def _argument_error(
    message: str,
    *,
    request_id: str | None,
    field: str,
    **details: Any,
) -> CaseFileError:
    return CaseFileError(
        ErrorCode.ARGUMENT_VALIDATION_FAILED,
        message,
        stage="argument_strategist.validate",
        agent=STRATEGIST,
        request_id=request_id,
        safe_details={"field": field, **details},
    )


def _sections(draft: ArgumentDraft) -> dict[str, ArgumentSection]:
    return {name: getattr(draft, name) for name in ARGUMENT_SECTION_NAMES}


def validate_argument_draft(
    draft: ArgumentDraft,
    *,
    request: ArgumentRequest,
    evidence_packet: EvidencePacket,
    request_id: str | None = None,
    previous_draft: ArgumentDraft | None = None,
) -> None:
    """Reject any draft that crosses the supplied evidence or session boundary."""

    if evidence_packet.resolution != request.resolution:
        raise _argument_error(
            "The evidence packet resolution does not match the argument request.",
            request_id=request_id,
            field="evidence_packet.resolution",
        )
    if evidence_packet.side != request.side:
        raise _argument_error(
            "The evidence packet side does not match the argument request.",
            request_id=request_id,
            field="evidence_packet.side",
        )
    if draft.resolution != request.resolution:
        raise _argument_error(
            "The Argument Strategist changed the requested resolution.",
            request_id=request_id,
            field="resolution",
        )
    if draft.side != request.side:
        raise _argument_error(
            "The Argument Strategist changed the requested side.",
            request_id=request_id,
            field="side",
        )

    available_ids = {card.card_id for card in evidence_packet.cards}
    cited_ids = {
        card_id for section in _sections(draft).values() for card_id in section.card_ids
    }
    unknown_ids = cited_ids - available_ids
    if unknown_ids:
        raise _argument_error(
            "The Argument Strategist cited a card outside the supplied EvidencePacket.",
            request_id=request_id,
            field="card_ids",
            unknown_card_count=len(unknown_ids),
        )

    incomplete_sections = [
        name
        for name, section in _sections(draft).items()
        if section.support in {"partially_supported", "unsupported"}
    ]
    if incomplete_sections and not draft.unsupported_facts:
        raise _argument_error(
            "A partial argument must report its unsupported factual gaps.",
            request_id=request_id,
            field="unsupported_facts",
            incomplete_sections=incomplete_sections,
        )
    if evidence_packet.empty_result:
        non_unsupported = [
            name
            for name, section in _sections(draft).items()
            if section.support != "unsupported"
        ]
        if non_unsupported:
            raise _argument_error(
                "A draft cannot mark factual sections supported when no evidence was supplied.",
                request_id=request_id,
                field="support",
                invalid_sections=non_unsupported,
            )

    scripted_sections = [
        name
        for name, section in _sections(draft).items()
        if _SPEECH_OR_SCRIPT.search(section.text)
    ]
    if scripted_sections:
        raise _argument_error(
            "The Argument Strategist returned speech or delivery-script language.",
            request_id=request_id,
            field="format",
            scripted_sections=scripted_sections,
        )

    if request.revision_instruction is None:
        if previous_draft is not None:
            raise _argument_error(
                "A previous draft was supplied without a revision instruction.",
                request_id=request_id,
                field="revision_instruction",
            )
        return
    if previous_draft is None:
        raise _argument_error(
            "A session-bound revision requires the previous argument draft.",
            request_id=request_id,
            field="previous_draft",
        )
    if (
        previous_draft.resolution != request.resolution
        or previous_draft.side != request.side
    ):
        raise _argument_error(
            "The previous draft does not match the active argument request.",
            request_id=request_id,
            field="previous_draft",
        )
    if request.preserve_citations and set(draft.source_card_ids) != set(
        previous_draft.source_card_ids
    ):
        raise _argument_error(
            "The revision did not preserve the requested citations.",
            request_id=request_id,
            field="source_card_ids",
        )
    if request.requested_sections:
        requested = set(request.requested_sections)
        changed_outside_scope = [
            name
            for name in ARGUMENT_SECTION_NAMES
            if name not in requested
            and getattr(draft, name) != getattr(previous_draft, name)
        ]
        if changed_outside_scope:
            raise _argument_error(
                "The revision changed sections outside its requested scope.",
                request_id=request_id,
                field="requested_sections",
                changed_sections=changed_outside_scope,
            )


class ArgumentStrategist:
    """Parse constraints and generate one validated artifact per model call."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def parse_request(
        self,
        context: ToolContext,
        *,
        original_request: str,
        evidence_packet: EvidencePacket,
        requested_side: Side | None = None,
        previous_request: ArgumentRequest | None = None,
        revision_instruction: str | None = None,
    ) -> ArgumentRequest:
        """Turn open-ended language into a strict, evidence-bounded request."""

        required_side = requested_side or (
            previous_request.side
            if previous_request is not None
            else evidence_packet.side
        )
        if required_side is None:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "An argument request requires a Pro or Con side.",
                stage="argument_strategist.parse_request",
                agent=STRATEGIST,
                request_id=context.request_id,
                safe_details={"field": "side"},
            )
        if evidence_packet.resolution != context.resolution:
            raise _argument_error(
                "The EvidencePacket does not match the active resolution.",
                request_id=context.request_id,
                field="evidence_packet.resolution",
            )
        if evidence_packet.side != required_side:
            raise _argument_error(
                "The EvidencePacket does not match the requested side.",
                request_id=context.request_id,
                field="evidence_packet.side",
            )
        if previous_request is None and revision_instruction is not None:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "A revision instruction requires an active argument request.",
                stage="argument_strategist.parse_request",
                agent=STRATEGIST,
                request_id=context.request_id,
                safe_details={"field": "previous_request"},
            )
        if previous_request is not None:
            if revision_instruction is None:
                raise CaseFileError(
                    ErrorCode.REQUEST_INVALID,
                    "An active argument revision requires a revision instruction.",
                    stage="argument_strategist.parse_request",
                    agent=STRATEGIST,
                    request_id=context.request_id,
                    safe_details={"field": "revision_instruction"},
                )
            if (
                previous_request.resolution != context.resolution
                or previous_request.side != required_side
            ):
                raise _argument_error(
                    "The prior argument request does not match the active session.",
                    request_id=context.request_id,
                    field="previous_request",
                )

        prompt = load_prompt("argument_strategist")
        payload = {
            "operation": "parse_constraints",
            "original_request": original_request,
            "active_resolution": context.resolution,
            "required_side": required_side,
            "available_source_files": evidence_packet.confirmed_source_files_considered,
            "available_cards": [
                {
                    "card_id": card.card_id,
                    "source_filename": card.source_filename,
                    "header": card.header,
                    "tag": card.tag,
                }
                for card in evidence_packet.cards
            ],
            "previous_request": (
                previous_request.model_dump(mode="json")
                if previous_request is not None
                else None
            ),
            "revision_instruction": revision_instruction,
        }
        try:
            raw = self.model.complete_json(
                system=prompt.content,
                user=json.dumps(payload, ensure_ascii=False),
                max_tokens=4_000,
                schema=ArgumentRequest,
                agent=STRATEGIST,
                prompt_template=prompt.template_name,
                prompt_version=prompt.version,
            )
        except CaseFileError as error:
            if context.request_id is not None:
                error.with_request_id(context.request_id)
            raise
        try:
            request = ArgumentRequest.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                "The Argument Strategist returned invalid argument constraints.",
                stage="argument_strategist.parse_request",
                agent=STRATEGIST,
                request_id=context.request_id,
                safe_details={"schema": ArgumentRequest.__name__},
                cause=exc,
            ) from exc
        self._validate_parsed_request(
            request,
            context=context,
            original_request=original_request,
            evidence_packet=evidence_packet,
            required_side=required_side,
            previous_request=previous_request,
            revision_instruction=revision_instruction,
        )
        return request

    def generate_argument(
        self,
        context: ToolContext,
        *,
        request: ArgumentRequest,
        evidence_packet: EvidencePacket,
        previous_draft: ArgumentDraft | None = None,
    ) -> ArgumentDraft:
        """Generate once, then apply deterministic grounding validation."""

        self._validate_request_packet(
            request,
            evidence_packet=evidence_packet,
            context=context,
        )
        prompt = load_prompt("argument_strategist")
        try:
            raw = self.model.complete_json(
                system=prompt.content,
                user=json.dumps(
                    {
                        "operation": "generate_argument",
                        "request": request.model_dump(mode="json"),
                        "evidence_packet": evidence_packet.model_dump(mode="json"),
                        "previous_draft": (
                            previous_draft.model_dump(mode="json")
                            if previous_draft is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                ),
                max_tokens=8_000,
                schema=ArgumentDraft,
                agent=STRATEGIST,
                prompt_template=prompt.template_name,
                prompt_version=prompt.version,
            )
        except CaseFileError as error:
            if context.request_id is not None:
                error.with_request_id(context.request_id)
            raise
        try:
            draft = ArgumentDraft.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                "The Argument Strategist returned an invalid structured argument.",
                stage="argument_strategist.generate",
                agent=STRATEGIST,
                request_id=context.request_id,
                safe_details={"schema": ArgumentDraft.__name__},
                cause=exc,
            ) from exc
        validate_argument_draft(
            draft,
            request=request,
            evidence_packet=evidence_packet,
            request_id=context.request_id,
            previous_draft=previous_draft,
        )
        return draft

    def create_argument(
        self,
        context: ToolContext,
        *,
        original_request: str,
        evidence_packet: EvidencePacket,
        requested_side: Side | None = None,
    ) -> tuple[ArgumentRequest, ArgumentDraft]:
        request = self.parse_request(
            context,
            original_request=original_request,
            evidence_packet=evidence_packet,
            requested_side=requested_side,
        )
        return request, self.generate_argument(
            context,
            request=request,
            evidence_packet=evidence_packet,
        )

    def revise_argument(
        self,
        context: ToolContext,
        *,
        instruction: str,
        evidence_packet: EvidencePacket,
        previous_request: ArgumentRequest,
        previous_draft: ArgumentDraft,
    ) -> tuple[ArgumentRequest, ArgumentDraft]:
        """Revise against explicit prior session artifacts, never hidden state."""

        validate_argument_draft(
            previous_draft,
            request=previous_request,
            evidence_packet=evidence_packet,
            request_id=context.request_id,
        )
        request = self.parse_request(
            context,
            original_request=previous_request.original_request,
            evidence_packet=evidence_packet,
            requested_side=previous_request.side,
            previous_request=previous_request,
            revision_instruction=instruction,
        )
        return request, self.generate_argument(
            context,
            request=request,
            evidence_packet=evidence_packet,
            previous_draft=previous_draft,
        )

    @staticmethod
    def _validate_request_packet(
        request: ArgumentRequest,
        *,
        evidence_packet: EvidencePacket,
        context: ToolContext,
    ) -> None:
        if request.resolution != context.resolution:
            raise _argument_error(
                "The argument request does not match the active resolution.",
                request_id=context.request_id,
                field="resolution",
            )
        if evidence_packet.resolution != request.resolution:
            raise _argument_error(
                "The EvidencePacket does not match the argument resolution.",
                request_id=context.request_id,
                field="evidence_packet.resolution",
            )
        if evidence_packet.side != request.side:
            raise _argument_error(
                "The EvidencePacket does not match the argument side.",
                request_id=context.request_id,
                field="evidence_packet.side",
            )
        unknown_files = set(request.source_files) - set(
            evidence_packet.confirmed_source_files_considered
        )
        if unknown_files:
            raise _argument_error(
                "The argument request names a source outside the EvidencePacket.",
                request_id=context.request_id,
                field="source_files",
                unknown_file_count=len(unknown_files),
            )

    def _validate_parsed_request(
        self,
        request: ArgumentRequest,
        *,
        context: ToolContext,
        original_request: str,
        evidence_packet: EvidencePacket,
        required_side: Side,
        previous_request: ArgumentRequest | None,
        revision_instruction: str | None,
    ) -> None:
        invalid_field: str | None = None
        if request.original_request != original_request:
            invalid_field = "original_request"
        elif request.resolution != context.resolution:
            invalid_field = "resolution"
        elif request.side != required_side:
            invalid_field = "side"
        elif request.revision_instruction != revision_instruction:
            invalid_field = "revision_instruction"
        elif set(request.source_files) - set(
            evidence_packet.confirmed_source_files_considered
        ):
            invalid_field = "source_files"
        elif previous_request is not None:
            if request.original_request != previous_request.original_request:
                invalid_field = "original_request"
            elif request.subject != previous_request.subject:
                invalid_field = "subject"
            elif set(request.entities) != set(previous_request.entities):
                invalid_field = "entities"
            elif set(request.source_files) != set(previous_request.source_files):
                invalid_field = "source_files"
            elif not set(previous_request.constraints).issubset(request.constraints):
                invalid_field = "constraints"
            elif previous_request.preserve_citations and not request.preserve_citations:
                invalid_field = "preserve_citations"
        if invalid_field is not None:
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "The Argument Strategist changed a protected request constraint.",
                stage="argument_strategist.parse_request",
                agent=STRATEGIST,
                request_id=context.request_id,
                safe_details={"field": invalid_field},
            )
        self._validate_request_packet(
            request,
            evidence_packet=evidence_packet,
            context=context,
        )
