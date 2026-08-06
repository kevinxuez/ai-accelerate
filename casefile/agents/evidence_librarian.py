"""Model-planned, confirmed-only evidence and reference retrieval."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import ValidationError

from casefile.retrieval import COLLECTION_SCHEMA_VERSION
from casefile.tools import CaseFileTools, ToolContext

from .contracts import (
    ClarificationRequest,
    EvidenceCard,
    EvidencePacket,
    EvidenceProvenance,
    EvidenceQueryPlan,
    IngestionMetadataPlan,
    IngestionCommitResult,
    IngestionPreview,
    RuleChunk,
    RulePacket,
    Side,
    TextSpan,
    TopicPacket,
)
from .errors import CaseFileError, ErrorCode
from .prompt_registry import load_prompt


LIBRARIAN = "evidence_librarian"


class EvidenceLibrarian:
    """Plan bounded searches and return only typed, grounded artifacts."""

    def __init__(self, tools: CaseFileTools, model: Any) -> None:
        self.tools = tools
        self.model = model

    def plan_evidence(
        self,
        context: ToolContext,
        *,
        request: str,
        requested_side: Side | None,
        confirmed_files: list[str],
    ) -> EvidenceQueryPlan:
        prompt = load_prompt("evidence_librarian")
        raw = self.model.complete_json(
            system=prompt.content,
            user=json.dumps(
                {
                    "operation": "plan_evidence",
                    "request": request,
                    "active_resolution": context.resolution,
                    "requested_side": requested_side,
                    "confirmed_available_files": confirmed_files,
                },
                ensure_ascii=False,
            ),
            max_tokens=1_200,
            schema=EvidenceQueryPlan,
            agent=LIBRARIAN,
            prompt_template=prompt.template_name,
            prompt_version=prompt.version,
        )
        try:
            plan = EvidenceQueryPlan.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                "The Evidence Librarian returned an invalid evidence query plan.",
                stage="evidence_librarian.plan",
                agent=LIBRARIAN,
                request_id=context.request_id,
                safe_details={"schema": EvidenceQueryPlan.__name__},
                cause=exc,
            ) from exc

        if plan.resolution != context.resolution:
            self._invalid_plan(
                context,
                "The Evidence Librarian changed the active resolution.",
                field="resolution",
            )
        if (
            requested_side is not None
            and plan.side != requested_side
            and not (plan.clarification_needed and plan.side is None)
        ):
            self._invalid_plan(
                context,
                "The Evidence Librarian changed the requested side.",
                field="side",
            )
        unknown_files = sorted(set(plan.source_files) - set(confirmed_files))
        if unknown_files:
            self._invalid_plan(
                context,
                "The Evidence Librarian selected a file that is not confirmed.",
                field="source_files",
                unknown_file_count=len(unknown_files),
            )
        if not plan.clarification_needed and plan.side is None:
            self._invalid_plan(
                context,
                "The Evidence Librarian omitted the side for an executable search.",
                field="side",
            )
        return plan

    def plan_ingestion_metadata(
        self,
        context: ToolContext,
        *,
        request: str,
    ) -> IngestionMetadataPlan:
        """Validate the active resolution and model-extract only an explicit side."""

        prompt = load_prompt("evidence_librarian")
        try:
            raw = self.model.complete_json(
                system=prompt.content,
                user=json.dumps(
                    {
                        "operation": "plan_ingestion_metadata",
                        "request": request,
                        "active_resolution": context.resolution,
                    },
                    ensure_ascii=False,
                ),
                max_tokens=600,
                schema=IngestionMetadataPlan,
                agent=LIBRARIAN,
                prompt_template=prompt.template_name,
                prompt_version=prompt.version,
            )
        except CaseFileError as error:
            if context.request_id is not None:
                error.with_request_id(context.request_id)
            raise
        try:
            plan = IngestionMetadataPlan.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                "The Evidence Librarian returned invalid ingestion metadata.",
                stage="evidence_librarian.ingestion_metadata",
                agent=LIBRARIAN,
                request_id=context.request_id,
                safe_details={"schema": IngestionMetadataPlan.__name__},
                cause=exc,
            ) from exc
        if plan.resolution != context.resolution:
            self._invalid_plan(
                context,
                "The Evidence Librarian changed the active ingestion resolution.",
                field="resolution",
            )
        return plan

    def retrieve_evidence(
        self,
        context: ToolContext,
        *,
        request: str,
        requested_side: Side | None,
    ) -> EvidencePacket | ClarificationRequest:
        context = self._context(context)
        confirmed_files = self.tools.registry.invoke(
            LIBRARIAN,
            "list_confirmed_files",
            context,
            {"resolution": context.resolution, "side": requested_side},
        )
        plan = self.plan_evidence(
            context,
            request=request,
            requested_side=requested_side,
            confirmed_files=confirmed_files,
        )
        if plan.clarification_needed:
            return ClarificationRequest(
                question=plan.clarification_question or "What evidence should I find?",
                missing_fields=["research_constraints"],
                reason_code="librarian_research_needs_input",
            )

        considered = list(dict.fromkeys(plan.source_files or confirmed_files))
        if len(considered) > 100:
            return ClarificationRequest(
                question="Which confirmed source files should I search?",
                missing_fields=["source_files"],
                reason_code="too_many_confirmed_files",
            )

        queries = list(dict.fromkeys(plan.queries))
        cards_by_id: dict[str, EvidenceCard] = {}
        for query in queries:
            raw_cards = self.tools.registry.invoke(
                LIBRARIAN,
                "search_cards",
                context,
                {
                    "query": query,
                    "side": plan.side,
                    "resolution": plan.resolution,
                    "n": min(10, plan.result_limit),
                    "source_files": list(dict.fromkeys(plan.source_files)),
                },
            )
            for raw_card in raw_cards:
                card = self._evidence_card(
                    raw_card,
                    confirmed_files=set(confirmed_files),
                    considered_files=set(considered),
                    context=context,
                )
                current = cards_by_id.get(card.card_id)
                if current is None or card.retrieval_score > current.retrieval_score:
                    cards_by_id[card.card_id] = card

        cards = sorted(
            cards_by_id.values(),
            key=lambda card: card.retrieval_score,
            reverse=True,
        )[: plan.result_limit]
        return EvidencePacket(
            request_summary=self._summary(request),
            resolution=plan.resolution,
            side=plan.side,
            confirmed_source_files_considered=considered,
            queries_executed=queries,
            cards=cards,
            empty_result=not cards,
            provenance=EvidenceProvenance(
                ledger_schema_version=COLLECTION_SCHEMA_VERSION,
                retrieval_backend="chroma",
                embedding_model=self.tools.index.embedding_model,
                confirmed_only=True,
            ),
        )

    def stage_ingestion(
        self,
        context: ToolContext,
        *,
        file_path: str,
        resolution: str | None,
        side: Side | None,
    ) -> IngestionPreview | ClarificationRequest:
        """Build a staged preview only after deterministic metadata is complete."""

        missing: list[str] = []
        if not (resolution or context.resolution).strip():
            missing.append("resolution")
        if side is None:
            missing.append("side")
        if missing:
            labels = [
                "the active resolution" if item == "resolution" else "Pro or Con"
                for item in missing
            ]
            return ClarificationRequest(
                question="Please provide " + " and ".join(labels) + ".",
                missing_fields=missing,
                reason_code="ingestion_metadata_required",
            )
        raw = self.tools.registry.invoke(
            LIBRARIAN,
            "stage_ingestion_preview",
            self._context(context),
            {
                "file_path": file_path,
                "resolution": resolution or context.resolution,
                "side": side,
            },
        )
        try:
            return IngestionPreview.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "The ingestion preview did not match its handoff contract.",
                stage="evidence_librarian.ingestion_preview",
                agent=LIBRARIAN,
                request_id=context.request_id,
                safe_details={"schema": IngestionPreview.__name__},
                cause=exc,
            ) from exc

    def commit_ingestion(
        self,
        context: ToolContext,
        *,
        confirmation_token: str,
        idempotency_key: str | None = None,
    ) -> IngestionCommitResult:
        raw = self.tools.registry.invoke(
            LIBRARIAN,
            "commit_ingestion",
            self._context(context),
            {
                "confirmation_token": confirmation_token,
                "idempotency_key": idempotency_key,
            },
        )
        try:
            return IngestionCommitResult.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "The ingestion commit result did not match its handoff contract.",
                stage="evidence_librarian.ingestion_commit",
                agent=LIBRARIAN,
                request_id=context.request_id,
                safe_details={"schema": IngestionCommitResult.__name__},
                cause=exc,
            ) from exc

    def retrieve_rules(
        self,
        context: ToolContext,
        *,
        question: str,
        event: str = "Public Forum",
        limit: int = 3,
    ) -> RulePacket:
        context = self._context(context)
        raw_chunks = self.tools.registry.invoke(
            LIBRARIAN,
            "search_rules",
            context,
            {"question": question, "n": limit},
        )
        try:
            chunks = [
                RuleChunk(
                    chunk_id=str(raw.get("_chunk_id") or raw.get("id") or ""),
                    section_number=str(raw.get("section_number") or "unsectioned"),
                    section_title=str(raw.get("section_title") or "Untitled section"),
                    text=str(raw.get("text") or ""),
                    document=str(raw.get("document") or "unknown document"),
                    score=float(raw.get("score", 0.0)),
                )
                for raw in raw_chunks
            ]
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_INDEX_MISMATCH,
                "Rules retrieval returned a malformed authoritative chunk.",
                stage="evidence_librarian.rules",
                agent=LIBRARIAN,
                tool="search_rules",
                request_id=context.request_id,
                cause=exc,
            ) from exc
        return RulePacket(
            request_summary=self._summary(question),
            event=event,
            chunks=chunks,
            empty_result=not chunks,
        )

    def retrieve_topic(
        self,
        context: ToolContext,
        *,
        event: str = "Public Forum",
        as_of: str | None = None,
    ) -> TopicPacket:
        context = self._context(context)
        raw = self.tools.registry.invoke(
            LIBRARIAN,
            "get_current_topic",
            context,
            {"event": event, "as_of": as_of},
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("topic"), dict):
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                str(raw) if isinstance(raw, str) else "No topic is available.",
                stage="evidence_librarian.topic",
                agent=LIBRARIAN,
                request_id=context.request_id,
            )
        topic = raw["topic"]
        backend = "http" if raw.get("backend") == "http" else "fixture"
        synthetic = bool(raw.get("synthetic") or topic.get("synthetic"))
        try:
            return TopicPacket(
                event=str(topic.get("event") or event),
                resolution=str(topic.get("resolution") or ""),
                effective_from=self._date(topic.get("effective_from")),
                effective_to=self._date(topic.get("effective_to")),
                provider=str(raw.get("provider") or "NSDA-compatible provider"),
                backend=backend,
                synthetic=synthetic,
                source_ref=str(
                    topic.get("source_ref")
                    or topic.get("id")
                    or raw.get("dataset_version")
                    or "configured-provider"
                ),
            )
        except CaseFileError:
            raise
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.NSDA_UPSTREAM_ERROR,
                "The configured topic provider returned a malformed topic.",
                stage="evidence_librarian.topic",
                agent=LIBRARIAN,
                request_id=context.request_id,
                cause=exc,
            ) from exc

    @staticmethod
    def _context(context: ToolContext) -> ToolContext:
        if context.agent == LIBRARIAN:
            return context
        return ToolContext(
            role=context.role,
            user_id=context.user_id,
            resolution=context.resolution,
            request_id=context.request_id,
            agent=LIBRARIAN,
        )

    @staticmethod
    def _summary(value: str) -> str:
        return value.strip()[:500] or "Evidence request"

    @staticmethod
    def _date(value: Any) -> date | None:
        if value is None or isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise CaseFileError(
                ErrorCode.NSDA_UPSTREAM_ERROR,
                "The configured topic provider returned an invalid effective date.",
                stage="evidence_librarian.topic",
                agent=LIBRARIAN,
                cause=exc,
            ) from exc

    @staticmethod
    def _spans(raw_spans: Any) -> list[TextSpan]:
        spans: list[TextSpan] = []
        for raw in raw_spans or []:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                spans.append(TextSpan(start=int(raw[0]), end=int(raw[1])))
            elif isinstance(raw, dict):
                spans.append(TextSpan.model_validate(raw))
            else:
                raise ValueError("evidence span must contain start and end offsets")
        return spans

    def _evidence_card(
        self,
        raw: dict[str, Any],
        *,
        confirmed_files: set[str],
        considered_files: set[str],
        context: ToolContext,
    ) -> EvidenceCard:
        source = str(raw.get("source_file") or "")
        if source not in confirmed_files or source not in considered_files:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_INDEX_MISMATCH,
                "Evidence retrieval returned a card outside the confirmed source boundary.",
                stage="evidence_librarian.packet",
                agent=LIBRARIAN,
                tool="search_cards",
                request_id=context.request_id,
            )
        try:
            return EvidenceCard(
                card_id=str(raw.get("id") or raw.get("_chunk_id") or ""),
                source_filename=source,
                citation=str(raw.get("cite_full") or ""),
                header=str(raw.get("header") or ""),
                tag=str(raw.get("tag") or ""),
                body=str(raw.get("body") or ""),
                read_spans=self._spans(raw.get("read_spans")),
                emphasis_spans=self._spans(raw.get("emphasis_spans")),
                resolution=str(raw.get("resolution") or ""),
                side=str(raw.get("side") or ""),
                retrieval_score=float(raw.get("score", 0.0)),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_INDEX_MISMATCH,
                "Evidence retrieval returned a malformed committed card.",
                stage="evidence_librarian.packet",
                agent=LIBRARIAN,
                tool="search_cards",
                request_id=context.request_id,
                cause=exc,
            ) from exc

    @staticmethod
    def _invalid_plan(
        context: ToolContext,
        message: str,
        *,
        field: str,
        **details: Any,
    ) -> None:
        raise CaseFileError(
            ErrorCode.AGENT_OUTPUT_INVALID,
            message,
            stage="evidence_librarian.plan",
            agent=LIBRARIAN,
            request_id=context.request_id,
            safe_details={"field": field, **details},
        )
