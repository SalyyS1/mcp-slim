"""MS017: Zero-dependency BM25+ intent-aware MCP tool catalog pruner.
MS018: JSON-RPC 2.0 batch, duplicate-ID, and unsolicited response-spoofing firewall.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import fnmatch
import json
import math
import re
import time
from typing import Any

_CATALOG_PRUNER_RULE_ID = "MS017"
_RPC_FIREWALL_RULE_ID = "MS018"

_CAMEL_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_ALPHA_DIGIT_BOUNDARY = re.compile(r"([a-zA-Z])([0-9])|([0-9])([a-zA-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_PRUNER_STOP_WORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "must",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "tool",
    "tools",
    "use",
    "used",
    "using",
    "via",
    "want",
    "was",
    "we",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
    "you",
    "your",
})

_STEM_PROTECTED = frozenset({
    "alias",
    "analysis",
    "atlas",
    "aws",
    "axis",
    "basis",
    "bias",
    "bus",
    "canvas",
    "chaos",
    "class",
    "consensus",
    "corpus",
    "cosmos",
    "cross",
    "data",
    "dns",
    "dos",
    "gcs",
    "gas",
    "glass",
    "grass",
    "https",
    "ingress",
    "is",
    "k8s",
    "kubernetes",
    "less",
    "loss",
    "mass",
    "metrics",
    "news",
    "nexus",
    "ops",
    "os",
    "pass",
    "plus",
    "postgres",
    "process",
    "prometheus",
    "radius",
    "rds",
    "redis",
    "rss",
    "saas",
    "series",
    "serverless",
    "sms",
    "status",
    "stress",
    "success",
    "this",
    "tls",
    "us",
    "virus",
    "wasm",
    "yes",
})

_SYNONYM_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("repo", "repository", "repositories", "git"),
    ("pr", "pullrequest", "merge"),
    ("db", "database", "sql", "query", "table"),
    ("msg", "message", "chat", "notify", "notification"),
    ("doc", "docs", "document", "documentation", "markdown"),
    ("cfg", "config", "configuration", "setting", "settings"),
    ("auth", "authenticate", "authentication", "credential", "token", "login"),
    ("env", "environment"),
    ("dir", "directory", "folder"),
    ("fs", "file", "files", "filesystem", "path"),
    ("cmd", "command", "exec", "execute", "run", "shell"),
    ("k8s", "kubernetes", "pod", "cluster", "deployment"),
    ("vm", "microvm", "instance", "sandbox", "container"),
    ("err", "error", "exception", "crash", "failure"),
    ("sec", "security", "vulnerability", "vuln", "audit", "secret"),
    ("del", "delete", "remove", "drop", "purge"),
    ("upd", "update", "edit", "modify", "patch"),
    ("add", "create", "insert", "new"),
    ("get", "read", "fetch", "retrieve", "inspect", "view"),
    ("list", "search", "find", "lookup", "filter"),
)

_SYNONYM_MAP: dict[str, tuple[str, ...]] = {}
for _cluster in _SYNONYM_CLUSTERS:
    for _term in _cluster:
        _existing = set(_SYNONYM_MAP.get(_term, ()))
        _existing.update(_cluster)
        _SYNONYM_MAP[_term] = tuple(sorted(_existing))


def _stem_word(word: str) -> str:
    """Lightweight deterministic suffix stemmer safe for technical identifiers."""
    if len(word) <= 3 or word in _STEM_PROTECTED:
        return word
    if word.endswith("ational") and len(word) >= 9:
        return word[:-5] + "e"
    if word.endswith("tional") and len(word) >= 8:
        return word[:-2]
    if word.endswith("izing") and len(word) >= 7:
        return word[:-3] + "e"
    if word.endswith("ized") and len(word) >= 6:
        return word[:-1]
    if word.endswith("izer") and len(word) >= 6:
        return word[:-1]
    if word.endswith("ying") and len(word) >= 6:
        return word[:-4] + "y"
    if word.endswith("ies") and len(word) >= 5:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) >= 6:
        base = word[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in ("l", "s", "z"):
            base = base[:-1]
        return base
    if word.endswith("ed") and len(word) >= 5 and not word.endswith("eed"):
        base = word[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in ("l", "s", "z"):
            base = base[:-1]
        return base
    if word.endswith("es") and len(word) >= 5:
        if word.endswith(("ches", "shes", "xes", "zes", "sses")):
            return word[:-2]
    if word.endswith("s") and len(word) >= 4 and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize_mcp_text(text: str, *, expand_synonyms: bool = False) -> list[str]:
    """Splits camelCase, snake_case, kebab-case, and prose into normalized stem tokens."""
    if not text:
        return []
    separated = _CAMEL_BOUNDARY_1.sub(r"\1 \2", text)
    separated = _CAMEL_BOUNDARY_2.sub(r"\1 \2", separated)
    # Preserve compact alphanumeric numeronyms (e.g. k8s, i18n, s3, ec2, sha256, oauth2)
    raw_alnum_chunks = [w for w in _NON_ALNUM_RE.split(separated.lower()) if len(w) >= 2]
    raw_words: list[str] = []
    for chunk in raw_alnum_chunks:
        has_alpha = any(c.isalpha() for c in chunk)
        has_digit = any(c.isdigit() for c in chunk)
        if has_alpha and has_digit:
            raw_words.append(chunk)
            split_parts = _ALPHA_DIGIT_BOUNDARY.sub(
                lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}",
                chunk,
            ).split()
            for sp in split_parts:
                if len(sp) >= 2 and sp.isalpha():
                    raw_words.append(sp)
        else:
            raw_words.append(chunk)

    tokens: list[str] = []
    for raw in raw_words:
        if raw in _PRUNER_STOP_WORDS:
            continue
        stemmed = _stem_word(raw)
        if len(stemmed) >= 2 and stemmed not in _PRUNER_STOP_WORDS:
            tokens.append(stemmed)
        if expand_synonyms:
            matched_syn_keys = set()
            if raw in _SYNONYM_MAP:
                matched_syn_keys.add(raw)
            if stemmed in _SYNONYM_MAP:
                matched_syn_keys.add(stemmed)
            if not matched_syn_keys and len(raw) >= 5:
                for sk in _SYNONYM_MAP:
                    if len(sk) >= 5 and _is_one_edit_away(raw, sk):
                        matched_syn_keys.add(sk)
            for sk in matched_syn_keys:
                for syn in _SYNONYM_MAP.get(sk, ()):
                    syn_stem = _stem_word(syn)
                    if syn_stem != stemmed and syn_stem not in _PRUNER_STOP_WORDS:
                        tokens.append(syn_stem)
    return tokens


def _is_one_edit_away(a: str, b: str) -> bool:
    """Fast O(L) Damerau/Levenshtein distance <= 1 check for typo tolerance on >= 5 char tokens."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or min(la, lb) < 5:
        return False
    if a == b:
        return True
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i = diffs[0]
            return a[i] == b[i + 1] and a[i + 1] == b[i]
        return False
    # Insertion / deletion
    short, long_s = (a, b) if la < lb else (b, a)
    i = j = edits = 0
    while i < len(short) and j < len(long_s):
        if short[i] == long_s[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def _extract_schema_zone_tokens(schema: Any, depth: int = 0) -> tuple[list[str], list[str]]:
    """Extracts (property_and_enum_tokens, description_tokens) from a JSON Schema tree."""
    if depth > 6 or not isinstance(schema, dict):
        return ([], [])
    prop_tokens: list[str] = []
    desc_tokens: list[str] = []

    raw_enums = schema.get("enum")
    if isinstance(raw_enums, list):
        for ev in raw_enums[:25]:
            if isinstance(ev, str):
                prop_tokens.extend(tokenize_mcp_text(ev))

    props = schema.get("properties")
    if isinstance(props, dict):
        for p_name, p_node in props.items():
            if isinstance(p_name, str):
                prop_tokens.extend(tokenize_mcp_text(p_name))
            if isinstance(p_node, dict):
                p_desc = p_node.get("description")
                if isinstance(p_desc, str) and p_desc:
                    desc_tokens.extend(tokenize_mcp_text(p_desc))
                sub_p, sub_d = _extract_schema_zone_tokens(p_node, depth + 1)
                prop_tokens.extend(sub_p)
                desc_tokens.extend(sub_d)

    items = schema.get("items")
    if isinstance(items, dict):
        sub_p, sub_d = _extract_schema_zone_tokens(items, depth + 1)
        prop_tokens.extend(sub_p)
        desc_tokens.extend(sub_d)

    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list):
            for br in branches[:8]:
                sub_p, sub_d = _extract_schema_zone_tokens(br, depth + 1)
                prop_tokens.extend(sub_p)
                desc_tokens.extend(sub_d)

    for defs_key in ("$defs", "definitions"):
        defs_map = schema.get(defs_key)
        if isinstance(defs_map, dict):
            for d_name, d_node in defs_map.items():
                if isinstance(d_name, str):
                    prop_tokens.extend(tokenize_mcp_text(d_name))
                sub_p, sub_d = _extract_schema_zone_tokens(d_node, depth + 1)
                prop_tokens.extend(sub_p)
                desc_tokens.extend(sub_d)

    return (prop_tokens, desc_tokens)


def _matches_keep_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if name == pat:
            return True
        if ("*" in pat or "?" in pat or "[" in pat) and fnmatch.fnmatchcase(name, pat):
            return True
    return False


@dataclass
class _IndexedToolDoc:
    name: str
    raw_tool: dict[str, Any]
    weighted_tf: dict[str, float]
    weighted_len: float
    vocab: set[str]


@dataclass
class ToolPruneResult:
    """Outcome of BM25+ intent-aware MCP tool catalog pruning (MS017)."""

    intent: str
    tools: list[dict[str, Any]]
    ranked_scores: list[dict[str, Any]]
    pruned_names: list[str]
    tools_before: int
    tools_after: int
    tokens_before: int
    tokens_after: int
    fallback_used: bool = False

    @property
    def tools_pruned(self) -> int:
        return max(0, self.tools_before - self.tools_after)

    @property
    def reduction_pct(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        saved = max(0, self.tokens_before - self.tokens_after)
        return round((saved / self.tokens_before) * 100.0, 1)

    def to_finding(self) -> dict[str, Any] | None:
        if self.tools_pruned <= 0 and not self.fallback_used:
            return None
        if self.fallback_used:
            return {
                "rule_id": _CATALOG_PRUNER_RULE_ID,
                "level": "note",
                "tool": "tools/list",
                "kind": "intent_prune_fallback",
                "tools_before": self.tools_before,
                "tools_after": self.tools_after,
                "message": (
                    f"Intent query {self.intent!r} had zero lexical overlap across "
                    f"{self.tools_before} tools; retained fallback slice of {self.tools_after} tools."
                ),
            }
        return {
            "rule_id": _CATALOG_PRUNER_RULE_ID,
            "level": "note",
            "tool": "tools/list",
            "kind": "intent_catalog_pruned",
            "tools_before": self.tools_before,
            "tools_after": self.tools_after,
            "message": (
                f"Pruned MCP catalog from {self.tools_before} to {self.tools_after} tools "
                f"for intent {self.intent!r} ({self.tokens_before} -> {self.tokens_after} tokens, "
                f"-{self.reduction_pct}%)."
            ),
        }

    def to_audit_report(self) -> dict[str, Any]:
        finding = self.to_finding()
        findings = [finding] if finding is not None else []
        return {
            "intent": self.intent,
            "tools_input": self.tools_before,
            "tools_output": self.tools_after,
            "tools_pruned": self.tools_pruned,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": self.reduction_pct,
            "fallback_used": self.fallback_used,
            "security_findings": 0,
            "findings": findings,
            "ranked_scores": list(self.ranked_scores),
            "pruned_names": list(self.pruned_names),
            "tools": list(self.tools),
        }

    def to_sarif_results(self) -> list[dict[str, Any]]:
        finding = self.to_finding()
        if finding is None:
            return []
        return [
            {
                "ruleId": _CATALOG_PRUNER_RULE_ID,
                "level": finding["level"],
                "message": {"text": finding["message"]},
                "locations": [
                    {
                        "logicalLocations": [
                            {"name": "tools/list", "kind": "function"}
                        ]
                    }
                ],
                "properties": {
                    "kind": finding["kind"],
                    "tools_before": self.tools_before,
                    "tools_after": self.tools_after,
                    "reduction_pct": self.reduction_pct,
                },
            }
        ]


class ToolCatalogPruner:
    """MS017: Zero-dependency multi-field BM25+ intent-aware MCP tool catalog pruner.

    Ranks upstream ``tools/list`` definitions against a natural-language task intent
    using subword/identifier splitting, technical synonym expansion, and field-weighted
    BM25+ (Lv & Zhai, CIKM 2011; ``k1=1.5, b=0.75, delta=0.5``) so verbose multi-field
    JSON Schemas are never unfairly penalized relative to terse single-argument tools.
    """

    _W_NAME = 3.2
    _W_NAMESPACE = 1.8
    _W_DESC = 1.3
    _W_PROP = 1.6
    _W_PROP_DESC = 0.7

    def __init__(
        self,
        intent: str | None = None,
        *,
        top_k: int = 12,
        min_score_ratio: float = 0.15,
        always_keep: set[str] | list[str] | tuple[str, ...] | None = None,
        fallback_to_all: bool = False,
        slim_schemas: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not (0.0 <= min_score_ratio <= 1.0):
            raise ValueError("min_score_ratio must be in [0.0, 1.0]")
        self.intent = (intent or "").strip()
        self.top_k = int(top_k)
        self.min_score_ratio = float(min_score_ratio)
        self.always_keep: tuple[str, ...] = (
            tuple(sorted({str(x).strip() for x in always_keep if str(x).strip()}))
            if always_keep
            else ()
        )
        self.fallback_to_all = bool(fallback_to_all)
        self.slim_schemas = bool(slim_schemas)
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)

    @property
    def enabled(self) -> bool:
        return bool(self.intent)

    @staticmethod
    def _index_tool(tool: dict[str, Any]) -> _IndexedToolDoc | None:
        raw_name = str(tool.get("name", "")).strip()
        if not raw_name:
            return None

        ns_part = ""
        local_name = raw_name
        for sep in ("__", "/", ":", "."):
            if sep in raw_name:
                ns_part, _, local_name = raw_name.partition(sep)
                break

        name_toks = tokenize_mcp_text(local_name)
        ns_toks = tokenize_mcp_text(ns_part) if ns_part else []
        desc_toks = tokenize_mcp_text(str(tool.get("description", "")))
        prop_toks, prop_desc_toks = _extract_schema_zone_tokens(tool.get("inputSchema"))

        weighted_tf: dict[str, float] = {}
        weighted_len = 0.0
        for zone_toks, weight in (
            (name_toks, ToolCatalogPruner._W_NAME),
            (ns_toks, ToolCatalogPruner._W_NAMESPACE),
            (desc_toks, ToolCatalogPruner._W_DESC),
            (prop_toks, ToolCatalogPruner._W_PROP),
            (prop_desc_toks, ToolCatalogPruner._W_PROP_DESC),
        ):
            for tok in zone_toks:
                weighted_tf[tok] = weighted_tf.get(tok, 0.0) + weight
                weighted_len += weight

        return _IndexedToolDoc(
            name=raw_name,
            raw_tool=tool,
            weighted_tf=weighted_tf,
            weighted_len=max(1.0, weighted_len),
            vocab=set(weighted_tf.keys()),
        )

    def prune(
        self,
        tools: list[dict[str, Any]],
        *,
        intent: str | None = None,
        schema_slimmer: Any | None = None,
    ) -> ToolPruneResult:
        """Rank and prune ``tools`` for ``intent`` (falls back to ``self.intent``)."""
        from .slimmer import SchemaSlimmer, estimate_tokens  # noqa: PLC0415

        eff_intent = (intent if intent is not None else self.intent).strip()
        valid_tools = [t for t in tools if isinstance(t, dict) and str(t.get("name", "")).strip()]
        tokens_before = estimate_tokens(valid_tools) if valid_tools else 0

        if not valid_tools:
            return ToolPruneResult(
                intent=eff_intent,
                tools=[],
                ranked_scores=[],
                pruned_names=[],
                tools_before=0,
                tools_after=0,
                tokens_before=0,
                tokens_after=0,
                fallback_used=False,
            )

        slimmer = (
            schema_slimmer
            if schema_slimmer is not None
            else (SchemaSlimmer() if self.slim_schemas else None)
        )

        if not eff_intent:
            out_tools = (
                slimmer.slim_tools_list(copy.deepcopy(valid_tools))
                if slimmer is not None
                else copy.deepcopy(valid_tools)
            )
            tokens_after = estimate_tokens(out_tools)
            return ToolPruneResult(
                intent="",
                tools=out_tools,
                ranked_scores=[
                    {"name": str(t["name"]), "score": 1.0, "matched_terms": [], "pinned": False}
                    for t in out_tools
                ],
                pruned_names=[],
                tools_before=len(valid_tools),
                tools_after=len(out_tools),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                fallback_used=False,
            )

        docs: list[_IndexedToolDoc] = []
        df: dict[str, int] = {}
        for t in valid_tools:
            indexed = self._index_tool(t)
            if indexed is not None:
                docs.append(indexed)
                for tok in indexed.vocab:
                    df[tok] = df.get(tok, 0) + 1

        n_docs = len(docs)
        avgdl = sum(d.weighted_len for d in docs) / max(1, n_docs)

        # Primary query terms get weight 1.0; expanded synonyms get weight 0.55
        primary_q = tokenize_mcp_text(eff_intent, expand_synonyms=False)
        expanded_q = tokenize_mcp_text(eff_intent, expand_synonyms=True)
        q_weights: dict[str, float] = {}
        for qt in primary_q:
            q_weights[qt] = max(q_weights.get(qt, 0.0), 1.0)
        for qt in expanded_q:
            if qt not in q_weights:
                q_weights[qt] = 0.55

        norm_intent_sub = re.sub(r"[^a-z0-9]+", "_", eff_intent.lower()).strip("_")
        scored_docs: list[tuple[float, list[str], bool, _IndexedToolDoc]] = []

        for doc in docs:
            pinned = _matches_keep_pattern(doc.name, self.always_keep)
            score = 0.0
            matched_terms: list[str] = []

            low_name = doc.name.lower()
            if norm_intent_sub and len(norm_intent_sub) >= 3 and norm_intent_sub in low_name:
                score += 2.5

            for q_term, q_boost in q_weights.items():
                tf_val = doc.weighted_tf.get(q_term, 0.0)
                fuzzy_discount = 1.0
                if tf_val == 0.0 and q_boost >= 1.0 and len(q_term) >= 4:
                    # Prefix or 1-edit typo fallback against document vocabulary
                    for d_tok, d_tf in doc.weighted_tf.items():
                        if len(d_tok) >= 4 and (
                            d_tok.startswith(q_term)
                            or q_term.startswith(d_tok)
                            or _is_one_edit_away(q_term, d_tok)
                        ):
                            tf_val = max(tf_val, d_tf)
                            fuzzy_discount = 0.45

                if tf_val > 0.0:
                    doc_freq = df.get(q_term, 1)
                    idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                    norm_denom = tf_val + self.k1 * (
                        1.0 - self.b + self.b * (doc.weighted_len / max(1.0, avgdl))
                    )
                    bm25_plus = ((tf_val * (self.k1 + 1.0)) / norm_denom) + self.delta
                    contrib = idf * bm25_plus * q_boost * fuzzy_discount
                    score += contrib
                    if q_term not in matched_terms:
                        matched_terms.append(q_term)

            scored_docs.append((round(score, 4), sorted(matched_terms), pinned, doc))

        scored_docs.sort(key=lambda item: (-int(item[2]), -item[0], item[3].name))
        max_non_pinned_score = max(
            (s for s, _, _, _ in scored_docs),
            default=0.0,
        )
        cutoff = max_non_pinned_score * self.min_score_ratio if max_non_pinned_score > 0.0 else 0.0

        kept_docs: list[_IndexedToolDoc] = []
        ranked_entries: list[dict[str, Any]] = []
        non_pinned_kept = 0

        for score, matched_terms, pinned, doc in scored_docs:
            keep = False
            if pinned:
                keep = True
            elif score > 0.0 and score >= cutoff and non_pinned_kept < self.top_k:
                keep = True
                non_pinned_kept += 1

            if keep:
                kept_docs.append(doc)
            ranked_entries.append(
                {
                    "name": doc.name,
                    "score": score,
                    "matched_terms": matched_terms,
                    "pinned": pinned,
                    "kept": keep,
                }
            )

        fallback_used = False
        if not kept_docs and self.fallback_to_all:
            fallback_used = True
            sorted_by_name = sorted(docs, key=lambda d: d.name)[: self.top_k]
            kept_docs = sorted_by_name
            kept_set_fb = {d.name for d in kept_docs}
            for entry in ranked_entries:
                if entry["name"] in kept_set_fb:
                    entry["kept"] = True

        kept_names = {d.name for d in kept_docs}
        pruned_names = sorted(d.name for d in docs if d.name not in kept_names)
        raw_kept = [copy.deepcopy(d.raw_tool) for d in kept_docs]
        final_tools = (
            slimmer.slim_tools_list(raw_kept) if slimmer is not None and raw_kept else raw_kept
        )
        tokens_after = estimate_tokens(final_tools) if final_tools else 0

        return ToolPruneResult(
            intent=eff_intent,
            tools=final_tools,
            ranked_scores=ranked_entries,
            pruned_names=pruned_names,
            tools_before=len(docs),
            tools_after=len(final_tools),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            fallback_used=fallback_used,
        )


def filter_tools_for_intent(
    tools: list[dict[str, Any]],
    intent: str,
    *,
    top_k: int = 12,
    min_score_ratio: float = 0.15,
    always_keep: set[str] | list[str] | tuple[str, ...] | None = None,
    fallback_to_all: bool = False,
    slim_schemas: bool = False,
) -> ToolPruneResult:
    """Convenience function to rank and prune an MCP tools manifest by task intent (MS017)."""
    pruner = ToolCatalogPruner(
        intent=intent,
        top_k=top_k,
        min_score_ratio=min_score_ratio,
        always_keep=always_keep,
        fallback_to_all=fallback_to_all,
        slim_schemas=slim_schemas,
    )
    return pruner.prune(tools)


# ---------------------------------------------------------------------------
# MS018 -- JsonRpcResponseFirewall
# ---------------------------------------------------------------------------


def _rpc_id_key(raw_id: Any) -> tuple[type, int | str] | None:
    """Returns a type-strict correlation key for valid JSON-RPC 2.0 non-null IDs.

    Distinguishes ``int`` from ``str`` and explicitly rejects ``bool`` (since
    ``isinstance(True, int)`` is True in Python, which otherwise allows ``id: true``
    to collide with ``id: 1``).
    """
    if isinstance(raw_id, bool) or raw_id is None:
        return None
    if isinstance(raw_id, int):
        return (int, raw_id)
    if isinstance(raw_id, float):
        if raw_id.is_integer() and math.isfinite(raw_id):
            return (int, int(raw_id))
        return None
    if isinstance(raw_id, str):
        return (str, raw_id)
    return None


@dataclass(frozen=True)
class _InflightRequest:
    raw_id: int | str
    method: str
    sent_in_batch: bool
    registered_at: float


@dataclass(frozen=True)
class RpcFirewallViolation:
    """Single JSON-RPC 2.0 envelope or correlation violation (MS018)."""

    kind: str
    level: str  # "error" or "warning"
    direction: str  # "server" or "client"
    rpc_id: Any
    message: str

    def to_finding(self) -> dict[str, Any]:
        loc = f"rpc_id:{self.rpc_id!r}" if self.rpc_id is not None else "jsonrpc_envelope"
        return {
            "rule_id": _RPC_FIREWALL_RULE_ID,
            "level": self.level,
            "tool": loc,
            "kind": self.kind,
            "direction": self.direction,
            "rpc_id": repr(self.rpc_id) if self.rpc_id is not None else "null",
            "message": self.message,
        }

    def to_sarif_result(self) -> dict[str, Any]:
        f = self.to_finding()
        return {
            "ruleId": _RPC_FIREWALL_RULE_ID,
            "level": self.level,
            "message": {"text": self.message},
            "locations": [
                {
                    "logicalLocations": [
                        {"name": f["tool"], "kind": "function"}
                    ]
                }
            ],
            "properties": {
                "kind": self.kind,
                "direction": self.direction,
                "rpc_id": f["rpc_id"],
            },
        }


@dataclass
class RpcFirewallDecision:
    """Result of inspecting an inbound server JSON-RPC message or batch (MS018)."""

    allowed: bool
    sanitized_payload: Any | None
    violations: list[RpcFirewallViolation] = field(default_factory=list)
    dropped_count: int = 0


class JsonRpcResponseFirewall:
    """MS018: Stateful JSON-RPC 2.0 ID-correlation, replay, and batch-smuggling firewall.

    Guards MCP stdio/SSE sessions against:
      1. Unsolicited or type-confused server response IDs (e.g. ``"1"`` vs ``1``, ``true`` vs ``1``).
      2. Duplicate response ID replay (second response shadowing an already-resolved call).
      3. Illicit ``"id": null`` result injection or unsolicited ``"id": null`` error frames.
      4. Ambiguous dual-role envelopes carrying both ``method`` and ``result``/``error``.
      5. Unrequested or oversized JSON-RPC batch arrays and notifications/requests smuggled inside response batches.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        allow_batches: bool = True,
        require_batch_correlation: bool = True,
        max_batch_size: int = 32,
        max_inflight: int = 4096,
        max_completed_history: int = 4096,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_inflight < 16:
            raise ValueError("max_inflight must be >= 16")
        self.enabled = bool(enabled)
        self.allow_batches = bool(allow_batches)
        self.require_batch_correlation = bool(require_batch_correlation)
        self.max_batch_size = int(max_batch_size)
        self.max_inflight = int(max_inflight)
        self.max_completed_history = max(16, int(max_completed_history))
        self._inflight: OrderedDict[tuple[type, int | str], _InflightRequest] = OrderedDict()
        self._completed: OrderedDict[tuple[type, int | str], str] = OrderedDict()
        self._pending_batch_responses: int = 0
        self._unparsed_client_allowance: int = 0
        self.violations: list[RpcFirewallViolation] = []

    def reset(self) -> None:
        self._inflight.clear()
        self._completed.clear()
        self._pending_batch_responses = 0
        self._unparsed_client_allowance = 0
        self.violations.clear()

    def note_unparseable_client_frame(self) -> None:
        """Records that the client sent a malformed frame that may legitimately elicit an id=null error."""
        self._unparsed_client_allowance += 1

    def _record_violation(self, v: RpcFirewallViolation) -> None:
        self.violations.append(v)

    def _register_single_client_obj(
        self,
        obj: Any,
        *,
        in_batch: bool,
        now: float,
    ) -> list[RpcFirewallViolation]:
        errs: list[RpcFirewallViolation] = []
        if not isinstance(obj, dict):
            self._unparsed_client_allowance += 1
            v = RpcFirewallViolation(
                kind="invalid_client_envelope",
                level="warning",
                direction="client",
                rpc_id=None,
                message="Client JSON-RPC frame element is not an object.",
            )
            errs.append(v)
            self._record_violation(v)
            return errs

        method = obj.get("method")
        has_res_or_err = "result" in obj or "error" in obj
        if isinstance(method, str) and has_res_or_err:
            v = RpcFirewallViolation(
                kind="ambiguous_dual_role_envelope",
                level="error",
                direction="client",
                rpc_id=obj.get("id"),
                message="Client JSON-RPC frame contains both 'method' and 'result'/'error'.",
            )
            errs.append(v)
            self._record_violation(v)
            return errs

        if not isinstance(method, str) or "id" not in obj:
            # Notification or client->server response to server request
            return errs

        raw_id = obj.get("id")
        if raw_id is None:
            self._unparsed_client_allowance += 1
            return errs

        key = _rpc_id_key(raw_id)
        if key is None:
            self._unparsed_client_allowance += 1
            v = RpcFirewallViolation(
                kind="invalid_id_type",
                level="error",
                direction="client",
                rpc_id=raw_id,
                message=(
                    f"Client request for '{method}' uses non-compliant JSON-RPC id "
                    f"{raw_id!r} ({type(raw_id).__name__})."
                ),
            )
            errs.append(v)
            self._record_violation(v)
            return errs

        if key in self._inflight:
            v = RpcFirewallViolation(
                kind="duplicate_inflight_request_id",
                level="warning",
                direction="client",
                rpc_id=raw_id,
                message=(
                    f"Client reused in-flight request id {raw_id!r} for '{method}' "
                    f"before prior '{self._inflight[key].method}' completed."
                ),
            )
            errs.append(v)
            self._record_violation(v)

        # Allow intentional client ID reuse after a previous request already completed
        self._completed.pop(key, None)
        while len(self._inflight) >= self.max_inflight:
            _, evicted = self._inflight.popitem(last=False)
            if evicted.sent_in_batch and self._pending_batch_responses > 0:
                self._pending_batch_responses -= 1

        self._inflight[key] = _InflightRequest(
            raw_id=key[1],
            method=method,
            sent_in_batch=in_batch,
            registered_at=now,
        )
        if in_batch:
            self._pending_batch_responses += 1
        return errs

    def register_client_payload(self, payload: Any) -> list[RpcFirewallViolation]:
        """Registers outbound client -> server request IDs for stateful correlation."""
        now = time.monotonic()
        if isinstance(payload, dict):
            return self._register_single_client_obj(payload, in_batch=False, now=now)

        if isinstance(payload, list):
            errs: list[RpcFirewallViolation] = []
            if not payload:
                self._unparsed_client_allowance += 1
                v = RpcFirewallViolation(
                    kind="empty_batch",
                    level="error",
                    direction="client",
                    rpc_id=None,
                    message="Client sent an empty JSON-RPC batch array [].",
                )
                errs.append(v)
                self._record_violation(v)
                return errs
            if not self.allow_batches:
                v = RpcFirewallViolation(
                    kind="batch_disabled",
                    level="error",
                    direction="client",
                    rpc_id=None,
                    message="Client sent a JSON-RPC batch array while batching is disabled.",
                )
                errs.append(v)
                self._record_violation(v)
                return errs
            if len(payload) > self.max_batch_size:
                v = RpcFirewallViolation(
                    kind="oversized_batch",
                    level="error",
                    direction="client",
                    rpc_id=None,
                    message=(
                        f"Client JSON-RPC batch of {len(payload)} items exceeds "
                        f"max_batch_size={self.max_batch_size}."
                    ),
                )
                errs.append(v)
                self._record_violation(v)
            for item in payload:
                errs.extend(self._register_single_client_obj(item, in_batch=True, now=now))
            return errs

        self._unparsed_client_allowance += 1
        return []

    def _mark_completed(self, key: tuple[type, int | str], method: str) -> None:
        while len(self._completed) >= self.max_completed_history:
            self._completed.popitem(last=False)
        self._completed[key] = method

    def _inspect_single_server_obj(
        self,
        obj: Any,
        *,
        in_batch: bool,
        seen_batch_keys: set[tuple[type, int | str]] | None = None,
    ) -> tuple[bool, list[RpcFirewallViolation]]:
        violations: list[RpcFirewallViolation] = []
        if not isinstance(obj, dict):
            v = RpcFirewallViolation(
                kind="invalid_server_envelope",
                level="error",
                direction="server",
                rpc_id=None,
                message="Server JSON-RPC payload element is not a JSON object.",
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        has_method = "method" in obj
        has_result = "result" in obj
        has_error = "error" in obj

        if has_method and (has_result or has_error):
            v = RpcFirewallViolation(
                kind="ambiguous_dual_role_envelope",
                level="error",
                direction="server",
                rpc_id=obj.get("id"),
                message=(
                    "Server sent an ambiguous dual-role JSON-RPC envelope containing "
                    "both 'method' and 'result'/'error' (MS018)."
                ),
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        if has_result and has_error:
            v = RpcFirewallViolation(
                kind="ambiguous_result_and_error",
                level="error",
                direction="server",
                rpc_id=obj.get("id"),
                message=(
                    "Server sent a malformed JSON-RPC response containing both "
                    "'result' and 'error' fields (MS018)."
                ),
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        if has_method:
            if not isinstance(obj.get("method"), str) or not str(obj.get("method")).strip():
                v = RpcFirewallViolation(
                    kind="invalid_server_method",
                    level="error",
                    direction="server",
                    rpc_id=obj.get("id"),
                    message="Server sent a request/notification with a non-string or empty 'method'.",
                )
                violations.append(v)
                self._record_violation(v)
                return (False, violations)
            if in_batch:
                v = RpcFirewallViolation(
                    kind="batch_request_smuggling",
                    level="error",
                    direction="server",
                    rpc_id=obj.get("id"),
                    message=(
                        f"Server smuggled request/notification '{obj.get('method')}' "
                        "inside a JSON-RPC response batch (MS018)."
                    ),
                )
                violations.append(v)
                self._record_violation(v)
                return (False, violations)
            if "id" in obj and obj.get("id") is not None and _rpc_id_key(obj.get("id")) is None:
                v = RpcFirewallViolation(
                    kind="invalid_id_type",
                    level="error",
                    direction="server",
                    rpc_id=obj.get("id"),
                    message=(
                        f"Server-initiated request '{obj.get('method')}' uses invalid "
                        f"JSON-RPC id {obj.get('id')!r}."
                    ),
                )
                violations.append(v)
                self._record_violation(v)
                return (False, violations)
            return (True, violations)

        if not has_result and not has_error:
            v = RpcFirewallViolation(
                kind="invalid_server_envelope",
                level="error",
                direction="server",
                rpc_id=obj.get("id"),
                message="Server JSON-RPC object lacks 'method', 'result', and 'error'.",
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        if "id" not in obj:
            v = RpcFirewallViolation(
                kind="missing_response_id",
                level="error",
                direction="server",
                rpc_id=None,
                message="Server JSON-RPC response is missing required 'id' member (MS018).",
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        raw_id = obj.get("id")
        if raw_id is None:
            if has_result:
                v = RpcFirewallViolation(
                    kind="null_response_id",
                    level="error",
                    direction="server",
                    rpc_id=None,
                    message="Server sent a JSON-RPC 'result' with 'id': null (MS018).",
                )
                violations.append(v)
                self._record_violation(v)
                return (False, violations)
            if self._unparsed_client_allowance > 0:
                self._unparsed_client_allowance -= 1
                return (True, violations)
            v = RpcFirewallViolation(
                kind="null_response_id",
                level="error",
                direction="server",
                rpc_id=None,
                message=(
                    "Server sent an unsolicited JSON-RPC 'error' with 'id': null "
                    "when no malformed client frame was pending (MS018)."
                ),
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        key = _rpc_id_key(raw_id)
        if key is None:
            v = RpcFirewallViolation(
                kind="invalid_id_type",
                level="error",
                direction="server",
                rpc_id=raw_id,
                message=(
                    f"Server response uses non-compliant JSON-RPC id {raw_id!r} "
                    f"({type(raw_id).__name__}) (MS018)."
                ),
            )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        if seen_batch_keys is not None:
            if key in seen_batch_keys:
                v = RpcFirewallViolation(
                    kind="duplicate_batch_id",
                    level="error",
                    direction="server",
                    rpc_id=raw_id,
                    message=(
                        f"Server JSON-RPC batch contains duplicate response id {raw_id!r} (MS018)."
                    ),
                )
                violations.append(v)
                self._record_violation(v)
                return (False, violations)
            seen_batch_keys.add(key)

        pending = self._inflight.pop(key, None)
        if pending is None:
            if key in self._completed:
                prev_method = self._completed[key]
                v = RpcFirewallViolation(
                    kind="duplicate_response_id",
                    level="error",
                    direction="server",
                    rpc_id=raw_id,
                    message=(
                        f"Server replayed duplicate JSON-RPC response for already-completed "
                        f"request id {raw_id!r} ('{prev_method}') (MS018)."
                    ),
                )
            else:
                # Check if an ID of a different scalar type exists (e.g. "42" vs 42)
                alt_type_match = None
                if isinstance(raw_id, str) and raw_id.isdigit() and (int, int(raw_id)) in self._inflight:
                    alt_type_match = f"integer {int(raw_id)}"
                elif isinstance(raw_id, int) and (str, str(raw_id)) in self._inflight:
                    alt_type_match = f"string {str(raw_id)!r}"
                hint = (
                    f" (type-confused against pending {alt_type_match})"
                    if alt_type_match
                    else ""
                )
                v = RpcFirewallViolation(
                    kind="unsolicited_response_id",
                    level="error",
                    direction="server",
                    rpc_id=raw_id,
                    message=(
                        f"Server sent unsolicited JSON-RPC response for unknown request id "
                        f"{raw_id!r}{hint} (MS018)."
                    ),
                )
            violations.append(v)
            self._record_violation(v)
            return (False, violations)

        if pending.sent_in_batch and self._pending_batch_responses > 0:
            self._pending_batch_responses -= 1
        self._mark_completed(key, pending.method)
        return (True, violations)

    def inspect_server_payload(self, payload: Any) -> RpcFirewallDecision:
        """Inspects an inbound server -> client JSON-RPC object or batch array."""
        if not self.enabled:
            return RpcFirewallDecision(
                allowed=True,
                sanitized_payload=payload,
                violations=[],
                dropped_count=0,
            )

        if isinstance(payload, dict):
            ok, viols = self._inspect_single_server_obj(payload, in_batch=False)
            return RpcFirewallDecision(
                allowed=ok,
                sanitized_payload=payload if ok else None,
                violations=viols,
                dropped_count=0 if ok else 1,
            )

        if isinstance(payload, list):
            viols_all: list[RpcFirewallViolation] = []
            if not payload:
                v = RpcFirewallViolation(
                    kind="empty_batch",
                    level="error",
                    direction="server",
                    rpc_id=None,
                    message="Server sent an empty JSON-RPC batch array [] (MS018).",
                )
                viols_all.append(v)
                self._record_violation(v)
                return RpcFirewallDecision(
                    allowed=False,
                    sanitized_payload=None,
                    violations=viols_all,
                    dropped_count=1,
                )

            if not self.allow_batches:
                v = RpcFirewallViolation(
                    kind="unrequested_batch_response",
                    level="error",
                    direction="server",
                    rpc_id=None,
                    message="Server sent a JSON-RPC batch array while batching is disabled (MS018).",
                )
                viols_all.append(v)
                self._record_violation(v)
                return RpcFirewallDecision(
                    allowed=False,
                    sanitized_payload=None,
                    violations=viols_all,
                    dropped_count=len(payload),
                )

            if len(payload) > self.max_batch_size:
                v = RpcFirewallViolation(
                    kind="oversized_batch",
                    level="error",
                    direction="server",
                    rpc_id=None,
                    message=(
                        f"Server JSON-RPC batch of {len(payload)} items exceeds "
                        f"max_batch_size={self.max_batch_size} (MS018)."
                    ),
                )
                viols_all.append(v)
                self._record_violation(v)
                return RpcFirewallDecision(
                    allowed=False,
                    sanitized_payload=None,
                    violations=viols_all,
                    dropped_count=len(payload),
                )

            if self.require_batch_correlation and self._pending_batch_responses <= 0:
                v = RpcFirewallViolation(
                    kind="unrequested_batch_response",
                    level="error",
                    direction="server",
                    rpc_id=None,
                    message=(
                        f"Server returned a {len(payload)}-item JSON-RPC batch array "
                        "when the client sent only single-frame requests (MS018)."
                    ),
                )
                viols_all.append(v)
                self._record_violation(v)
                return RpcFirewallDecision(
                    allowed=False,
                    sanitized_payload=None,
                    violations=viols_all,
                    dropped_count=len(payload),
                )

            seen_batch_keys: set[tuple[type, int | str]] = set()
            kept_items: list[dict[str, Any]] = []
            dropped = 0
            for elem in payload:
                ok_elem, elem_viols = self._inspect_single_server_obj(
                    elem,
                    in_batch=True,
                    seen_batch_keys=seen_batch_keys,
                )
                viols_all.extend(elem_viols)
                if ok_elem and isinstance(elem, dict):
                    kept_items.append(elem)
                else:
                    dropped += 1

            if not kept_items:
                return RpcFirewallDecision(
                    allowed=False,
                    sanitized_payload=None,
                    violations=viols_all,
                    dropped_count=max(1, dropped),
                )
            return RpcFirewallDecision(
                allowed=True,
                sanitized_payload=kept_items,
                violations=viols_all,
                dropped_count=dropped,
            )

        v = RpcFirewallViolation(
            kind="invalid_server_envelope",
            level="error",
            direction="server",
            rpc_id=None,
            message="Server sent a top-level JSON value that is neither an object nor a batch array (MS018).",
        )
        self._record_violation(v)
        return RpcFirewallDecision(
            allowed=False,
            sanitized_payload=None,
            violations=[v],
            dropped_count=1,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_batches": self.allow_batches,
            "require_batch_correlation": self.require_batch_correlation,
            "max_batch_size": self.max_batch_size,
            "inflight_requests": len(self._inflight),
            "completed_tracked": len(self._completed),
            "violations_total": len(self.violations),
        }


def audit_rpc_transcript(
    events: list[dict[str, Any]],
    *,
    allow_batches: bool = True,
    require_batch_correlation: bool = True,
    max_batch_size: int = 32,
) -> dict[str, Any]:
    """Audits an ordered sequence of JSON-RPC 2.0 client/server events for MS018 anomalies.

    Each element in ``events`` should be ``{"direction": "client"|"server", "message": <obj_or_raw_json>}``.
    """
    fw = JsonRpcResponseFirewall(
        enabled=True,
        allow_batches=allow_batches,
        require_batch_correlation=require_batch_correlation,
        max_batch_size=max_batch_size,
    )
    frames_evaluated = 0
    frames_blocked = 0
    findings: list[dict[str, Any]] = []

    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            continue
        direction = str(ev.get("direction", "server")).strip().lower()
        raw_msg = ev.get("message")
        parsed_msg = raw_msg
        if isinstance(raw_msg, str):
            try:
                parsed_msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                if direction == "client":
                    fw.note_unparseable_client_frame()
                else:
                    frames_blocked += 1
                    findings.append({
                        "rule_id": _RPC_FIREWALL_RULE_ID,
                        "level": "error",
                        "tool": f"frame_{idx}",
                        "kind": "invalid_json",
                        "direction": "server",
                        "rpc_id": "null",
                        "message": f"Server frame #{idx} is not valid JSON (MS018).",
                    })
                continue

        frames_evaluated += 1
        if direction == "client":
            c_viols = fw.register_client_payload(parsed_msg)
            for cv in c_viols:
                findings.append(cv.to_finding())
        else:
            dec = fw.inspect_server_payload(parsed_msg)
            if not dec.allowed or dec.dropped_count > 0:
                frames_blocked += max(1, dec.dropped_count)
            for sv in dec.violations:
                findings.append(sv.to_finding())

    blocking = sum(1 for f in findings if f.get("level") == "error")
    return {
        "frames_evaluated": frames_evaluated,
        "frames_blocked": frames_blocked,
        "inflight_unanswered": len(fw._inflight),
        "security_findings": blocking,
        "findings": findings,
    }
