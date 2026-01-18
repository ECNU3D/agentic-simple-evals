"""LangGraph pipeline for migrating AEM components to a CMS React library.

This module provides a concrete, extensible workflow for:
- Parsing AEM component bundles.
- Mapping components to a BDL design spec.
- Generating React components and CMS edit configuration.
- Running automated and human review gates.
- Converting AEM page JSON into CMS-renderable JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, TypedDict

from langgraph.graph import END, StateGraph


class ComponentBundle(TypedDict):
    component_id: str
    template_files: List[str]
    style_files: List[str]
    script_files: List[str]
    dialog_files: List[str]
    metadata: Dict[str, Any]


class NormalizedComponentSpec(TypedDict):
    layout_tree: Dict[str, Any]
    style_tokens: Dict[str, Any]
    behavior_map: Dict[str, Any]
    aem_dialog_schema: Dict[str, Any]
    component_meta: Dict[str, Any]


class BDLComponentSpec(TypedDict):
    bdl_type: str
    props_schema: Dict[str, Any]
    slot_map: Dict[str, Any]
    design_tokens: Dict[str, Any]
    interaction_contract: Dict[str, Any]
    compliance: Dict[str, Any]


class ReviewIssue(TypedDict):
    check_id: str
    severity: str
    message: str
    details: Dict[str, Any]


class ReviewReport(TypedDict):
    status: str
    issues: List[ReviewIssue]
    suggestions: List[Dict[str, Any]]


class PageConversionSpec(TypedDict):
    aem_page_json: Dict[str, Any]
    cms_page_json: Dict[str, Any]
    mapping_report: Dict[str, Any]


class PipelineState(TypedDict, total=False):
    component_bundle: ComponentBundle
    normalized_component_spec: NormalizedComponentSpec
    bdl_component_spec: BDLComponentSpec
    react_component_files: Dict[str, str]
    cms_config_schema: Dict[str, Any]
    review_report: ReviewReport
    human_review: Dict[str, Any]
    publish_status: Dict[str, Any]
    page_conversion_spec: PageConversionSpec
    audit_events: List[Dict[str, Any]]


@dataclass
class PipelineConfig:
    enforce_human_review: bool = True
    review_fail_threshold: int = 1
    component_registry_path: str = "./generated_components"
    page_registry_path: str = "./generated_pages"
    bdl_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    component_templates: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineContext:
    config: PipelineConfig
    audit_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.audit_log.append(message)


class ComponentParser(Protocol):
    def parse(self, bundle: ComponentBundle) -> NormalizedComponentSpec:
        raise NotImplementedError


class BDLMapper(Protocol):
    def map(self, spec: NormalizedComponentSpec) -> BDLComponentSpec:
        raise NotImplementedError


class ReactGenerator(Protocol):
    def generate(self, spec: BDLComponentSpec) -> Dict[str, str]:
        raise NotImplementedError


class ConfigSchemaGenerator(Protocol):
    def generate(self, spec: BDLComponentSpec) -> Dict[str, Any]:
        raise NotImplementedError


class PageConverter(Protocol):
    def convert(self, aem_json: Dict[str, Any]) -> PageConversionSpec:
        raise NotImplementedError


class ReviewRule(Protocol):
    def check(self, state: PipelineState) -> Optional[ReviewIssue]:
        raise NotImplementedError


def _audit(state: PipelineState, ctx: PipelineContext, event: str, payload: Dict[str, Any]) -> PipelineState:
    audit_events = list(state.get("audit_events", []))
    audit_events.append({"event": event, "payload": payload})
    return {**state, "audit_events": audit_events}


@dataclass
class DefaultComponentParser:
    def parse(self, bundle: ComponentBundle) -> NormalizedComponentSpec:
        return {
            "layout_tree": {"root": bundle["component_id"], "children": []},
            "style_tokens": {"source_files": bundle["style_files"]},
            "behavior_map": {"source_files": bundle["script_files"]},
            "aem_dialog_schema": {"source_files": bundle["dialog_files"]},
            "component_meta": bundle.get("metadata", {}),
        }


@dataclass
class DefaultBDLMapper:
    bdl_registry: Dict[str, Dict[str, Any]]

    def map(self, spec: NormalizedComponentSpec) -> BDLComponentSpec:
        component_type = spec["component_meta"].get("bdl_type", "Card")
        registry_entry = self.bdl_registry.get(component_type, {})
        return {
            "bdl_type": component_type,
            "props_schema": registry_entry.get("props_schema", {}),
            "slot_map": registry_entry.get("slot_map", {"default": "root"}),
            "design_tokens": spec["style_tokens"],
            "interaction_contract": spec["behavior_map"],
            "compliance": {"bdl_version": registry_entry.get("version", "unknown")},
        }


@dataclass
class DefaultReactGenerator:
    templates: Dict[str, str]

    def generate(self, spec: BDLComponentSpec) -> Dict[str, str]:
        component_name = spec["bdl_type"]
        template = self.templates.get(
            component_name,
            "import React from 'react';\n\n"
            "export const {component_name}: React.FC<any> = (props) => (\n"
            "  <div data-bdl=\"{component_name}\">{props.children}</div>\n"
            ");\n",
        )
        rendered = template.format(component_name=component_name)
        return {
            f"{component_name}.tsx": rendered,
            f"{component_name}.module.scss": ".root { display: block; }\n",
            "index.ts": f"export * from './{component_name}';\n",
        }


@dataclass
class DefaultConfigSchemaGenerator:
    def generate(self, spec: BDLComponentSpec) -> Dict[str, Any]:
        return {
            "component": spec["bdl_type"],
            "fields": spec["props_schema"],
            "slots": spec["slot_map"],
            "validation": {"required": []},
        }


@dataclass
class DefaultPageConverter:
    def convert(self, aem_json: Dict[str, Any]) -> PageConversionSpec:
        cms_json = {
            "type": "page",
            "children": aem_json.get("children", []),
        }
        return {
            "aem_page_json": aem_json,
            "cms_page_json": cms_json,
            "mapping_report": {"mapped_components": len(cms_json["children"])},
        }


@dataclass
class ReviewEngine:
    rules: Iterable[ReviewRule]

    def run(self, state: PipelineState) -> ReviewReport:
        issues: List[ReviewIssue] = []
        for rule in self.rules:
            issue = rule.check(state)
            if issue:
                issues.append(issue)
        status = "fail" if issues else "pass"
        return {"status": status, "issues": issues, "suggestions": []}


@dataclass
class HasReactFilesRule:
    def check(self, state: PipelineState) -> Optional[ReviewIssue]:
        if state.get("react_component_files"):
            return None
        return {
            "check_id": "react_files",
            "severity": "error",
            "message": "React component files missing",
            "details": {},
        }


@dataclass
class HasConfigSchemaRule:
    def check(self, state: PipelineState) -> Optional[ReviewIssue]:
        if state.get("cms_config_schema"):
            return None
        return {
            "check_id": "cms_schema",
            "severity": "error",
            "message": "CMS config schema missing",
            "details": {},
        }


@dataclass
class HasPageJsonRule:
    def check(self, state: PipelineState) -> Optional[ReviewIssue]:
        page_spec = state.get("page_conversion_spec", {})
        if page_spec.get("cms_page_json"):
            return None
        return {
            "check_id": "page_json",
            "severity": "error",
            "message": "CMS page JSON missing",
            "details": {},
        }


# -------- Component pipeline nodes --------

def ingest_aem_source(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Ingesting AEM source bundle")
    if "component_bundle" not in state:
        raise ValueError("component_bundle is required")
    return _audit(state, ctx, "component.ingest", {"component_id": state["component_bundle"]["component_id"]})


def parse_and_normalize(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Parsing and normalizing AEM component")
    parser = DefaultComponentParser()
    normalized = parser.parse(state["component_bundle"])
    next_state: PipelineState = {**state, "normalized_component_spec": normalized}
    return _audit(next_state, ctx, "component.normalized", {"layout": normalized["layout_tree"]})


def map_to_bdl(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Mapping AEM spec to BDL spec")
    mapper = DefaultBDLMapper(ctx.config.bdl_registry)
    bdl_spec = mapper.map(state["normalized_component_spec"])
    next_state: PipelineState = {**state, "bdl_component_spec": bdl_spec}
    return _audit(next_state, ctx, "component.mapped", {"bdl_type": bdl_spec["bdl_type"]})


def generate_react_component(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Generating React component files")
    generator = DefaultReactGenerator(ctx.config.component_templates)
    files = generator.generate(state["bdl_component_spec"])
    next_state: PipelineState = {**state, "react_component_files": files}
    return _audit(next_state, ctx, "component.react_generated", {"files": list(files.keys())})


def generate_cms_config(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Generating CMS config schema")
    generator = DefaultConfigSchemaGenerator()
    schema = generator.generate(state["bdl_component_spec"])
    next_state: PipelineState = {**state, "cms_config_schema": schema}
    return _audit(next_state, ctx, "component.cms_schema", {"component": schema["component"]})


def automated_review(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Running automated review")
    engine = ReviewEngine(rules=[HasReactFilesRule(), HasConfigSchemaRule()])
    report = engine.run(state)
    next_state: PipelineState = {**state, "review_report": report}
    return _audit(next_state, ctx, "component.review", {"status": report["status"]})


def human_review_gate(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Evaluating human review gate")
    report = state["review_report"]
    needs_review = ctx.config.enforce_human_review or report["status"] == "fail"
    decision = {
        "required": needs_review,
        "approved": not needs_review,
        "review_notes": [],
    }
    next_state: PipelineState = {**state, "human_review": decision}
    return _audit(next_state, ctx, "component.human_review", {"required": needs_review})


def publish_component(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Publishing component to registry")
    publish_status = {
        "component_id": state["component_bundle"]["component_id"],
        "status": "published",
        "registry_path": ctx.config.component_registry_path,
    }
    next_state: PipelineState = {**state, "publish_status": publish_status}
    return _audit(next_state, ctx, "component.publish", publish_status)


# -------- Page JSON conversion nodes --------

def ingest_aem_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Ingesting AEM page JSON")
    if "page_conversion_spec" not in state:
        raise ValueError("page_conversion_spec is required")
    return _audit(state, ctx, "page.ingest", {"has_json": True})


def normalize_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Normalizing AEM page JSON")
    page_spec = state["page_conversion_spec"]
    page_spec["aem_page_json"] = page_spec.get("aem_page_json", {})
    next_state: PipelineState = {**state, "page_conversion_spec": page_spec}
    return _audit(next_state, ctx, "page.normalized", {"keys": list(page_spec["aem_page_json"].keys())})


def bind_components_to_page(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Binding components and mapping slots")
    converter = DefaultPageConverter()
    page_spec = converter.convert(state["page_conversion_spec"]["aem_page_json"])
    next_state: PipelineState = {**state, "page_conversion_spec": page_spec}
    return _audit(next_state, ctx, "page.mapped", page_spec["mapping_report"])


def review_page_rendering(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Reviewing page rendering")
    engine = ReviewEngine(rules=[HasPageJsonRule()])
    report = engine.run(state)
    next_state: PipelineState = {**state, "review_report": report}
    return _audit(next_state, ctx, "page.review", {"status": report["status"]})


def publish_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Publishing page JSON")
    publish_status = {
        "page_status": "published",
        "registry_path": ctx.config.page_registry_path,
    }
    next_state: PipelineState = {**state, "publish_status": publish_status}
    return _audit(next_state, ctx, "page.publish", publish_status)


# -------- Graph builders --------

def build_component_graph() -> StateGraph:
    graph: StateGraph = StateGraph(PipelineState)
    graph.add_node("ingest_aem_source", ingest_aem_source)
    graph.add_node("parse_and_normalize", parse_and_normalize)
    graph.add_node("map_to_bdl", map_to_bdl)
    graph.add_node("generate_react_component", generate_react_component)
    graph.add_node("generate_cms_config", generate_cms_config)
    graph.add_node("automated_review", automated_review)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("publish_component", publish_component)

    graph.set_entry_point("ingest_aem_source")
    graph.add_edge("ingest_aem_source", "parse_and_normalize")
    graph.add_edge("parse_and_normalize", "map_to_bdl")
    graph.add_edge("map_to_bdl", "generate_react_component")
    graph.add_edge("generate_react_component", "generate_cms_config")
    graph.add_edge("generate_cms_config", "automated_review")
    graph.add_edge("automated_review", "human_review_gate")
    graph.add_edge("human_review_gate", "publish_component")
    graph.add_edge("publish_component", END)
    return graph


def build_page_graph() -> StateGraph:
    graph: StateGraph = StateGraph(PipelineState)
    graph.add_node("ingest_aem_page_json", ingest_aem_page_json)
    graph.add_node("normalize_page_json", normalize_page_json)
    graph.add_node("bind_components_to_page", bind_components_to_page)
    graph.add_node("review_page_rendering", review_page_rendering)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("publish_page_json", publish_page_json)

    graph.set_entry_point("ingest_aem_page_json")
    graph.add_edge("ingest_aem_page_json", "normalize_page_json")
    graph.add_edge("normalize_page_json", "bind_components_to_page")
    graph.add_edge("bind_components_to_page", "review_page_rendering")
    graph.add_edge("review_page_rendering", "human_review_gate")
    graph.add_edge("human_review_gate", "publish_page_json")
    graph.add_edge("publish_page_json", END)
    return graph


def build_pipeline() -> Dict[str, StateGraph]:
    return {
        "component_graph": build_component_graph(),
        "page_graph": build_page_graph(),
    }
