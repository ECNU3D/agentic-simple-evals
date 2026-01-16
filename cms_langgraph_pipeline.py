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
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph


class ComponentBundle(TypedDict):
    component_id: str
    template_files: List[str]
    style_files: List[str]
    script_files: List[str]
    dialog_files: List[str]


class NormalizedComponentSpec(TypedDict):
    layout_tree: Dict[str, Any]
    style_tokens: Dict[str, Any]
    behavior_map: Dict[str, Any]
    aem_dialog_schema: Dict[str, Any]


class BDLComponentSpec(TypedDict):
    bdl_type: str
    props_schema: Dict[str, Any]
    slot_map: Dict[str, Any]
    design_tokens: Dict[str, Any]
    interaction_contract: Dict[str, Any]


class ReviewReport(TypedDict):
    status: str
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]


class PageConversionSpec(TypedDict):
    aem_page_json: Dict[str, Any]
    cms_page_json: Dict[str, Any]


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


@dataclass
class PipelineConfig:
    enforce_human_review: bool = True
    review_fail_threshold: int = 1
    component_registry_path: str = "./generated_components"
    page_registry_path: str = "./generated_pages"


@dataclass
class PipelineContext:
    config: PipelineConfig
    audit_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.audit_log.append(message)


# -------- Component pipeline nodes --------

def ingest_aem_source(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Ingesting AEM source bundle")
    if "component_bundle" not in state:
        raise ValueError("component_bundle is required")
    return state


def parse_and_normalize(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Parsing and normalizing AEM component")
    bundle = state["component_bundle"]
    normalized: NormalizedComponentSpec = {
        "layout_tree": {"root": bundle["component_id"]},
        "style_tokens": {"source_files": bundle["style_files"]},
        "behavior_map": {"source_files": bundle["script_files"]},
        "aem_dialog_schema": {"source_files": bundle["dialog_files"]},
    }
    return {**state, "normalized_component_spec": normalized}


def map_to_bdl(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Mapping AEM spec to BDL spec")
    normalized = state["normalized_component_spec"]
    bdl_spec: BDLComponentSpec = {
        "bdl_type": "Card",
        "props_schema": {"fields": list(normalized["aem_dialog_schema"].keys())},
        "slot_map": {"default": "root"},
        "design_tokens": normalized["style_tokens"],
        "interaction_contract": normalized["behavior_map"],
    }
    return {**state, "bdl_component_spec": bdl_spec}


def generate_react_component(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Generating React component files")
    bdl_spec = state["bdl_component_spec"]
    component_name = bdl_spec["bdl_type"]
    react_files = {
        f"{component_name}.tsx": (
            "import React from 'react';\n"
            "\n"
            f"export const {component_name}: React.FC<any> = (props) => (\n"
            "  <div data-bdl=\"card\">\n"
            "    {props.children}\n"
            "  </div>\n"
            ");\n"
        ),
        f"{component_name}.module.scss": ".card { display: block; }\n",
        "index.ts": f"export * from './{component_name}';\n",
    }
    return {**state, "react_component_files": react_files}


def generate_cms_config(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Generating CMS config schema")
    bdl_spec = state["bdl_component_spec"]
    schema = {
        "component": bdl_spec["bdl_type"],
        "fields": bdl_spec["props_schema"],
        "slots": bdl_spec["slot_map"],
    }
    return {**state, "cms_config_schema": schema}


def automated_review(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Running automated review")
    issues: List[Dict[str, Any]] = []
    if not state.get("react_component_files"):
        issues.append({"type": "missing_component", "message": "React files missing"})
    if not state.get("cms_config_schema"):
        issues.append({"type": "missing_schema", "message": "CMS config schema missing"})
    report: ReviewReport = {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "suggestions": [],
    }
    return {**state, "review_report": report}


def human_review_gate(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Evaluating human review gate")
    report = state["review_report"]
    needs_review = ctx.config.enforce_human_review or report["status"] == "fail"
    decision = {
        "required": needs_review,
        "approved": not needs_review,
        "review_notes": [],
    }
    return {**state, "human_review": decision}


def publish_component(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Publishing component to registry")
    publish_status = {
        "component_id": state["component_bundle"]["component_id"],
        "status": "published",
        "registry_path": ctx.config.component_registry_path,
    }
    return {**state, "publish_status": publish_status}


# -------- Page JSON conversion nodes --------

def ingest_aem_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Ingesting AEM page JSON")
    if "page_conversion_spec" not in state:
        raise ValueError("page_conversion_spec is required")
    return state


def normalize_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Normalizing AEM page JSON")
    page_spec = state["page_conversion_spec"]
    page_spec["aem_page_json"] = page_spec.get("aem_page_json", {})
    return {**state, "page_conversion_spec": page_spec}


def bind_components_to_page(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Binding components and mapping slots")
    page_spec = state["page_conversion_spec"]
    page_spec["cms_page_json"] = {
        "type": "page",
        "children": page_spec.get("aem_page_json", {}).get("children", []),
    }
    return {**state, "page_conversion_spec": page_spec}


def review_page_rendering(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Reviewing page rendering")
    page_spec = state["page_conversion_spec"]
    issues = []
    if not page_spec.get("cms_page_json"):
        issues.append({"type": "missing_page", "message": "CMS page JSON missing"})
    report: ReviewReport = {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "suggestions": [],
    }
    return {**state, "review_report": report}


def publish_page_json(state: PipelineState, ctx: PipelineContext) -> PipelineState:
    ctx.log("Publishing page JSON")
    publish_status = {
        "page_status": "published",
        "registry_path": ctx.config.page_registry_path,
    }
    return {**state, "publish_status": publish_status}


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
