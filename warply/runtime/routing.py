from __future__ import annotations

from dataclasses import replace

from warply.compiler.plan import DeploymentPlan, RoutingConfig
from warply.providers.base import Node


def node_http_url(node: Node) -> str:
    return f"http://{node.host}:{node.port}"


def primary_pool_url(nodes: list[Node]) -> str:
    """Return the HTTP URL for the first healthy node in a pool."""
    if not nodes:
        raise ValueError("cannot resolve routing for an empty node pool")
    return node_http_url(nodes[0])


def resolve_routing(
    plan: DeploymentPlan,
    *,
    prefill_nodes: list[Node],
    decode_nodes: list[Node],
    router_node: Node,
) -> DeploymentPlan:
    """Replace placeholder routing with URLs derived from provisioned nodes."""
    routing = RoutingConfig(
        mode=plan.routing.mode,
        router_port=plan.routing.router_port,
        endpoint=node_http_url(router_node),
        prefill_base_url=primary_pool_url(prefill_nodes),
        decode_base_url=primary_pool_url(decode_nodes),
    )
    return replace(plan, routing=routing)


def resolve_router_endpoint(plan: DeploymentPlan, *, router_node: Node) -> DeploymentPlan:
    """Replace only the public router endpoint after cluster launch."""
    routing = RoutingConfig(
        mode=plan.routing.mode,
        router_port=plan.routing.router_port,
        endpoint=node_http_url(router_node),
        prefill_base_url=plan.routing.prefill_base_url,
        decode_base_url=plan.routing.decode_base_url,
    )
    return replace(plan, routing=routing)
