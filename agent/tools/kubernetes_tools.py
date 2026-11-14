"""Kubernetes tools. Read ops are free; mutating ops need confirmation and the approved namespace."""
from __future__ import annotations

from datetime import UTC, datetime

from agent.tools.common import tool
from kubernetes import client, config

APPROVED_NAMESPACE = "mlops-app"
APPROVED_DEPLOYMENTS = {"inference"}
_loaded = False


def _load() -> None:
    global _loaded
    if not _loaded:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        _loaded = True


def _core() -> client.CoreV1Api:
    _load()
    return client.CoreV1Api()


def _apps() -> client.AppsV1Api:
    _load()
    return client.AppsV1Api()


def _check_ns(namespace: str) -> str:
    if namespace != APPROVED_NAMESPACE:
        raise PermissionError(
            f"Namespace {namespace!r} is not approved. Allowed: {APPROVED_NAMESPACE!r}"
        )
    return namespace


def _pod_summary(p) -> dict:
    statuses = p.status.container_statuses or []
    waiting = [
        s.state.waiting.reason for s in statuses if s.state and s.state.waiting
    ]
    return {
        "name": p.metadata.name,
        "phase": p.status.phase,
        "ready": all(s.ready for s in statuses) if statuses else False,
        "restarts": sum(s.restart_count for s in statuses),
        "waiting_reason": waiting[0] if waiting else None,
        "node": p.spec.node_name,
        "image": p.spec.containers[0].image,
    }


@tool
def list_pods(namespace: str = APPROVED_NAMESPACE) -> list[dict]:
    """List pods in the approved namespace."""
    pods = _core().list_namespaced_pod(_check_ns(namespace)).items
    return [_pod_summary(p) for p in pods]


@tool
def list_deployments(namespace: str = APPROVED_NAMESPACE) -> list[dict]:
    """List deployments with desired/ready replicas."""
    deps = _apps().list_namespaced_deployment(_check_ns(namespace)).items
    return [
        {
            "name": d.metadata.name,
            "desired": d.spec.replicas,
            "ready": d.status.ready_replicas or 0,
            "available": d.status.available_replicas or 0,
            "image": d.spec.template.spec.containers[0].image,
        }
        for d in deps
    ]


@tool
def get_pod_logs(
    pod_name: str, namespace: str = APPROVED_NAMESPACE, tail: int = 50
) -> dict:
    """Last N log lines of a pod."""
    if not 1 <= tail <= 500:
        raise ValueError("tail must be between 1 and 500")
    text = _core().read_namespaced_pod_log(
        pod_name, _check_ns(namespace), tail_lines=tail
    )
    return {"pod": pod_name, "lines": text.splitlines()}


@tool
def list_services(namespace: str = APPROVED_NAMESPACE) -> list[dict]:
    """Services and their ports/endpoints."""
    ns = _check_ns(namespace)
    core = _core()
    out = []
    for service in core.list_namespaced_service(ns).items:
        endpoints = core.list_namespaced_endpoints(
            ns, field_selector=f"metadata.name={service.metadata.name}"
        )
        ready = sum(
            len(subset.addresses or [])
            for endpoint in endpoints.items
            for subset in (endpoint.subsets or [])
        )
        out.append({
            "name": service.metadata.name,
            "type": service.spec.type,
            "cluster_ip": service.spec.cluster_ip,
            "ports": [f"{port.port}->{port.target_port}" for port in service.spec.ports],
            "ready_endpoints": ready,
        })
    return out


@tool
def find_unhealthy_pods(namespace: str = APPROVED_NAMESPACE) -> dict:
    """Unhealthy pods, restart counts and recent warning events."""
    ns = _check_ns(namespace)
    core = _core()
    pods = [_pod_summary(pod) for pod in core.list_namespaced_pod(ns).items]
    bad = [
        pod
        for pod in pods
        if not pod["ready"]
        or pod["restarts"] > 0
        or pod["waiting_reason"]
        or pod["phase"] != "Running"
    ]
    events = core.list_namespaced_event(ns).items
    warnings = [
        {"object": event.involved_object.name, "reason": event.reason, "message": event.message}
        for event in events
        if event.type == "Warning"
    ][-10:]
    return {
        "unhealthy": bad,
        "healthy_count": len(pods) - len(bad),
        "warning_events": warnings,
    }


@tool
def get_deployed_version(
    namespace: str = APPROVED_NAMESPACE, deployment: str = "inference"
) -> dict:
    """Deployed image tag of a deployment in the approved namespace."""
    ns = _check_ns(namespace)
    deployment_obj = _apps().read_namespaced_deployment(deployment, ns)
    image = deployment_obj.spec.template.spec.containers[0].image
    return {
        "deployment": deployment,
        "image": image,
        "tag": image.rsplit(":", 1)[-1],
    }


@tool
def restart_deployment(
    deployment: str = "inference",
    namespace: str = APPROVED_NAMESPACE,
    confirmed: bool = False,
) -> dict:
    """Rolling restart. Requires confirmation, approved namespace and approved deployment."""
    ns = _check_ns(namespace)
    if deployment not in APPROVED_DEPLOYMENTS:
        raise PermissionError(f"Deployment {deployment!r} is not approved.")
    if confirmed is not True:
        raise PermissionError("Restarting a deployment needs explicit user confirmation.")
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": datetime.now(
                            UTC
                        ).isoformat()
                    }
                }
            }
        }
    }
    _apps().patch_namespaced_deployment(deployment, ns, body)
    return {"deployment": deployment, "action": "rolling restart requested"}