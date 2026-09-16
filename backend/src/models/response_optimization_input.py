"""Pure input contract for response-plan optimization."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.optimization_validation import resource_sort_key, validate_positive_int


@dataclass(frozen=True)
class ResponseOptimizationInput:
    """Normalized facts required by future response-allocation optimization."""

    fire_event_id: int
    response_target_set_id: int
    route_planning_run_id: int
    targets: tuple[OptimizationTarget, ...]
    resources: tuple[OptimizationResource, ...]
    route_options: tuple[OptimizationRouteOption, ...]

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_set_id", self.response_target_set_id)
        validate_positive_int("route_planning_run_id", self.route_planning_run_id)

        targets = self._normalize_targets(self.targets)
        resources = self._normalize_resources(self.resources)
        route_options = self._normalize_route_options(self.route_options, targets, resources)

        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "route_options", route_options)

    @staticmethod
    def _normalize_targets(targets) -> tuple[OptimizationTarget, ...]:
        try:
            target_tuple = tuple(targets)
        except TypeError as exc:
            raise ValueError("targets must be iterable.") from exc
        for target in target_tuple:
            if not isinstance(target, OptimizationTarget):
                raise ValueError(f"targets must contain OptimizationTarget items, got {target!r}")
        target_ids = [target.response_target_id for target in target_tuple]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("targets must have unique response_target_id values.")
        target_orders = [target.target_order for target in target_tuple]
        if len(target_orders) != len(set(target_orders)):
            raise ValueError("targets must have unique target_order values.")
        return tuple(sorted(target_tuple, key=lambda target: target.ordering_key))

    @staticmethod
    def _normalize_resources(resources) -> tuple[OptimizationResource, ...]:
        try:
            resource_tuple = tuple(resources)
        except TypeError as exc:
            raise ValueError("resources must be iterable.") from exc
        for resource in resource_tuple:
            if not isinstance(resource, OptimizationResource):
                raise ValueError(f"resources must contain OptimizationResource items, got {resource!r}")
        resource_ids = [resource.resource_id for resource in resource_tuple]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resources must have unique resource_id values.")
        return tuple(sorted(resource_tuple, key=lambda resource: resource.ordering_key))

    @staticmethod
    def _normalize_route_options(
        route_options,
        targets: tuple[OptimizationTarget, ...],
        resources: tuple[OptimizationResource, ...],
    ) -> tuple[OptimizationRouteOption, ...]:
        try:
            route_tuple = tuple(route_options)
        except TypeError as exc:
            raise ValueError("route_options must be iterable.") from exc
        for route_option in route_tuple:
            if not isinstance(route_option, OptimizationRouteOption):
                raise ValueError(f"route_options must contain OptimizationRouteOption items, got {route_option!r}")

        route_ids = [route.route_result_id for route in route_tuple]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_options must have unique route_result_id values.")

        resource_target_pairs = [route.resource_target_key for route in route_tuple]
        if len(resource_target_pairs) != len(set(resource_target_pairs)):
            raise ValueError("route_options must not duplicate resource_id/response_target_id pairs.")

        resource_ids = {resource.resource_id for resource in resources}
        target_ids = {target.response_target_id for target in targets}
        for route in route_tuple:
            if route.resource_id not in resource_ids:
                raise ValueError(f"route option references unknown resource_id {route.resource_id!r}.")
            if route.response_target_id not in target_ids:
                raise ValueError(
                    f"route option references unknown response_target_id {route.response_target_id!r}."
                )

        target_order_by_id = {target.response_target_id: target.ordering_key for target in targets}
        return tuple(
            sorted(
                route_tuple,
                key=lambda route: (
                    resource_sort_key(route.resource_id),
                    target_order_by_id[route.response_target_id],
                    route.route_result_id,
                ),
            )
        )
