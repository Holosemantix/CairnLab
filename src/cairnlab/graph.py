from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .models import Relation, RelationType
from .utils import enum_value


AUTHORITY_RELATION_TYPES = {
    RelationType.APPROVED_BY.value,
    RelationType.RELEASED_BY.value,
    RelationType.VERIFIED_BY.value,
}


@dataclass(frozen=True)
class GraphStep:
    object_id: str
    relation_path: tuple[str, ...]


class RelationGraph:
    """Dependency graph over imported CairnLab relations.

    The graph is intentionally independent of storage. Host projects can build
    it directly from their own exported relation list.
    """

    def __init__(self, relations: list[Relation]):
        self.relations = relations
        self.forward: dict[str, list[Relation]] = defaultdict(list)
        self.reverse: dict[str, list[Relation]] = defaultdict(list)
        for relation in relations:
            self.forward[relation.source].append(relation)
            self.reverse[relation.target].append(relation)

    def incoming(self, object_id: str) -> list[Relation]:
        return list(self.reverse.get(object_id, []))

    def outgoing(self, object_id: str) -> list[Relation]:
        return list(self.forward.get(object_id, []))

    def downstream_of(self, target_id: str) -> list[GraphStep]:
        visited = {target_id}
        steps: list[GraphStep] = []
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(target_id, tuple())])

        while queue:
            current, path = queue.popleft()
            for relation, next_id in self._propagation_edges(current):
                if next_id in visited:
                    continue
                visited.add(next_id)
                next_path = path + (relation.id,)
                steps.append(GraphStep(object_id=next_id, relation_path=next_path))
                queue.append((next_id, next_path))
        return steps

    def _propagation_edges(self, object_id: str) -> list[tuple[Relation, str]]:
        edges: list[tuple[Relation, str]] = []
        for relation in self.forward.get(object_id, []):
            edges.append((relation, relation.target))
        for relation in self.reverse.get(object_id, []):
            if enum_value(relation.type) in AUTHORITY_RELATION_TYPES:
                edges.append((relation, relation.source))
        return edges
