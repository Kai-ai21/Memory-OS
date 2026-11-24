"""One connector per walkable kind, behind the one `Connector` port.

`SyncSource` takes a single connector and never asks what kind of source it is
walking, which is the property that keeps the pipeline source-blind. Until M12.2
that held trivially: the container handed it a `FilesystemConnector` because
there was nothing else. With two walkable kinds the choice has to be made
somewhere, and here — keyed by `Source.kind`, which every source already
carries — it is made once, below the port, rather than by a branch in the use
case.
"""

from collections.abc import AsyncIterator, Mapping

from memoryos.application.ports import Connector, ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.values import SourceKind


class UnwalkableSource(LookupError):
    """A source of a kind no connector walks — chat and uploads, by design."""


class ConnectorsByKind(Connector):
    # The port declares a `kind`. This one answers for several, so it carries
    # the first; nothing reads it from a dispatcher.
    kind: SourceKind = SourceKind.FILESYSTEM

    def __init__(self, connectors: Mapping[SourceKind, Connector]) -> None:
        self._connectors = dict(connectors)

    @property
    def kinds(self) -> tuple[SourceKind, ...]:
        return tuple(self._connectors)

    def observe(self, source: Source, *, full: bool) -> AsyncIterator[ObservedItem]:
        connector = self._connectors.get(source.kind)
        if connector is None:
            raise UnwalkableSource(
                f"no connector walks a {source.kind.value} source ({source.name!r}); "
                "chat and upload sources are pushed, not synced"
            )
        return connector.observe(source, full=full)
