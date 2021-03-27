from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, Union, Dict, TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Stream, Trace, Inventory
    from obspy.core.event import Origin
    from obspy.core.trace import Stats


# input argument for auxiliary data
@dataclass
class ASDFAuxiliary:
    """Auxiliary data and parameters in ASDFDataSet"""
    # data array
    data: np.ndarray

    # data parameters
    parameters: dict = field(default_factory=dict)


@dataclass
class ASDFAccessor:
    """A combination of an opened ASDFDataSet and a path.
        Points to waveform or auxiliary data.
    """
    # target dataset
    ds: ASDFDataSet

    # path to the data
    # [0]: data type
    # [1]: waveform tag for stream / trace, group for auxiliary
    # [2]: station for stream, station + component for trace, data path for auxiliary
    key: Tuple[Literal['stream', 'trace', 'auxiliary', 'auxiliary_group'], str, str]

    # all accessors in self.ds (for pairwise processing)
    fellows: Optional[list[ASDFAccessor]] = None

    @property
    def data(self) -> Optional[np.ndarray]:
        """Raw data as numpy array."""
        if self.key[0] == 'auxiliary':
            return self.auxiliary.data
        
        if self.key[0] == 'trace':
            return cast(np.ndarray, self.trace.data)
    
    @property
    def stats(self) -> Optional[Stats]:
        """Stats for waveform data"""
        if self.key[0] == 'trace':
            return self.trace.stats
        
    @property
    def parameters(self) -> Optional[dict]:
        """Parameters for auxiliary data"""
        if self.key[0] == 'auxiliary':
            return self.auxiliary.parameters
    
    @property
    def auxiliary(self) -> Optional[ASDFAuxiliary]:
        """Auxiliary data."""
        if self.key[0] == 'auxiliary':
            group = self.ds.auxiliary_data[self.key[1]][self.key[2]]
            return ASDFAuxiliary(np.array(group.data), dict(group.parameters))
    
    @property
    def auxiliary_group(self) -> Optional[Dict[str, ASDFAuxiliary]]:
        """Group of auxiliary data."""
        if self.key[0] == 'auxiliary_group':
            groups = {}
            auxiliaries = {}
            n = len(self.key[2]) + 1
            ds = self.ds.auxiliary_data[self.key[1]]

            for key in ds.list():
                if key.startswith(self.key[2] + '_'):
                    groups[key[n:]] = ds[key]
                
            for key, group in groups.items():
                auxiliaries[key] = ASDFAuxiliary(np.array(group.data), dict(group.parameters))

            return auxiliaries

    @property
    def stream(self) -> Optional[Stream]:
        """Obspy Stream object."""
        if self.key[0] in ('trace', 'stream'):
            return self.ds.waveforms[self.station][self.key[1]]

    @property
    def trace(self) -> Optional[Trace]:
        """Obspy Trace object."""
        if self.key[0] == 'trace':
            return self.stream.select(component=self.component)[0] # type: ignore
    
    @property
    def event(self) -> Optional[str]:
        """Event name."""
        if len(self.ds.events) > 0:
            for d in self.ds.events[0].event_descriptions:
                if d.type == 'earthquake name':
                    return d.text
    
    @property
    def station(self) -> Optional[str]:
        """Station name."""
        if len(path := self.key[2].split('_')) >= 2:
            return '.'.join(path[:2])
    
    @property
    def channel(self) -> Optional[str]:
        """Trace channel."""
        if self.key[0] == 'trace':
            return self.key[2].split('_')[-1]

        if self.key[2] == 'auxiliary' and len(path := self.key[2].split('_')) >= 3:
            return path[-1]
    
    @property
    def component(self) -> Optional[str]:
        """Trace component."""
        if channel := self.channel:
            return channel[-1]

    @property
    def inventory(self) -> Optional[Inventory]:
        """Attached station data."""
        if station := self.station:
            if hasattr(waveform := self.ds.waveforms[station], 'StationXML'):
                return getattr(waveform, 'StationXML')
    
    @property
    def origin(self) -> Optional[Origin]:
        """Event origin."""
        if len(self.ds.events):
            return self.ds.events[0].preferred_origin()
    
    @property
    def target(self) -> Union[Stream, Trace, ASDFAuxiliary]:
        """Target object that self.key points to."""
        return getattr(self, self.key[0])
