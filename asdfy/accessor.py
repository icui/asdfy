from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, Union, TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Stream, Trace, Inventory
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
    key: Tuple[Literal['stream', 'trace', 'auxiliary'], str, str]

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
        """Auxiliary data group."""
        if self.key[0] == 'auxiliary':
            group = self.ds.auxiliary_data[self.key[1]][self.key[2]]
            return ASDFAuxiliary(np.array(group.data), dict(group.parameters))

    @property
    def stream(self) -> Optional[Stream]:
        """Obspy Stream object."""
        if self.key[0] != 'auxiliary':
            return self.ds.waveforms[self.station][self.key[1]]

    @property
    def trace(self) -> Optional[Trace]:
        """Obspy Trace object."""
        if self.key[0] == 'trace':
            return self.stream.select(component=self.component)[0] # type: ignore
    
    @property
    def station(self) -> Optional[str]:
        """Station name."""
        if self.key[0] != 'auxiliary':
            return '.'.join(self.key[2].split('_')[:2])
    
    @property
    def component(self) -> Optional[str]:
        """Trace component."""
        if self.key[0] == 'trace':
            return self.key[2].split('_')[-1][-1]

    @property
    def inventory(self) -> Optional[Inventory]:
        """Attached station data."""
        if self.key[0] != 'auxiliary':
            if hasattr(waveform := self.ds.waveforms[self.station], 'StationXML'):
                return getattr(waveform, 'StationXML')
    
    @property
    def target(self) -> Union[Stream, Trace, ASDFAuxiliary]:
        """Target object that self.key points to."""
        return getattr(self, self.key[0])
