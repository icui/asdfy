from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyasdf import ASDFDataSet


# input argument for auxiliary data
ASDFAuxiliary = namedtuple('ASDFAuxiliary', ['data', 'parameters'])


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
    def data(self):
        """Raw data as numpy array."""
        if self.key[0] == 'auxiliary':
            return self.auxiliary.data
        
        if self.key[0] == 'trace':
            return self.trace.data
        
        return None
    
    @property
    def stats(self):
        """Stats for waveform data"""
        if self.key[0] == 'trace':
            return self.trace.stats
        
        return {}

    @property
    def parameters(self):
        """Parameters for auxiliary data"""
        if self.key[0] == 'auxiliary':
            return self.auxiliary.parameters
        
        return {}
    
    @property
    def auxiliary(self):
        """Auxiliary data group."""
        if self.key[0] == 'auxiliary':
            group = self.ds.auxiliary_data[self.key[1]][self.key[2]]
            return ASDFAuxiliary(np.array(group.data), dict(group.parameters))
        
        return None
    
    @property
    def waveform(self):
        """ASDF waveform accessor."""
        if self.key[0] == 'auxiliary':
            return None
        
        if self.key[0] == 'trace':
            return self.ds.waveforms['_'.join(self.key[2].split('_')[:2])]
        
        else:
            return self.ds.waveforms[self.key[2]]

    @property
    def stream(self):
        """Stream data (None if path[0] == auxiliary)."""
        if self.key[0] != 'auxiliary':
            return self.waveform[self.key[1]]
        
        return None

    @property
    def trace(self):
        """Trace data (None if path[0] == auxiliary)."""
        if self.key[0] == 'trace':
            return self.stream.select(component=self.key[2].split('_')[-1][-1])[0]
        
        return None

    @property
    def catalog(self):
        """Attached event data."""
        return self.ds.events

    @property
    def inventory(self):
        """Attached station data."""
        if wav := self.waveform:
            if hasattr(wav, 'StationXML'):
                return wav.StationXML
        
        return None
    
    @property
    def target(self):
        """Target object that self.key points to."""
        return getattr(self, self.key[0])
