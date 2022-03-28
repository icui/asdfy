from __future__ import annotations

from traceback import format_exc
from sys import stderr
from os.path import dirname
from subprocess import check_call
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Union, Iterable, Literal, Optional, TYPE_CHECKING


from .accessor import ASDFAccessor, ASDFAuxiliary
from .writer import ASDFWriter

if TYPE_CHECKING:
    import numpy as np
    from pyasdf import ASDFDataSet
    from obspy import Stream, Trace


# type of input data
ASDFInput = Literal['stream', 'trace', 'auxiliary', 'auxiliary_group']


# type of output data
ASDFOutput = Optional[Union['Stream', 'Trace', ASDFAuxiliary, Tuple['np.ndarray', dict]]]


# type of processing function
ASDFFunction = Callable[..., Union[ASDFOutput, Dict[str, ASDFOutput]]]


@dataclass
class ASDFProcessor:
    """Iterates and processes data from one or multiple ASDFDataSet."""
    # path to input ASDFDataSet(s)
    src: Union[str, Iterable[str]]

    # path to output ASDFDataSet
    dst: Optional[str] = None

    # processing function
    func: Optional[ASDFFunction] = None

    # type of input data
    input_type: Union[ASDFInput, List[ASDFInput]] = 'trace'

    # input waveform tag or auxiliary group, None for using the first available
    input_tag: Optional[str] = None

    # output waveform tag or auxiliary group, None for using input_tag or input_type
    output_tag: Optional[str] = None

    # pass the origional accessor to the processing function
    accessor: bool = False

    # process input data pairwise
    pairwise: bool = False

    # callback when error occurs
    onerror: Union[Callable[[Exception], None], None, Literal['raise']] = None

    def _input_type(self, j: int) -> ASDFInput:
        """List of input data types."""
        if isinstance(self.input_type, list):
            return self.input_type[j]
        
        return self.input_type

    def _check(self):
        """Make sure properties are correct."""
        if self.pairwise and not self.accessor:
            raise ValueError('accessor must be True to enable pairwise processing')
        
        for j in range(1 if isinstance(self.src, str) else len(list(self.src))):
            if self._input_type(j) not in ('stream', 'trace', 'auxiliary', 'auxiliary_group'):
                raise ValueError('unsupported input type', self.input_type)
    
    def _raise(self, e: Exception):
        if self.onerror == 'raise':
            raise e

        elif self.onerror:
            self.onerror(e)
        
        else:
            print(format_exc(), file=stderr)

    def _open(self):
        """Open input dataset(s)."""
        from pyasdf import ASDFDataSet

        opened = {}
        input_ds: List[ASDFDataSet] = []

        for src in ((self.src,) if isinstance(self.src, str) else self.src):
            if src not in opened:
                opened[src] = ASDFDataSet(src, mode='r', mpi=False)

            input_ds.append(opened[src])
    
        return input_ds
    
    def _copy_meta(self, input_ds: List[ASDFDataSet]):
        """Copy event / station info to output dataset."""
        from pyasdf import ASDFDataSet
        
        if self.dst is None:
            return

        # make sure output directory is ready
        if cwd := dirname(self.dst):
            check_call(f'mkdir -p {cwd}', shell=True)

        check_call(f'rm -f {self.dst}', shell=True)
        check_call(f'rm -f {self.dst}.lock', shell=True)

        # copy event info
        output_ds = ASDFDataSet(self.dst, mode='w', mpi=False, compression=None)
        
        for ds in input_ds:
            for event in ds.events:
                if event not in output_ds.events:
                    output_ds.add_quakeml(event)
        
        del output_ds

    def _get_keys(self, input_ds: List[ASDFDataSet]):
        """Get paths to the data to be processed."""
        keys = {}

        def add(k, t):
            if ds is input_ds[0]:
                keys[k] = [t]
            
            elif k in keys:
                keys[k].append(t)

        for j, ds in enumerate(input_ds):
            if self._input_type(j) == 'auxiliary':
                # add auxiliary data
                tag = self.input_tag or ds.auxiliary_data.list()[0]
                
                for key in ds.auxiliary_data[tag].list():
                    add(key, tag)
            
            elif self._input_type(j) == 'auxiliary_group':
                # add auxiliary data group
                tag = self.input_tag or ds.auxiliary_data.list()[0]
                groups = set()

                for key in ds.auxiliary_data[tag].list():
                    keypath = key.split('_')

                    if len(keypath := key.split('_')) > 1:
                        groups.add('_'.join(keypath[:-1]))
                    
                for key in groups:
                    add(key, tag)

            else:
                # add waveform data
                for key in (key.replace('.', '_') for key in ds.waveforms.list()):
                    tags = ds.waveforms[key].get_waveform_tags()
                    
                    if len(tags) == 0 or (self.input_tag and self.input_tag not in tags):
                        # no available waveform tag
                        continue

                    tag = self.input_tag or tags[0]

                    if self._input_type(j) == 'trace':
                        for trace in ds.waveforms[key][tag]:
                            add(key + '_' + trace.stats.component, tag)
                    
                    else:
                        add(key, tag)
        
        # remove incomplete entries
        for key in list(keys.keys()):
            if len(keys[key]) != len(input_ds):
                del keys[key]
        
        return keys

    def _get_accessors(self, input_ds: List[ASDFDataSet], keys: Dict[str, List[str]]):
        """Create accessors from input_ds and keys."""
        accessors: Dict[str, List[ASDFAccessor]] = {}

        for key in keys:
            accessors[key] = []

            for j, ds in enumerate(input_ds):
                accessors[key].append(ASDFAccessor(ds, (self._input_type(j), keys[key][j], key)))
        
        return accessors
    
    def _process(self, input_ds: List[ASDFDataSet], keys: Dict[str, List[str]], writer: Optional[ASDFWriter] = None):
        """Process data in current rank."""
        from mpi4py.MPI import COMM_WORLD as comm
        
        if self.func is None:
            return

        myrank = comm.Get_rank()
        nranks = comm.Get_size()

        # all accessors in input_ds
        accessors = self._get_accessors(input_ds, keys)

        # add accessor.fellows
        if self.accessor:
            for j in range(len(input_ds)):
                fellows: List[ASDFAccessor] = []

                for key in keys:
                    fellows.append(accessors[key][j])
            
                for key in keys:
                    accessors[key][j].fellows = fellows

        for i, key in enumerate(keys):
            if i % nranks == myrank:
                args = []
                inventory = None
                station = None

                # output tag
                output_tag = self.output_tag or self.input_tag or keys[key][0]

                # get parameters for processing function
                for j, ds in enumerate(input_ds):
                    accessor = accessors[key][j]
                    args.append(accessor if self.accessor else accessor.target)

                    if inventory is None or station is None:
                        inventory = accessor.inventory
                        station = accessor.station
                
                # process data
                try:
                    result = self.func(*args)

                    if writer and result is not None:
                        if isinstance(result, dict):
                            for cmp, val in result.items():
                                if val is None:
                                    continue

                                if isinstance(val, tuple):
                                    val = ASDFAuxiliary(*val)

                                writer.add(val, output_tag, key + '_' + cmp)
                        
                        else:
                            if isinstance(result, tuple):
                                result = ASDFAuxiliary(*result)
                            
                            writer.add(result, output_tag, key)
                    
                        if inventory and station:
                            writer.add(inventory, station)
                
                except Exception as e:
                    self._raise(e)

        # close input dataset
        for ds in input_ds:
            ds.__del__()
    
    def run(self):
        """Process and write dataset."""
        from mpi4py.MPI import COMM_WORLD as comm

        self._check()
        
        # open input dataset(s)
        try:
            input_ds = self._open()
        
        except Exception as e:
            self._raise(e)
            input_ds = []

        # get keys to be processed
        if comm.Get_rank() == 0:
            try:
                self._copy_meta(input_ds)
                keys = self._get_keys(input_ds)
            
            except Exception as e:
                self._raise(e)
                keys = {}
        
        else:
            keys = None
        
        keys = comm.bcast(keys, root=0)

        # process and save output
        try:
            if self.dst:
                self._process(input_ds, keys, writer := ASDFWriter(self.dst))
                writer.write()
            
            else:
                self._process(input_ds, keys)
        
        except Exception as e:
            self._raise(e)
    
    def access(self):
        """Get all accessors."""
        self._check()

        input_ds = self._open()
        keys = self._get_keys(input_ds)

        return self._get_accessors(input_ds, keys)
