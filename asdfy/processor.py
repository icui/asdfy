from __future__ import annotations

from traceback import format_exc
from sys import stderr
from os.path import dirname, exists
from subprocess import check_call
from dataclasses import dataclass, field, fields
from typing import Callable, List, Dict, Tuple, Union, Iterable, Literal, Optional, TYPE_CHECKING
from time import sleep

import numpy as np
from obspy import Stream, Trace

from .accessor import ASDFAccessor, ASDFAuxiliary

if TYPE_CHECKING:
    from pyasdf import ASDFDataSet


# type of output data
ASDFOutput = Optional[Union[Stream, Trace, ASDFAuxiliary, Tuple[np.array, dict]]]


@dataclass
class ASDFProcessor:
    """Iterates and processes data from one or multiple ASDFDataSet."""
    # path to input ASDFDataSet(s)
    src: Union[str, Iterable[str]]

    # path to output ASDFDataSet
    dst: str

    # processing function
    func: Callable[..., ASDFOutput]

    # type of input data
    input_type: Literal['stream', 'trace', 'auxiliary'] = 'trace'

    # input waveform tag or auxiliary group, None for using the first available
    input_tag: Optional[str] = None

    # output waveform tag or auxiliary group, None for using input_tag or input_type
    output_tag: Optional[str] = None

    # process input data pairwise
    pairwise: bool = False

    # pass the origional accessor to the processing function
    accessor: bool = False

    # callback when error occurs
    onerror: Optional[Callable[[Exception], None]] = None

    # current MPI rank
    _rank: int = field(init=False)

    # MPI rank size
    _size: int = field(init=False)

    def _check(self):
        """Make sure properties are correct."""
        if self.pairwise and not isinstance(self.src, str) and len(list(self.src)) > 1:
            raise ValueError('Pairwise processing is only available for single dataset')
        
        if self.input_type not in ('stream', 'trace', 'auxiliary'):
            raise ValueError('Unsupported input type', self.input_type)
    
    def _raise(self, e: Exception):
        if self.onerror:
            self.onerror(e)
        
        else:
            print(format_exc(), file=stderr)

    def _open(self):
        """Open input dataset(s)."""
        from pyasdf import ASDFDataSet

        input_ds: List[ASDFDataSet] = []

        for src in ((self.src,) if isinstance(self.src, str) else self.src):
            input_ds.append(ASDFDataSet(src, mode='r', mpi=False))
    
        return input_ds
    
    def _read(self, input_ds: List[ASDFDataSet]):
        """Copy event / station info to output dataset and get data keys to be processed."""
        from pyasdf import ASDFDataSet

        # make sure output directory is ready
        if cwd := dirname(self.dst):
            check_call(f'mkdir -p {cwd}', shell=True)

        check_call(f'rm -f {self.dst}', shell=True)
        check_call(f'rm -f {self.dst}.lock', shell=True)

        # copy event / station info
        output_ds = ASDFDataSet(self.dst, mode='w', mpi=False, compression=None)
        
        for ds in input_ds:
            added: list = []

            for event in ds.events:
                if event not in output_ds.events:
                    output_ds.add_quakeml(event)
            
            for station in ds.waveforms.list():
                if station not in added:
                    wav = ds.waveforms[station]

                    if hasattr(wav, 'StationXML'):
                        output_ds.add_stationxml(wav.StationXML)
                        added.append(station)
        
        del output_ds

        # get data keys
        keys = {}

        def add(k, t):
            if ds is input_ds[0]:
                keys[k] = [t]
            
            elif k in keys:
                keys[k].append(t)

        for ds in input_ds:
            if self.input_type == 'auxiliary':
                tag = self.input_tag or ds.auxiliary_data.list()[0]
                
                for key in ds.auxiliary_data[tag].list():
                    add(key, tag)

            else:
                for key in (key.replace('.', '_') for key in ds.waveforms.list()):
                    tags = ds.waveforms[key].get_waveform_tags()
                    
                    if len(tags) == 0 or self.input_tag in tags:
                        continue

                    tag = self.input_tag or tags[0]

                    if self.input_type == 'trace':
                        for trace in ds.waveforms[key][tag]:
                            add(key + '_MX' + trace.stats.component, tag)
                    
                    else:
                        add(key, tag)
        
        # remove incomplete entries
        for key in list(keys.keys()):
            if len(keys[key]) != len(input_ds):
                del keys[key]
        
        return keys
    
    def _process(self, input_ds: List[ASDFDataSet], keys: Dict[str, List[str]]):
        """Process data in current rank."""
        output = {}

        for i, key in enumerate(keys):
            if i % self._size == self._rank:
                accessors = []

                # get parameters for processing function
                for j, ds in enumerate(input_ds):
                    accessor = ASDFAccessor(ds, (self.input_type, keys[key][j], key))
                    accessors.append(accessor if self.accessor else accessor.target)
                
                # process data
                try:
                    output[key] = self.func(*accessors)

                    if isinstance(output[key], tuple):
                        output[key] = ASDFAuxiliary(*output[key]) # type: ignore
                
                except Exception as e:
                    self._raise(e)

        # close input dataset
        for ds in input_ds:
            del ds
        
        return output

    def _write(self, output: Dict[str, ASDFOutput]):
        """Write data to output ASDFDataSet."""
        from mpi4py.MPI import COMM_WORLD as comm
        from pyasdf import ASDFDataSet

        output_tag = self.output_tag or self.input_tag or self.input_type

        # write to output dataset
        def write():
            with ASDFDataSet(self.dst, mode='a', mpi=False, compression=None) as ds:
                """Write to output dataset."""
                for key, data in output.items():
                    if isinstance(data, Stream) or isinstance(data, Trace):
                        # write waveform data
                        ds.add_waveforms(data, output_tag)
                    
                    elif isinstance(data, ASDFAuxiliary):
                        # write auxiliary data
                        ds.add_auxiliary_data(
                            data = data.data,
                            data_type = output_tag,
                            path = key,
                            parameters = data.parameters)
            
        # determine the process that manages write operations
        lock_file = self.dst + '.lock'

        if not exists(lock_file):
            with open(lock_file, 'a') as f:
                f.write(str(self._rank) + ' ')

        source = None
        
        for _ in range(3):
            with open(lock_file, 'r') as f:
                try:
                    source = int(f.read().split(' ')[0])
                
                except Exception as e:
                    sleep(0.1)
                
                else:
                    break
        
        if source is None:
            raise IOError('unable to read lock file')
        
        if source == self._rank:
            write()

            # current writing processes
            writing = None

            # processes finished processing and await writing
            pending: List[int] = []

            # processes finished writing
            done = 1

            while done < self._size:
                rank = comm.recv()

                if writing is None:
                    comm.send(rank, dest=rank)
                    writing = rank
                
                elif rank == writing:
                    done += 1

                    if len(pending):
                        writing = pending.pop(0)
                        comm.send(writing, dest=writing)
                    
                    else:
                        writing = None
                
                else:
                    pending.append(rank)
            
            check_call(f'rm -f {self.dst}.lock', shell=True)
        
        else:
            # request write
            comm.send(self._rank, dest=source)
            
            if comm.recv(source=source) == self._rank:
                write()

                # notify write complete
                comm.send(self._rank, dest=source)
    
    def run(self, **kwargs):
        """Process and write dataset."""
        from mpi4py.MPI import COMM_WORLD as comm

        # backup current fields and update with kwargs
        backup = {}

        for f in fields(self):
            if f.name[0] != '_':
                backup[f.name] = getattr(self, f.name)
        
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._check()
        
        # get MPI info
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        
        # open input dataset(s)
        try:
            input_ds = self._open()
        
        except Exception as e:
            self._raise(e)
            input_ds = []

        # get keys to be processed
        if self._rank == 0:
            try:
                keys = self._read(input_ds)
            
            except Exception as e:
                self._raise(e)
                keys = {}
        
        else:
            keys = None
        
        keys = comm.bcast(keys, root=0)

        # process and save output
        self._write(self._process(input_ds, keys))

        # restore origional fields
        for key, val in backup.items():
            setattr(self, key, val)
        
        # sync processes
        comm.Barrier()
