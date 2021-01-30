from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from os.path import exists
from time import sleep
from subprocess import check_call
from dataclasses import dataclass, field

from .accessor import ASDFAuxiliary

if TYPE_CHECKING:
    from obspy import Stream, Trace
    from .processor import ASDFOutput


@dataclass
class ASDFWriter:
    """Parallel ASDF writer."""
    # output ASDF file
    dst: str

    # buffer of auxiliary data to be written
    _auxiliary: Dict[str, Tuple[ASDFAuxiliary, str]] = field(init=False, default_factory=dict)

    # buffer of waveform data to be written
    _waveform: List[Tuple[Union[Stream, Trace], str]] = field(init=False, default_factory=list)

    def _write(self):
        """Write buffer content to disk."""
        from pyasdf import ASDFDataSet

        with ASDFDataSet(self.dst, mode='a', mpi=False, compression=None) as ds:
            # write waveform data
            for (waveform, tag) in self._waveform:
                ds.add_waveforms(waveform, tag)
            
            self._waveform.clear()
            
            # write auxiliary data
            for path, (data, tag) in self._auxiliary.items():
                ds.add_auxiliary_data(
                    data = data.data,
                    data_type = tag,
                    path = path,
                    parameters = data.parameters)
            
            self._auxiliary.clear()

    def add(self, data: ASDFOutput, tag: str, path: Optional[str] = None):
        """Add data to buffer."""
        from obspy import Stream, Trace

        if isinstance(data, Stream) or isinstance(data, Trace):
            self._waveform.append((data, tag))

        elif isinstance(data, ASDFAuxiliary):
            if path is None:
                raise TypeError('auxiliary data path not specified ({tag} {data})')
            
            self._auxiliary[path] = (data, tag)
    
    def write(self):
        """Write buffer content in parallel, making sure one process writes at a time."""
        from mpi4py.MPI import COMM_WORLD as comm

        myrank = comm.Get_rank()
        nranks = comm.Get_size()

        # raise error after Barrier() to ensure MPI communication
        error = None
        
        # determine the process that manages write operations
        lock_file = self.dst + '.lock'

        if not exists(lock_file):
            with open(lock_file, 'a') as f:
                f.write(str(myrank) + ' ')

        manager = None
        
        # avoid occasional OSError by trying 3 times
        for _ in range(3):
            with open(lock_file, 'r') as f:
                try:
                    manager = int(f.read().split(' ')[0])
                
                except Exception as e:
                    manager = e
                    sleep(0.1)
                
                else:
                    break
        
        if isinstance(manager, Exception):
            raise manager
        
        if manager == myrank:
            try:
                self._write()
            
            except Exception as e:
                error = e

            # current writing processes
            writing = None

            # processes finished processing and await writing
            pending: List[int] = []

            # processes finished writing
            done = 1

            while done < nranks:
                # rank from which next signal received
                src = comm.recv()

                if writing is None:
                    # tell rank to start writing
                    comm.send(src, dest=src)
                    writing = src
                
                elif src == writing:
                    # rank notifies that writing is done
                    done += 1

                    if len(pending):
                        writing = pending.pop(0)
                        comm.send(writing, dest=writing)
                    
                    else:
                        writing = None
                
                else:
                    # wait until current write process finishes
                    pending.append(src)
            
            check_call(f'rm -f {self.dst}.lock', shell=True)
        
        else:
            # request write
            comm.send(myrank, dest=manager)
            
            if comm.recv(source=manager) == myrank:
                try:
                    self._write()
                
                except Exception as e:
                    error = e

                # notify write complete
                comm.send(myrank, dest=manager)
        
        # sync processes
        comm.Barrier()

        if error:
            raise error
