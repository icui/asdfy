from __future__ import annotations

from sys import stderr
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from os.path import exists
from time import sleep
from subprocess import check_call
from dataclasses import dataclass, field

from .accessor import ASDFAuxiliary

if TYPE_CHECKING:
    from obspy import Stream, Trace, Inventory
    from .processor import ASDFOutput


@dataclass
class ASDFWriter:
    """Parallel ASDF writer."""
    # output ASDF file
    dst: str

    # MPI comm
    comm: Any = None

    # buffer of auxiliary data
    _auxiliary: Dict[str, Tuple[ASDFAuxiliary, str]] = field(init=False, default_factory=dict)

    # buffer of waveform data
    _waveform: List[Tuple[Union[Stream, Trace], str]] = field(init=False, default_factory=list)

    # buffer of StationXML data
    _inventory: Dict[str, Inventory] = field(init=False, default_factory=dict)

    def _write(self):
        """Write buffer content to disk."""
        if self.dst.endswith('.h5'):
            self._write_h5()
        
        elif self.dst.endswith('.json'):
            self._write_json()

    def _write_json(self):
        import json

        if exists(self.dst):
            with open(self.dst, 'r') as f:
                output = json.load(f)

        else:
            output = {}

        # write auxiliary data
        for path, (data, _) in self._auxiliary.items():
            output[path] = data.parameters
        
        with open(self.dst, 'w') as f:
            json.dump(output, f)

    def _write_h5(self):
        from pyasdf import ASDFDataSet

        with ASDFDataSet(self.dst, mode='a', mpi=False, compression=None) as ds:
            # write waveform data
            for (waveform, tag) in self._waveform:
                try:
                    ds.add_waveforms(waveform, tag)
                
                except Exception as e:
                    print(waveform, file=stderr)
                    raise e
            
            self._waveform.clear()
            
            # write auxiliary data
            for path, (data, tag) in self._auxiliary.items():
                try:
                    ds.add_auxiliary_data(
                        data = data.data,
                        data_type = tag,
                        path = path,
                        parameters = data.parameters)
                
                except Exception as e:
                    print(data, path, file=stderr)
                    raise e
            
            # write station data
            for data in self._inventory.values():
                try:
                    ds.add_stationxml(data)
                
                except Exception as e:
                    print(data, file=stderr)
            
            self._auxiliary.clear()

    def _get_comm(self):
        """Get MPI comm."""
        if self.comm is None:
            from mpi4py.MPI import COMM_WORLD as comm
            return comm

        return self.comm

    def add(self, data: Union[ASDFOutput, Inventory], tag: str, path: Optional[str] = None):
        """Add data to buffer."""
        from obspy import Stream, Trace, Inventory

        if isinstance(data, Stream) or isinstance(data, Trace):
            self._waveform.append((data, tag))

        elif isinstance(data, ASDFAuxiliary):
            if path is None:
                raise TypeError('auxiliary data path not specified ({tag} {data})')
            
            self._auxiliary[path] = (data, tag)

        elif isinstance(data, Inventory):
            self._inventory[tag] = data
    
    def write(self):
        """Write buffer content in parallel, making sure one process writes at a time."""
        comm = self._get_comm()

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
