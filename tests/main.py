from __future__ import annotations
from typing import TYPE_CHECKING
from os import chdir, path

from asdfy import ASDFProcessor, ASDFAccessor

if TYPE_CHECKING:
    from obspy import Trace, Stream


if not path.exists('traces.h5') and path.exists('tests/traces.h5'):
    chdir('tests')


def func1(stream: Stream):
    # save waveform by returning a Stream
    return stream


def func2(acc: ASDFAccessor):
    # save waveform by returning a Stream
    assert acc.fellows and len(acc.fellows) == 9, f'incorrect station number'

    for acc2 in acc.fellows:
        assert acc2.component == acc.component
        assert acc2.ds is acc.ds
    
    output = {}

    for trace in acc.stream:
        output[trace.stats.channel] = trace
    
    return output


def func3(trace: Trace):
    trace.filter('lowpass', freq=1/17)
    # save waveform by returning a Trace
    return trace


def func4(syn_acc, obs_acc):
    syn = syn_acc.trace
    obs = obs_acc.trace
    data = syn.data - obs.data # type: ignore
    stats = syn.stats

    assert len(syn_acc.fellows) == 27, f'incorrect station number {len(syn_acc.fellows)}'
    assert len(obs_acc.fellows) == 27, f'incorrect station number {len(obs_acc.fellows)}'

    for acc in syn_acc.fellows:
        assert acc.ds is syn_acc.ds
    
    for acc in obs_acc.fellows:
        assert acc.ds is obs_acc.ds

    # save as auxiliary data by returning a tuple
    return data, {
        'misfit': data.std(),
        'network': stats.network,
        'station': stats.station,
        'component': stats.component}


def func5(acc):
    from asdfy import ASDFAuxiliary

    # save as auxiliary data by returning namedtuple `ASDFAuxiliary`
    return ASDFAuxiliary(acc.data, acc.auxiliary.parameters)


def func6(aux_group):
    from obspy import Trace, Stream

    # save waveform by returning a Trace
    traces = []

    for cha, aux in aux_group.items():
        assert cha[-1] == aux.parameters['component']
        traces.append(Trace(aux.data, header=aux.parameters))
    
    return Stream(traces)

def reset():
    from subprocess import check_call

    check_call('rm -f proc1.h5', shell=True)
    check_call('rm -f proc2.h5', shell=True)
    check_call('rm -f proc3.h5', shell=True)
    check_call('rm -f proc4.h5', shell=True)
    check_call('rm -f proc5.h5', shell=True)
    check_call('rm -f proc6.h5', shell=True)


def verify():
    from numpy.linalg import norm
    from pyasdf import ASDFDataSet

    with ASDFDataSet('proc1.h5', mode='r', mpi=False) as ds:
        assert len(ds.events) == 1
        assert hasattr(ds.waveforms['II.BFO'], 'StationXML')

    with ASDFDataSet('proc6.h5', mode='r', mpi=False) as ds:
        data_proc = ds.waveforms['II.BFO'].test[0].data # type: ignore
    
    with ASDFDataSet('traces_proc.h5', mode='r', mpi=False) as ds:
        data_ref = ds.waveforms['II.BFO'].test[0].data # type: ignore
    
    assert norm(data_proc - data_ref) / norm(data_ref) < 1e-4

    print('pass')
    reset()


def verify_mpi():
    from mpi4py.MPI import COMM_WORLD as comm

    rank = comm.Get_rank()

    if rank == 0:
        verify()


def test():
    from mpi4py.MPI import COMM_WORLD as comm

    rank = comm.Get_rank()

    if rank == 0:
        reset()

    # process stream data
    ap = ASDFProcessor('traces.h5', 'proc1.h5', func1, input_type='stream', input_tag='synthetic')

    if rank == 0:
        print('test1: stream -> stream')
        assert len(ap.access()) == 9
    
    ap.run()

    # process stream data with more info passed
    if rank == 0:
        print('test2: accessor -> stream')
    
    ASDFProcessor('traces.h5', 'proc2.h5', func2, input_type='stream', accessor=True).run()

    # process trace data
    if rank == 0:
        print('test3: trace -> trace')
    
    ASDFProcessor('proc2.h5', 'proc3.h5', func3).run()

    # process trace data (save with a different tag)
    if rank == 0:
        print('test4: (trace, trace) -> auxiliary')
    
    ASDFProcessor(('proc1.h5', 'proc3.h5'), 'proc4.h5', func4, accessor=True, output_tag='test').run()

    # process auxiliary data with more info passed
    if rank == 0:
        print('test5: accessor -> auxiliary')
    
    ASDFProcessor('proc4.h5', 'proc5.h5', func5, input_type='auxiliary', accessor=True, input_tag='test').run()

    # process auxiliary data
    if rank == 0:
        print('test6: auxiliary_group -> stream')
    
    ASDFProcessor('proc5.h5', 'proc6.h5', func6, input_type='auxiliary_group').run()

    if rank == 0:
        verify()

if __name__ == '__main__':
    test()
