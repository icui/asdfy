# Asdfy

----

**A library to process ASDFDataSet(s) from [pyasdf](https://github.com/seismicdata/pyasdf/).**

----

### Prerequisites
```
pyasdf
mpi4py
```

### Installation
```
pip install asdfy
```

### Usage
##### Basic usage
```
from asdfy import ASDFProcessor
ASDFProcessor('input.h5', 'output.h5', process_func).run()
```

##### Defailed explination
TBD. Refer to ```tests/main.py``` for now. To run tests:
```
cd tests
mpi4un -n 4 python main.py
```

### ASDFProcessor
##### src
```
Union[str, Iterable[str]]
```
Path to input ASDFDataSet(s)

##### dst
```
str
```
Path to output ASDFDataSet

##### func
```
Callable[..., ASDFOutput]
```
Processing function, each argument correspond to an input dataset.

##### input_type
```
Literal['stream', 'trace', 'auxiliary'] = 'trace'
```
Type of input data

##### input_tag
```
Optional[str] = None
```
Input waveform tag or auxiliary group, None for using the first available

##### output_tag
```
Optional[str] = None
```
Output waveform tag or auxiliary group, None for using input_tag or input_type

##### pairwise
```
bool = False
```
process input data pairwise

##### accessor
```
bool = False
```
Pass the origional accessor to the processing function. Set to .True. if you need event or station info.

##### onerror
```
Optional[Callable[[Exception], None]] = None
```
Callback when error occurs
