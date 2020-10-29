# Asdfy

----

**Library to process ASDFDataSet(s) from pyasdf.**

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
TBD. Check out ```tests/main.py``` reference for now. To run tests:
```
cd tests
mpi4un -n 4 python main.py
```

### ASDFProcessor
##### src
```
Union[str, Iterable[str]]
```
path to input ASDFDataSet(s)

##### dst
```
str
```
path to output ASDFDataSet

##### unc
```
Callable[..., ASDFOutput]
```
processing function

##### input_type
```
Literal['stream', 'trace', 'auxiliary'] = 'trace'
```
type of input data

##### input_tag
```
Optional[str] = None
```
input waveform tag or auxiliary group, None for using the first available

##### output_tag
```
Optional[str] = None
```
output waveform tag or auxiliary group, None for using input_tag or input_type

##### pairwise
```
bool = False
```
process input data pairwise

##### accessor
```
bool = False
```
pass the origional accessor to the processing function

##### onerror
```
Optional[Callable[[Exception], None]] = None
```
callback when error occurs
