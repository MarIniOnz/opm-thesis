# xdf2mne
A small tool to transform time series data from a .xdf-file into mne's RawArray data structure.
To find out how to use, take a look at [example.py](example.py) and the comments in [xdf2mne.py](xdf2mne/xdf2mne.py).

Requirements as to sampling rate steadiness and other things are also pointed out in the code comments. 

In short:
```python
import pyxdf
from xdf2mne import stream2raw


streams, fileheader = pyxdf.load_xdf('your_xdf_filepath', dejitter_timestamps=True)
raw, events, event_id = stream2raw(streams[0], marker_stream=streams[1])
```

Enjoy!

## Requirements
- [pyxdf](https://pypi.org/project/pyxdf/) (v1.16.3)
- [mne](https://mne.tools/stable/index.html) (v0.22.0)
- [numpy](https://numpy.org/) (v1.18.2)

I cannot promise that this tool will work with older or newer versions of these packages.
In fact, I won't even promise it will work at all, but for me it does what I need it to do.
