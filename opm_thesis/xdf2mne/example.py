import os
import pyxdf
import mne
from xdf2mne.xdf2mne import streams2raw

import logging
logger = logging.getLogger(__name__)

filename = 'your_own.xdf'
filepath = os.path.join('/', filename)

streams, fileheader = pyxdf.load_xdf(filepath, dejitter_timestamps=True)

##%
stream = streams[0]
marker_stream = streams[1]

raw, events, event_id = streams2raw(stream, marker_stream=[marker_stream])

print(f"raw contains the following annotations: {raw.annotations}")
# The events array is redundant with annotations, you can do
events_from_annotations = mne.annotations.events_from_annotations(raw, event_id)
# and should receive the same array as events
