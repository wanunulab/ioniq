Signal Preprocessing
=====================

**Segmentation System**

**Signal Manipulation**

Pre-processing prepares raw signals for analysis by:

- Filtering to reduce high-frequency noise (e.g., butter, bandstop filter)
- Downsampling to improve computational performance
- Compressing voltage data for memory efficiency
- Trimming

These operations are performed using tools in ``utils.py`` and parser modules