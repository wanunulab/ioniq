.. ioniq documentation master file, created by
   sphinx-quickstart on Tue Feb 11 10:00:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ioniq: A Nanopore Signal Analysis Framework
==========================================

**Authors:**
Ali Fallahi

**Date:**
Feb 11, 2025

**Version:**
0.1.0 (Development)

Overview
--------

Ioniq is a Python module for processing and analyzing nanopore signal data. It provides tools for reading, filtering, segmenting, and analyzing nanopore experimental data. The framework supports structured workflows for signal processing and quality control, allowing researchers to extract relevant information from raw current and voltage traces.

Features
--------

- **File Handling**
  - Supports `.edh`, `.opt`, and `.xml` formats.
  - Extracts raw current and voltage traces.
  - Parses experimental metadata for structured analysis.

- **Signal Processing**
  - Segmentation and event detection.


- **Analysis Modules**
  - IV curve computation and voltage pattern detection.
  - Step response and dwell-time analysis.
  - Customizable parsers and filters for flexible workflows.

- **Usability**
  - Python library for scripting and automation.


Installation (Development)
--------------------------

To install the development version of Ioniq, clone the repository and install it in editable mode:

.. code-block:: bash

    git clone https://github.com/wanunulab/ioniq.git
    cd ioniq
    pip install ioniq

**Requirements:**
- Python > 3.10
- Dependencies listed in `requirements.txt`

Future Development
------------------

- **Additional analysis modules** for extended event detection and pattern recognition.
- **Integration with Jupyter Notebooks** for interactive visualization.
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules
   install_beginner.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
