# PySAMOSA

![CI](https://github.com/floschl/pysamosa/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/floschl/pysamosa/actions/workflows/release.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/pysamosa)
[![DOI](https://zenodo.org/badge/646028227.svg)](https://zenodo.org/badge/latestdoi/646028227)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

![]() <div align="center"><img src="https://github.com/floschl/pysamosa/blob/main/resources/logo_name.png?raw=true"
width="500"></div>

PySAMOSA is a Python-based software for processing open ocean and coastal waveforms from SAR satellite
altimetry to measure sea surface heights, wave heights, and wind speed for the oceans and inland waters.
Satellite altimetry is a space-borne remote sensing technique used for Earth observation. More details on satellite
altimetry can be found [here](https://www.altimetry.info/file/Radar_Altimetry_Tutorial.pdf).

The process of extracting of the three geophysical parameters from the reflected echoes/waveforms is called retracking. The measured (noisy) waveforms are fitted against the open ocean power return echo waveform model SAMOSA2 [1,2].

In the coastal zone, the return echoes are affected by spurious signals from strongly reflective targets such as sand and mud banks, tidal flats, shipping platforms, sheltered bays, or calm waters close to the shoreline.

The following European Space Agency (ESA) satellite altimetry missions are supported:
- Sentinel-3 (S3)
- Sentinel-6 Michael Freilich (S6-MF)

The software retracks the waveforms, i.e. the Level-1b (L1b) data, to extract the
retracked variables SWH, range, and Pu.

The open ocean retracker implementation specification documents from the official EUMETSAT baseline are used (S3 [1],
S6-MF [2]).

For retracking coastal waveforms the following retrackers are implemented:
- SAMOSA+ [3]
- CORAL [4,5]

In addition, FF-SAR-processed S6-MF data can be retracked using the zero-Doppler beam of the SAMOSA2 model and a
specially adapted $\alpha_p$ LUT table, created by the ESA L2 GPP project [7]. The application of the FF-SAR-processed data
has been validated in [5].

Not validated (experimental) features:
- CryoSat-2 (CS2) support
- SAMOSA++ coastal retracker [2]
- Monte-carlo SAMOSA2 simulator

## Getting-started

### Usage

Install pysamosa into your environment

    $ pip install pysamosa

This is the sample to retrack a single L1b file from the S6-MF mission

``` python
from pathlib import Path
import numpy as np

from pysamosa.common_types import L1bSourceType
from pysamosa.data_access import data_vars_s3, data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings, SettingsPreset


l1b_files = []
# l1b_files.append(Path('S6A_P4_1B_HR______20211120T051224_20211120T060836_20220430T212619_3372_038_018_009_EUM__REP_NT_F06.nc'))
l1b_files.append(Path.cwd().parent / '.data' / 's6' / 'l1b' / 'S6A_P4_1B_HR______20211120T051224_20211120T060836_20220430T212619_3372_038_018_009_EUM__REP_NT_F06.nc')

l1b_src_type = L1bSourceType.EUM_S6_F06
data_vars = data_vars_s6

# configure coastal CORAL retracker
pres = SettingsPreset.CORALv2
rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(settings_preset=pres, l1b_src_type=l1b_src_type)

rp_sets.nc_dest_dir = l1b_files[0].parent / 'processed'
rp_sets.n_offset = 0
rp_sets.n_inds = 0  #0 means all
rp_sets.n_procs = 6  #use 6 cores in parallel
rp_sets.skip_if_exists = False

additional_nc_attrs = {
    'L1B source type': l1b_src_type.value.upper(),
    'Retracker preset': pres.value.upper(),
}

rp = RetrackerProcessor(l1b_source=l1b_files, l1b_data_vars=data_vars['l1b'],
                        rp_sets=rp_sets,
                        retrack_sets=retrack_sets,
                        fitting_sets=fitting_sets,
                        wf_sets=wf_sets,
                        sensor_sets=sensor_sets,
                        nc_attrs_kw=additional_nc_attrs,
                        bbox=[np.array([-29.05, -29.00, 0, 360])],
                        )

rp.process()  #start processing

print(rp.output_l2)  #retracked L2 output can be found in here
```

A running minimal working example for retracking is shown in `notebooks/retracking_example.ipynb`.

### Development

It is highly recommended to use a proper Python IDE, such as
[PyCharm Community](https://www.jetbrainscom/pycharm/download/) or Visual Studio Code.
Using the IDE will allow you to familiarise yourself better with the code, debug and extend it.

Clone the repo

    $ git clone {repo_url}

Enter cloned directory

    $ cd pysamosa

Install dependencies into your conda env/virtualenv

    $ pip install -r requirements.txt

Compile the .pyx files (e.g. model_helpers.pyx) by running cython to build the extensions
For Windows users: An installed C/C++ compiler may be required for installation, e.g. MSCV, which comes with
the free [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)

    $ python setup.py build_ext --inplace

Optional: Compile on an HPC cluster (not normally required)

    $ LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace

## Tips

The following list provides a brief description of the recommended use of the software.
1. **Getting-started with Jupyter Notebook**
The `notebooks/retracking_example.ipynb` contains a sample script how to retrack a sample EUMETSAT baseline L1b file.
The retracked SWH and SWH data are compared with the EUMETSAT baseline L2 data. The `notebooks/demo_script.py` provides
the code example from above to quickly launch a small retracking example.

2. **More entry points**
The files `main_s3.py`, `main_s6.py`, `main_cs.py`, (`main_*.py`) etc. serve as entry points for batch processing of
   multiple nc files.
A list of L1b files (or a single file) is read for retracking, which are fully retracked or based on the given
   bounding box (bbox) paramater. A retracked L2 file is written out per processed
   L1b file.

3. **Settings**
The `RetrackerProcessor` inputs require the `RetrackerProcessorSettings`, `RetrackerSettings`, `FittingSettings`,
   `WaveformSettings`, and `SensorSettings` objects to be inserted during initialisation. The default settings of these settings objects can be retrieved with the `get_default_base_settings` function based on the three
   settings `L1bSourceType` and `SettingsPreset`.
   For instance, the following code snippet is taken from the `main_s3.py` file and retracks Sentinel-3 data with the default SAMOSA-based open ocean retracker with no SettingsPreset (100 waveforms from measurement index 25800,
   and using 6 cores).
```python
    l1b_src_type = L1bSourceType.EUM_S3
    pres = SettingsPreset.NONE  #use this for the standard SAMOSA-based retracker [2]
    # pres = SettingsPreset.CORALv2  #use this for CORALv2 [5]
    # pres = SettingsPreset.NONE  #use this for SAMOSA+ [3]
    rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(settings_preset=pres, l1b_src_type=l1b_src_type)

    rp_sets.nc_dest_dir = nc_dest_path / run_name
    rp_sets.n_offset = 25800
    rp_sets.n_inds = 100
    rp_sets.n_procs = 6
    rp_sets.skip_if_exists = False
```

4. **Evaluation environment**
There are several unit tests located in `./pysamosa/tests/` that aim to analyse the retracked output in more detail.
The most important test scripts are `test_retrack_multi.py`, which includes retracking of small along-track
segments of the S3A, S6, CS2 missions (and a generic input nc file).
`test_retrack_single` allows you to check the retracking result of a single waveform and compare it to reference
   retracking result.

<span style="color:red; font-weight:bold">Please uncomment the line `mpl.use('TkAgg')` in file `conftest.py` to
plot the test output, which is particularly useful for the retracking tests in files `tests/test_retrack_multi.
py` and `tests/test_retrack_multi.py`.</span>


5. **Difference between CORALv1 and CORALv2**
- v2 has two additional extensions that were required for S6-MF
- retrack_sets.interference_masking_mask_before_le = True
Interference signals before the leading edge are also masked out by the adaptive inteference mitigation scheme (AIM, CORAL feature)
- fitting_sets.Fit_Var_2_MinMax_Hs = (0.0, 20)
lower SWH boundary for fitting procedure is set to 0.0, as defined in [2]

## Validation

### Run tests

To run all the unit tests (using the pytest framework), run

    $ pytest

### Comparison with EUMETSAT L2 baseline data

Comparison of a retracked open ocean segment from S3 and S6-MF with the EUMETSAT L2 baseline (S3: 004, S6-MF: F06)
(generated by `notebooks/retracking_example.ipynb` Jupyter notebook)

S3 | S6-MF
:-:|:-:
![](https://github.com/floschl/pysamosa/blob/main/resources/S3_comparison_w_baseline.jpg?raw=true)  |  ![](https://github.com/floschl/pysamosa/blob/main/resources/S6_comparison_w_baseline.jpg?raw=true)

## Contributions

This software is intended to be a community-based project. Contributions to this project are very welcome.
In this case:
- Fork this repository
- Submit a pull request to be merged back into this repository.

Before submitting changes, please check that your changes pass flake8, black, isort and the
   tests, including testing other Python versions with tox:

    $ flake8 pysamosa tests scripts
    $ black . --check --diff
    $ isort . --check-only --diff
    $ pytest
    $ tox

If your pull request is accepted, you will be included in the next official release and will be listed as a
co-author for the DOI link created by Zenodo.

## Future work

Possible developments of this project are:

Retracking-related
- Align CS-2 retracking with the CS-2 baseline processing chain, validate against
[SAMpy](https://github.com/cls-obsnadir-dev/SAMPy) developed as part of the [ESA Cryo-TEMPO project](https://earth.esa.int/eogateway/documents/20142/37627/Cryo-TEMPO-ATBD-Coastal-Oceans.pdf)
- Implement evolutions of the EUMETSAT's baseline processing chain [6], e.g. the numerical retracking planned
  for Q3/2023

Software-related
- Create notebook for a coastal retracking demo
- Create richer documentation (readthedocs)

## Citation

If you use this software or the code, please cite this DOI:

Florian Schlembach; Marcello Passaro. PySAMOSA: An Open-source Software Framework for Retracking SAMOSA-based, Open
Ocean and Coastal Waveforms of SAR Satellite Altimetry. Zenodo. https://zenodo.org/badge/latestdoi/646028227.

## Acknowledgement

The authors are grateful to

Salvatore Dinardo for his support in implementing the SAMOSA-based and SAMOSA+ [3] retracking algorithms.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

## References

[1] SAMOSA Detailed Processing Model: Christine Gommenginger, Cristina Martin-Puig, Meric Srokosz, Marco Caparrini, Salvatore Dinardo, Bruno Lucas, Marco Restano, Américo, Ambrózio and Jérôme Benveniste, Detailed Processing Model of the Sentinel-3 SRAL SAR altimeter ocean waveform retracker, Version 2.5.2, 31 October 2017, Under ESA-ESRIN Contract No. 20698/07/I-LG (SAMOSA), Restricted access as defined in the Contract,  Jérôme Benveniste (Jerome.Benvensite@esa.int) pers. comm.

[2] EUMETSAT. Sentinel-6/Jason-CS ALT Level 2 Product Generation Specification (L2 ALT PGS), Version V4D; 2022.
https://www.eumetsat.int/media/48266.

[3] Dinardo, Salvatore. ‘Techniques and Applications for Satellite SAR Altimetry over Water, Land and Ice’.
Dissertation, Technische Universität, 2020. https://tuprints.ulb.tu-darmstadt.de/11343/.

[4] Schlembach, F.; Passaro, M.; Dettmering, D.; Bidlot, J.; Seitz, F. Interference-Sensitive Coastal SAR Altimetry
Retracking Strategy for Measuring Significant Wave Height. Remote Sensing of Environment 2022, 274, 112968. https://doi.org/10.1016/j.rse.2022.112968.

[5] Schlembach, F.; Ehlers, F.; Kleinherenbrink, M.; Passaro, M.; Dettmering, D.; Seitz, F.; Slobbe, C. Benefits of Fully Focused SAR Altimetry to Coastal Wave Height Estimates: A Case Study in the North Sea. Remote Sensing of Environment 2023, 289, 113517. https://doi.org/10.1016/j.rse.2023.113517.

[6] Scharroo, R.; Martin-Puig, C.; Meloni, M.; Nogueira Loddo, C.; Grant, M.; Lucas, B. Sentinel-6 Products Status. Ocean Surface Topography Science Team (OSTST) meeting in Venice 2022. https://doi.org/10.24400/527896/a03-2022.3671.

[7] ESA L2 GPP Project: FF-SAR SAMOSA LUT generation was funded under ESA contract 4000118128/16/NL/AI.
