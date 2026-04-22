<div align="center" style="background-color: white;">
<a href="https://www.transportforthenorth.com/">
<img src="https://www.transportforthenorth.com/wp-content/themes/tfn-theme/img/logo.svg"
  alt="Transport for the North logo">
</a>
</div>

<h1 align="center">CAF.cvt</h1>

<p align="center">
<a href="https://transport-for-the-north.github.io/CAF-Handbook/python_tools/framework.html">
  <img alt="CAF Status - Beta" src="https://img.shields.io/badge/CAF%20Status-Beta-yellow">
</a>
</p>
<p align="center">
<a href="https://pypi.org/project/caf.cvt/">
  <img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/caf.cvt.svg?style=flat-square">
</a>
<a href="https://pypi.org/project/caf.cvt/">
  <img alt="Latest release" src="https://img.shields.io/github/release/transport-for-the-north/caf.cvt.svg?style=flat-square&maxAge=86400">
</a>
<a href="https://anaconda.org/conda-forge/caf.cvt">
  <img alt="Conda" src="https://img.shields.io/conda/v/conda-forge/caf.cvt?style=flat-square&logo=condaforge">
</a>
</p>
<p align="center">
<a href="https://github.com/transport-for-the-north/caf.cvt.template/actions?query=event%3Apush">
  <img alt="Testing Badge" src="https://img.shields.io/github/actions/workflow/status/transport-for-the-north/caf.cvt/tests.yml?style=flat-square&logo=GitHub&label=Tests">
</a>
<a href="https://app.codecov.io/gh/transport-for-the-north/caf.cvt">
  <img alt="Coverage" src="https://img.shields.io/codecov/c/github/transport-for-the-north/caf.cvt.svg?branch=main&style=flat-square&logo=CodeCov">
</a>
<a href='https://cafcvt.readthedocs.io/en/stable/'>
  <img alt='Documentation Status' src="https://img.shields.io/readthedocs/cafcvt?style=flat-square&logo=readthedocs">
</a>
</p>


CAF.cvt is a python model that takes climate and transport data and generates a climate risk assessment for the North's key transport infrastructure. 

> [!TIP]
> For more detailed information including a user guide, tutorials and API reference see the full
> [caf.cvt documentation](https://cafcvt.readthedocs.io/en/stable/)


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [What does it do?](#what-does-it-do)
  - [Main Features](#main-features)
    - [Work-in-Progress](#work-in-progress)
  - [Who is it for?](#who-is-it-for)
- [Where to get it](#where-to-get-it)
  - [Installation from GitHub](#installation-from-github)
- [Usage](#usage)
  - [Command Line](#command-line)
- [Documentation](#documentation)
- [What is CAF?](#what-is-caf)
- [Contribution](#contribution)
- [Contact Us](#contact-us)
- [Template Usage](#template-usage)

## Overview

### What does it do?

The Common Analytical Framework (CAF) Climate Vulnerability Tool (CVT) is a geospatial python package that translates complex climate and infrastructure data into a set of climate risk assessments for the North of England's key transport networks. 

### Main Features

- **Data Cleaning** - Reads and cleans a wide variety of climate and transport datasets, preprocessing into a standard format in preparation for analysis.

- **Apply Functional Rules** - Applies a set of funtional rules and spatial overlays to the climate data to translate complex values into distinct hazard categories and grids. 

- **Infrastructure Layering** - Layers infrastructure on top of hazard grids to assign risk to the transport assets.

#### Work-in-Progress

There are currently no work-in-progress features.

> [!WARNING]
> These features are work-in-progress and are not available in a released version of caf.cvt, to
> access these features a specific branch of caf.cvt should be installed, see [Installation from GitHub](#installation-from-github).

### Who is it for?

- **Target audience:** Transport Analysts, Transport Planners, Climate Researchers
- **CAF Analytical Stage:** Modelling

![CAF Analytical Process Diagram](https://github.com/Transport-for-the-North/.github/blob/21a428e81880639839e221940881572cdee24d5a/profile/ProcessDiagram.png?raw=true)

For more details on CAF Analytical Stages see the [description within TfN's GitHub homepage](https://github.com/Transport-for-the-North)

## Where to get it

> [!IMPORTANT]
> caf.cvt has not been published yet so cannot be installed from
> conda-forge or PyPI, see [Installation from GitHub](#installation-from-github).

The latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/caf.cvt) and on [Conda](https://anaconda.org/conda-forge/caf.cvt).

```sh
conda install -c conda-forge caf.cvt
```

```sh
pip install caf.cvt
```

> [!TIP]
>
> - See the [Quick Start Guide](https://cafcvt.readthedocs.io/en/stable/start.html#quick-start) for more detailed instructions.
> - See the [requirements.txt](requirements.txt) for the full list of package dependencies.

### Installation from GitHub

> [!WARNING]
> Unreleased GitHub versions should **not** be considered stable.

The latest, unreleased, version can be installed directly from GitHub using:

```sh
pip install "git+https://github.com/transport-for-the-north/caf.cvt.template"
```

> [!TIP]
> `pip install` can install a specific tag, or branch, using `@{tag-name}`
> after the git URL.

## Usage

CAF.cvt provides and Command-line (CLI) and graphical interface (GUI) to use many of it's
features without the need to write any Python code, see the [Tool Usage section](https://cafcvt.readthedocs.io/en/stable/usage/index.html)
of the user guide for more details.

### Command Line

The tool can be run from command line, with the command:

```sh
caf.cvt
```

See [Command-Line Interface (User Guide)](https://cafcvt.readthedocs.io/en/stable/usage/cli.html)
for full explanations of the parameters.

## Documentation

The code documentation is hosted at <https://cafcvt.readthedocs.io/en/stable/>.


## What is CAF?

This tool is part of TfN's [Common Analytical Framework (CAF)](https://github.com/Transport-for-the-North).
CAF is Transport for the North's structured suite of analytical tools designed to support transport
modelling, appraisal, and strategic decision-making.

More information on CAF and details on other CAF tools can be found on [TfN's GitHub Homepage](https://github.com/Transport-for-the-North).

## Contribution

We encourage use of, and contributions to, the repositories within this organisation, licenses are provided within
the repositories and the [organisation contribution guide](https://github.com/Transport-for-the-North/.github/blob/main/CONTRIBUTING.rst)
provides details for contributions.

---

## Contact Us

For further information about using this tool or CAF tools in your projects and work contact Transport for the North - <TfNOffer@transportforthenorth.com>

---


[Go to Top](#table-of-contents)
