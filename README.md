This is a repository of examples for how to convert events from icetray's `.i3` file format to numpy's `.npy` format for use in IceCube's NuSources tools. 

## Format for output files

### Data
The format we need to use is set by the [NuSources Data Format guide](https://wiki.icecube.wisc.edu/images/b/b6/Nu-sources-data-format.pdf), which provides the minimum requirements. For all files, we need to include

- run: The run number that we pulled the event from. Typically from either the `I3EventHeader` (data) or from the file name (MC)
- event: The event ID number within the run from the `I3EventHeader`.
- subevent: The "sub-event ID" from the `I3EventHeader`. This refers to the index of the split from event splitter algorithms like HiveSplitter or TriggerSplitter.
- time: The MJD of the event. This is ignored for signal MC, but is necessary for time-independent analyses using scrambled data for background. Units are MJD days.
- azi, zen: The best-fit azimuth and zenith from the selected reconstruction in radians. 
- ra, dec: The equatorial coordinates in radians that you get by combining azimuth, zenith, and time. This can be calculated using either `astropy.coordinates` or `icecube.astro.dir_to_equa`.
- angErr: The estimated angular uncertainty[1] in radians. 
- logE: The log10(reconstructed energy proxy/GeV)[2]. 

[1]: Note that the "estimated angular uncertainty" is a different quantity from the "true angular error" and the "point spread function" ("PSF"): 
- The angErr parameter is an observable and is calculated based on the hits in the detector (paraboloid, cramer-rao) or other reconstruction quantities for a single event. It is an estimate of the width of the likelihood space at the best-fit point, regardless of the distance to the truth. This is effectively "how sure is your reconstruction that this is the correct direction?". 
- The "true angular error" is how far away your reconstruction really is from the true direction. Because this requires knowing a true direction, which is only possible in MC. 
- The PSF is the distribution of the "true angular error" from an ensemble of events. 

These quantities are often confused and sometimes the terms are used interchangeably. Please keep the differences in mind when discussing these uncertainties.

[2]: The values of the "reconstructed energy proxy" are not necessarily neutrino energies. They are instead a reconstructed value correlated with the energy in some way. This is most visible when discussing through-going muons:
- A muon neutrino has a charged-current interaction far outside of the detector, producing a high energy muon
- The muon passes an unknown distance in the uninstrumented ice, losing energy as it travels
- The muon reaches and passes through the detector, depositing photons in the detector
- Photons are detected by the PMTs and recorded as "hits" or "pulses"
- Pre-generated reconstruction splines mapping energy at each point in the detector are used to convert a collection of pulses to a reconstructed energy value

The "reconstructed energy" in this case is based solely on the photons visible in the detector: we are not able to say anything about how much energy was lost before reaching the detector. At best, we could give a "most likely neutrino energy", but that is a more involved process that involves making assumptions about how many events interact at various distances from the detector (ie, assuming a spectrum). For details on how this works, see the "Calculation of the neutrino energy" subsection of our [TXS paper](https://arxiv.org/abs/1807.08816). 

### Simulation events (MC)
For simulation events, there are additional quantities needed.

- trueE: The true neutrino energy of an event in GeV. Note that this does not include a log10() and is a neutrino energy instead of an energy proxy. This is needed for flux weighting.
- trueRa, trueDec: The true direction of the event in radians
- ow: The "OneWeight" parameter in GeV cm^2 sr. This is a simulation quantity available in the `I3MCWeightDict` that includes cross-section, acceptance, and simulation flux calculations. We multiply this by a flux in units of 1/GeV/cm2/sr/s and sum over all events to get the expected event rate in Hz. Note that the `OneWeight` value from the `I3MCWeightDict` needs some additional factors included before it's usable. See the existing converters for examples on how to modify `OneWeight` to be usable for us.


## Examples documentation
A few converters already exist in this folder. 

### Upgrade Simulation
The IceCube Upgrade does not yet include data, making the converter relatively straightforward. We use the GraphNet reconstruction with the [QUESO event selection](https://wiki.icecube.wisc.edu/index.php/IceCube_Upgrade_Simulation_2023#Quick_Upgrade_Event_Selection_for_Oscillations_.28QUESO.29). In addition to the standard quantities, we also extract several values that may be useful for training a regressor to better estimate angular uncertainties.

#### Usage:
```
usage: convert_upgrade_queso.py [-h] --output OUTPUT --nfiles NFILES
                                i3file [i3file ...]

Base script for converting i3 files to NuSources-compatible npy files for use
in csky and skyllh.

positional arguments:
  i3file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        The output npy filename and path to write to
  --nfiles NFILES, -n NFILES
                        Number of files from this dataset that will be converted in total (ie, across all jobs running on this script and including any that had 0 events passing!). Note that this is the number of files from this specific dataset! So eg 10 files from 140028 (GENIE NuMu) and 10 files from 141028 (GENIE NuMuBar) does *not* mean you should use --nfiles 20!
```

#### Example usage:

Convert 10 files in one run. This assumes we will *not* be including any other files from 140028.

```
python convert_upgrade_queso.py /data/sim/IceCubeUpgrade/genie/level4_queso/140028/upgrade_genie_level4_queso_140028_00000* -o upgrade_queso_140028.npy -n 10
```

Convert 20 files total across two jobs.

```
python convert_upgrade_queso.py /data/sim/IceCubeUpgrade/genie/level4_queso/140028/upgrade_genie_level4_queso_140028_00000* -o upgrade_queso_140028_0.npy -n 20
python convert_upgrade_queso.py /data/sim/IceCubeUpgrade/genie/level4_queso/140028/upgrade_genie_level4_queso_140028_00001* -o upgrade_queso_140028_1.npy -n 20
```

Combine these two files with

`python3 combine.py -o upgrade_queso_140028.npy upgrade_queso_140028_*.npy`
