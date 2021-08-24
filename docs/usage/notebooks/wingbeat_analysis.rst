Wingbeat analysis
=================

This notebook demonstrates how to analyse wingbeat data output from
camfi.

First, load the required libraries.

.. code:: ipython3

    from pathlib import Path
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    from camfi.projectconfig import CamfiConfig
    from camfi.plotting.matplotlib import MatplotlibWingbeatFrequencyPlotter
    from camfi.wingbeat import BcesEM, GMM

To run ``via_project.load_all_exif_metadata`` and
``wingbeat_extractor.extract_wingbeats()`` below, you will first need to
download the dataset ``2019-11_cabramurra.zip`` from the Zenodo
repository. The link to the repository is here:
https://doi.org/10.5281/zenodo.4950570.

If you uncomment the code in the next three code cells, it is assumed
you have extracted the images to ``"data/"``. Of course you can extract
it elsewhere and change ``root`` config variable accordingly.

``"data/cabramurra_all_annotations.json"`` already contains all the
wingbeat data we need, so in this instance, we don’t need to re-run the
time-consuming metadata and wingbeat extraction code. Therefore, we
don’t need the image files.

The next cell loads the annotations, and the location and time data used
to correct the camera timestamps, which can often be out (either if the
camera was set incorrectly, or if it just has an inaccurate clock… which
is often the case for cheap cameras).

.. code:: ipython3

    config_path = "data/cabramurra_config.yml"
    
    config = CamfiConfig.parse_yaml_file(config_path)
    
    # We can print out our config using config.json() or config.yaml()
    print(config.json(exclude_unset=True, indent=2))


.. parsed-literal::

    {
      "root": "data",
      "via_project_file": "data/cabramurra_all_annotations.json",
      "day_zero": "2019-01-01",
      "output_tz": "+10:00",
      "filters": {
        "image_filters": {
          "min_annotations": 1
        }
      },
      "camera": {
        "camera_time_to_actual_time_ratio": 1.0,
        "line_rate": 90500.0
      },
      "time": {
        "camera_placements": {
          "2019-11_cabramurra/0004": {
            "camera_start_time": "2019-10-14T13:00:00+11:00",
            "actual_start_time": "2019-11-14T13:00:00+11:00",
            "location": "cabramurra"
          },
          "2019-11_cabramurra": {
            "camera_start_time": "2019-11-14T13:00:00+11:00",
            "location": "cabramurra"
          }
        }
      },
      "place": {
        "locations": [
          {
            "name": "cabramurra",
            "lat": -35.9507,
            "lon": 148.3972,
            "elevation_m": 1513.9,
            "tz": "+10:00"
          }
        ],
        "weather_stations": [
          {
            "location": {
              "name": "cabramurra_smhea_aws_072161",
              "lat": -35.94,
              "lon": 148.38,
              "elevation_m": 1482.4,
              "tz": "+10:00"
            },
            "data_file": "data/cabramurra_bom_weather_201911.csv"
          }
        ],
        "location_weather_station_mapping": {
          "cabramurra": "cabramurra_smhea_aws_072161"
        }
      },
      "wingbeat_extraction": {
        "device": "cpu",
        "scan_distance": 50
      },
      "annotator": {
        "crop": {
          "x0": 0,
          "y0": 0,
          "x1": 4608,
          "y1": 3312
        },
        "training": {
          "mask_maker": {
            "shape": [
              3312,
              4608
            ],
            "mask_dilate": 5
          },
          "min_annotations": 1,
          "max_annotations": 50,
          "test_set_file": "data/cabramurra_test_set.txt",
          "device": "cuda",
          "batch_size": 5,
          "num_workers": 2,
          "num_epochs": 15,
          "outdir": "data",
          "save_intermediate": true
        },
        "inference": {
          "output_path": "data/cabramurra_autoannotated.json",
          "device": "cuda",
          "backup_device": "cpu",
          "score_thresh": 0.0
        },
        "validation": {
          "autoannotated_via_project_file": "data/cabramurra_autoannotated.json",
          "image_sets": [
            "all",
            "test",
            "train"
          ],
          "output_dir": "data"
        }
      }
    }


To get the timestamps for the images, we need to read the EXIF metadata
from the image files. Here we also apply time correction. The code is
commented out since the metadata has already been loaded into
``"data/cabramurra_all_annotations.json"``, but if you are working with
a different dataset, or would like to re-run IO intensive this step,
uncomment the code.

.. code:: ipython3

    # Uncomment if exif metadata hasn't been loaded already.
    # config.load_all_exif_metadata()

After the EXIF metadata has been loaded, we can run the camfi algorithm
to measure the wingbeat frequencies of moths seen in the images. Again,
this has already been run and the data is included in
``"data/cabramurra_all_annotations.json"``, so only uncomment if you
have downloaded the image dataset and want to re-run (or you are running
on your own dataset). This step may take a while to run.

**Note:** This step can be accelerated using a GPU. If you have one on
your system, consider setting
``"wingbeat_extraction":{"device":"cuda","backup_device":"cpu"}`` in
``data/cabramurra_config.json``.

.. code:: ipython3

    # Uncomment if wingbeat data hasn't been extracted already
    # config.extract_all_wingbeats()

After running the above two steps, you might like to save the results to
a new VIA project file. Uncommenting the following will save a new VIA
project file to ``"data/all_annotations_with_wingbeats.json"``.

.. code:: ipython3

    # with open("data/all_annotations_with_wingbeats.json", "w") as f:
    #     f.write(config.via_project.json(indent=2, exclude_unset=True))

The ``camfi.datamodel.via.ViaProject`` class is useful for loading and
validating files which are compatible with VIA, however for some
analyses it is more convenient to have a Pandas DataFrame. The
``to_region_dataframe`` method of ``camfi.datamodel.via.ViaProject``
makes this conversion simple.

.. code:: ipython3

    regions = config.via_project.to_region_dataframe()
    regions




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>img_key</th>
          <th>filename</th>
          <th>name</th>
          <th>datetime_corrected</th>
          <th>datetime_original</th>
          <th>exposure_time</th>
          <th>location</th>
          <th>pixel_x_dimension</th>
          <th>pixel_y_dimension</th>
          <th>score</th>
          <th>best_peak</th>
          <th>blur_length</th>
          <th>snr</th>
          <th>wb_freq_up</th>
          <th>wb_freq_down</th>
          <th>et_up</th>
          <th>et_dn</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2019-11_cabramurra/0001/DSCF0009.JPG-1</td>
          <td>2019-11_cabramurra/0001/DSCF0009.JPG</td>
          <td>polyline</td>
          <td>2019-11-14 20:20:26+11:00</td>
          <td>2019-11-14 20:20:26</td>
          <td>0.111111</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>111.0</td>
          <td>536.292725</td>
          <td>12.966407</td>
          <td>44.505436</td>
          <td>41.726944</td>
          <td>0.107531</td>
          <td>0.114691</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG-1</td>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG</td>
          <td>polyline</td>
          <td>2019-11-14 20:30:29+11:00</td>
          <td>2019-11-14 20:30:29</td>
          <td>0.111111</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>237.0</td>
          <td>1008.291016</td>
          <td>5.783094</td>
          <td>40.686996</td>
          <td>35.853527</td>
          <td>0.104095</td>
          <td>0.118128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG-1</td>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG</td>
          <td>polyline</td>
          <td>2019-11-14 20:30:29+11:00</td>
          <td>2019-11-14 20:30:29</td>
          <td>0.111111</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>165.0</td>
          <td>675.603577</td>
          <td>21.689453</td>
          <td>36.698574</td>
          <td>36.494766</td>
          <td>0.110802</td>
          <td>0.111421</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG-1</td>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG</td>
          <td>polyline</td>
          <td>2019-11-14 20:30:29+11:00</td>
          <td>2019-11-14 20:30:29</td>
          <td>0.111111</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>116.0</td>
          <td>660.467407</td>
          <td>6.046125</td>
          <td>51.069618</td>
          <td>50.624634</td>
          <td>0.110625</td>
          <td>0.111597</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG-1</td>
          <td>2019-11_cabramurra/0001/DSCF0010.JPG</td>
          <td>circle</td>
          <td>2019-11-14 20:30:29+11:00</td>
          <td>2019-11-14 20:30:29</td>
          <td>0.111111</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1414</th>
          <td>2019-11_cabramurra/0010/DSCF0747.JPG-1</td>
          <td>2019-11_cabramurra/0010/DSCF0747.JPG</td>
          <td>point</td>
          <td>2019-11-24 23:21:23+11:00</td>
          <td>2019-11-24 23:21:23</td>
          <td>0.100000</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1415</th>
          <td>2019-11_cabramurra/0010/DSCF0777.JPG-1</td>
          <td>2019-11_cabramurra/0010/DSCF0777.JPG</td>
          <td>polyline</td>
          <td>2019-11-25 04:22:54+11:00</td>
          <td>2019-11-25 04:22:54</td>
          <td>0.100000</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>107.0</td>
          <td>520.138428</td>
          <td>11.467738</td>
          <td>48.212074</td>
          <td>48.084389</td>
          <td>0.099867</td>
          <td>0.100133</td>
        </tr>
        <tr>
          <th>1416</th>
          <td>2019-11_cabramurra/0010/DSCF0779.JPG-1</td>
          <td>2019-11_cabramurra/0010/DSCF0779.JPG</td>
          <td>polyline</td>
          <td>2019-11-25 04:43:00+11:00</td>
          <td>2019-11-25 04:43:00</td>
          <td>0.100000</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>82.0</td>
          <td>389.173492</td>
          <td>3.202193</td>
          <td>47.306557</td>
          <td>46.436455</td>
          <td>0.099072</td>
          <td>0.100928</td>
        </tr>
        <tr>
          <th>1417</th>
          <td>2019-11_cabramurra/0010/DSCF0780.JPG-1</td>
          <td>2019-11_cabramurra/0010/DSCF0780.JPG</td>
          <td>polyline</td>
          <td>2019-11-25 04:53:04+11:00</td>
          <td>2019-11-25 04:53:04</td>
          <td>0.100000</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>129.0</td>
          <td>591.514160</td>
          <td>5.017477</td>
          <td>46.047268</td>
          <td>45.040775</td>
          <td>0.098895</td>
          <td>0.101105</td>
        </tr>
        <tr>
          <th>1418</th>
          <td>2019-11_cabramurra/0010/DSCF0851.JPG-1</td>
          <td>2019-11_cabramurra/0010/DSCF0851.JPG</td>
          <td>point</td>
          <td>2019-11-26 04:43:00+11:00</td>
          <td>2019-11-26 04:43:00</td>
          <td>0.100000</td>
          <td>cabramurra</td>
          <td>4608</td>
          <td>3456</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>1419 rows × 17 columns</p>
    </div>



Next we want proceed with just polyline annotations (as these are the
only type which enable wingbeat measurement), and we also want to filter
the data by an SNR threshold.

.. code:: ipython3

    snr_thresh = 4.0
    polyline_regions = regions[regions["name"] == "polyline"]
    above_thresh = polyline_regions[polyline_regions["snr"] >= snr_thresh]
    len(above_thresh)




.. parsed-literal::

    580



We can already visualise the data, with the SNR threshold indicated by a
red line.

.. code:: ipython3

    plotter = MatplotlibWingbeatFrequencyPlotter(
        polyline_regions=polyline_regions,
        snr_thresh=snr_thresh,
    )
    fig = plotter.plot()



.. image:: wingbeat_analysis_files/wingbeat_analysis_15_0.png


Based on the above plots, we now wish to select a number of target
classes. This will be used for fitting a Gaussian mixture-model to the
preliminary winbeat frequencies (which assume zero bodylength), and for
classification in the linear regression step for the unbiased estimate
of wingbeat frequency.

.. code:: ipython3

    n_classes = 2

Operating on the preliminary wingbeat data (figure a), we can fit a
Gaussian mixture-model.

.. code:: ipython3

    gmm = GMM.log10_from_region_dataframe(
        above_thresh,
        n_classes,
        seed=1234567890  # Not reuired to set, but makes the results
                         # reproducible
    )
    gmm_results = sorted(gmm.fit())  # Order of classes is random, so we sort
                                     # to make it predictable.
    
    print("log10 Gaussian Mixture Model parameters:")
    print("\n".join(str(r) for r in gmm_results))


.. parsed-literal::

    log10 Gaussian Mixture Model parameters:
    mean=1.3968704095248994 std=0.0755076055338635 weight=0.14025772662404282
    mean=1.6913164856912772 std=0.07106824294152196 weight=0.8597422733759577


In Hz, the mean preliminary wingbeat frequencies for the respective
classes are

.. code:: ipython3

    print("\n".join(f"{10 ** r.mean} Hz" for r in gmm_results))


.. parsed-literal::

    24.93850468145082 Hz
    49.126574840025675 Hz


We can set the ``gmm_results`` parameter to plot the figure with the
Gaussian mixture model shown.

.. code:: ipython3

    plotter = MatplotlibWingbeatFrequencyPlotter(
        polyline_regions=polyline_regions,
        snr_thresh=snr_thresh,
        gmm_results=gmm_results,
    )
    fig = plotter.plot()



.. image:: wingbeat_analysis_files/wingbeat_analysis_23_0.png


Now we use an EM algorithm to classify the data using BCES regressions
of :math:`L` vs. :math:`P \Delta t`.

.. code:: ipython3

    bces_em = BcesEM.from_region_dataframe(
        above_thresh, n_classes, seed=1234567890
    )
    bces_results = bces_em.fit()

The order of the classes is random, so we sort to make them predictable
(and line up better with the sorted classes from the GMM). This will
make colouring comparable across the two models. Unfortunately we can’t
*just* sort ``bces_results``, because then the colours of the scatter
plot wouldn’t be guaranteed to match the regression lines. We have to
also re-map the ``bces_em.class_mask`` values.

This can do this re-mapping with some indexing trickery using
``np.argsort``, and then we can sort ``bces_results`` the normal way.

.. code:: ipython3

    inverse_index = np.argsort(np.argsort(bces_results))
    class_mask = inverse_index[bces_em.class_mask]
    bces_results = sorted(bces_results)
    
    print("Classifications:")
    print("\n".join(str(n) for n in np.bincount(class_mask)))
    print("Multiple BCES linear regression parameters:")
    print("\n".join(str(b) for b in bces_results))


.. parsed-literal::

    Classifications:
    75
    505
    Multiple BCES linear regression parameters:
    gradient=23.684945219639722 y_intercept=21.164640791557133 gradient_stderr=1.8436575083709323 y_intercept_stderr=30.356672106881156 cov_xy=-54.37303718722761
    gradient=48.648490469072115 y_intercept=30.410128999818426 gradient_stderr=1.4469572965456388 y_intercept_stderr=18.927271182452444 cov_xy=-26.431054889875146


Finally, we reproduce the figure from the publication, which includes
both the GMM and EM classification

.. code:: ipython3

    plotter = MatplotlibWingbeatFrequencyPlotter(
        polyline_regions=polyline_regions,
        snr_thresh=snr_thresh,          # Must be the same threshold used to
                                        # make the class_mask.
        class_mask=class_mask,          # Adds colour to the measurements
        gmm_results=gmm_results,        # Include the GMM plots
        bces_results=bces_results,      # Include the BCES regression lines
    )
    fig = plotter.plot()



.. image:: wingbeat_analysis_files/wingbeat_analysis_29_0.png


``fig`` is just a matplotlib ``Figure`` instance, so we can save it
quite easily.

.. code:: ipython3

    fig.savefig("wingbeat_frequency_figure.pdf", dpi=600.0, pad_inches=0.0)
