## Training code of SRResCGAN:
- Train datasets: download the datasets from the [NTIRE 2020 Real World Super-Resolution Challenge - Track 1 Image Processing Artifacts](https://data.vision.ee.ethz.ch/cvl/ntire20/).
- Run the `create_dataset.py` to generate the LR/HR pairs according to the paper SRResCGAN section 3.3.
- Place the generated data into the `datasets` folder.
- Finally, run the `main_sr_color.py` file to train the network.
