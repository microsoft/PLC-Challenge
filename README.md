# ICASSP 2024 Audio Deep Packet Loss Concealment Grand Challenge

For the 2022 challenge, see below.

Data downloads
* The validation data for the 2024 edition of the challenge has now been released: https://aecchallengepublic.blob.core.windows.net/plc2024/plcchallenge24_val_v2.zip
* A tool to check and validate processing latency can be found here: https://aecchallengepublic.blob.core.windows.net/plc2024/latency_test_v2.zip
* The blind set has now been released, and can be found here: https://aecchallengepublic.blob.core.windows.net/plc2024/plc_challenge_2024_blind_release.zip

We recommend using the recently released new version of PLCMOS, which is part of the speechmos · PyPI(opens in new tab) package, to aid with development.

## Citation
If you use this dataset in a publication please cite the following paper:<br />

```BibTex
@inproceedings{diener2024icassp,
  title        = {The ICASSP 2024 Audio Deep Packet Loss Concealment Grand Challenge},
  author       = {Diener, Lorenz and Branets, Solomiya and Saabas, Ando and Cutler, Ross},
  year         = 2024,
  month        = apr,
  booktitle    = {{ICASSP} 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing},
  code         = {https://aka.ms/plc-challenge},
}
```

The previous challenges are: 
```BibTex
@inproceedings{diener2022interspeech,
  title        = {INTERSPEECH 2022 Audio Deep Packet Loss Concealment Challenge},
  author       = {Diener, Lorenz and Sootla, Sten and Branets, Solomiya and Saabas, Ando and Aichner, Robert and Cutler, Ross},
  year         = 2022,
  month        = sep,
  booktitle    = {{INTERSPEECH} 2022 - 22nd Annual Conference of the International Speech Communication Association},
  doi          = {10.21437/Interspeech.2022-10829},
  code         = {https://aka.ms/plc-challenge},
}
```

# INTERSPEECH 2022 Audio Deep Packet Loss Concealment Challenge

This repository will contain data and example code for the INTERSPEECH 2022 Audio Deep 
Packet Loss Concealment Challenge.

You can find more information about the challenge and how to enter at https://aka.ms/plc_challenge

If you have any questions, please contact us via e-mail at plc_challenge@microsoft.com

## Dataset

The training and validation dataset has now been released and is available as a tar.gz archive:

https://aecchallengepublic.blob.core.windows.net/plc2022/test_train.tar.gz

The blind set is now also available:

https://aecchallengepublic.blob.core.windows.net/plc2022/blind.tar.gz

Update (24. March 2022): The reference data for the blind set is now available:

https://aecchallengepublic.blob.core.windows.net/plc2022/blind_set_reference.tar.gz

Please make sure to submit your results by the deadline, March 8th 2022 23:59 AoE.

Additional information about the data included can be found in [our challenge paper](INTERSPEECH_2022_Deep_PLC_Challenge.pdf), and information about how to register for the challenge can be found at https://aka.ms/plc_challenge .

A multipart zip file download of the training set is available for people who cannot download it as one big file:

https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.001
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.002
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.003
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.004
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.005
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.006
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.007
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.008
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.009
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.010
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.011
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.012
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.013
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.014
https://aecchallengepublic.blob.core.windows.net/plc2022/split/test_train.zip.015


## PLC-MOS

To help with model development, we will provide access to a prototype PLC-MOS neural model API which will provide MOS score estimates for audio files with packet loss concealment applied.
For further details on how to get access to this API, refer to https://aka.ms/plc_challenge . You can find an API usage example in [PLC-MOS-API-Example.ipynb](PLC-MOS-API-Example.ipynb) .

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
