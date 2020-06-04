# Testing the SRResCGAN:
- Platform required to run our test code: 
	- Pytorch [1.3.1] 
	- Python [3.7.3] 
	- Numpy [1.16.4] 
	- cv2 [4.1.1] 

- Testing:
	- Run file named "test_srrescgan.py" (without self-ensemble strategy) to produce SR results for RWSR-track1 and RWSR-track2
	- Run file named "test_srrescgan_plus.py" (with self-ensemble strategy) to produce SR results for RWSR-track1 and RWSR-track2

- Contained Directories information: 
	- models: SRResCGAN Network structure.
	- trained_nets_x4: SRResCGAN generator trained network
	- LR: Given LR images .
	- sr_results_x4: produced output images of our network saved here. 
