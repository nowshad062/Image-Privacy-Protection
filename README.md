# Image-Privacy-Protection

Step 1: Clone or download the project from the following repository.

	https://github.com/nowshad062/Image-Privacy-Protection


Step 2: Download 32K images from the following link and put them in a folder called 'images' inside the 'Image-Privacy-Protection' directory.

	https://drive.google.com/open?id=0B7omjtJVmHaxRnAxeVd1bW80b0U

Step 3. Install Anaconda with python 2.7 from the following link.

	https://www.anaconda.com/distribution/

Step 4: Open anaconda navigator and create a new environment.


Step 5: Add the following packages by typing in the ‘search package’ box.

	tensorflow, keras, numpy, opencv, jupyter-notebook


Step 7: Open Jupyter Notebook from Anaconda Navigator and browse to 'Image-Privacy-Protection folder'

Step 8: Open the presentation.ipynb file and click Kernel -> Restart & Run all.


Running on GPU:

To upload a file to GPU use the following command:

	scp source_address destination_address

For example if we want to upload 'presentation.py' file to GPU directory 'students/adil/Places365/Keras-VGG16-places365/' we use the following command.


	scp /Users/adil/presentation.py lilin@129.207.86.81:students/adil/Places365/Keras-VGG16-places365/

Then from the GPU terminal go to 'adil/Places365/Keras-VGG16-places365/' directory and run 
	python presentation.py



To download a file from GPU use the following command as a sample:

	scp destination_address source_address 

For example if we want to download 'scenes.csv' file from GPU directory 'students/adil/Places365/Keras-VGG16-places365/objects.csv' we use the following command.

	scp lilin@129.207.86.81:students/adil/Places365/Keras-VGG16-places365/scenes.csv /Users/adil/

The downloaded file will be in the 'Users/adil/' directory
