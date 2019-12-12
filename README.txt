Essential Requirements to run code:
    1.) python 3 installed 
    2.) numpy installed
 
How to run the code locally:
    1.) change to directory with the .py file you want to run
    2.) make sure all of the weights and test/training dataset and labels are in the same directory at the same level as the python file
    3.) run the code using the command "python A1Part2.py"
    4.) the program will ask if you want to train or test
	A.) Type “train” if you want to train the network. This will train it on the first 2000 images in the MNIST test dataset.
            B.) Type “test” if you want to test the network on the next 8000 images in the MNIST test data set (images it has never been trained on before)
 	Note: You may or may not have to put “test” or “train” in quotes when the program asks for user input
    
How to run the code on Jupiter (preferred):
    1.) click on the "upload" button and select the file A1Part2.ipynb
    2.) make sure to upload all of the weights and test/training dataset and labels in the same directory at the same level as the python file
    3.) click on the uploaded file, this will load it in Jupiter
    4.) click on "cell -> run all" option in the task bar
    5.) the program will ask if you want to train or test
	A.) Type “train” if you want to train the network. This will train it on the first 2000 images in the MNIST test dataset.
            B.) Type “test” if you want to test the network on the next 8000 images in the MNIST test data set (images it has never been trained on before)
 	Note: In my testing you do not need quotes around input on Jupyter
    
Interpreting the results:
Training will not produce any output to the screen besides an error graph. It will only update the weights in the files weights1.bin and weights2.bin
Testing will output a 10X10 confusion matrix for 8000 image samples and the overall accuracy of the testing
