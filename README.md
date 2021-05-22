# FlowersImageClassifierTerminalApp
# This repository contains:
 * A jupyter notebook p1_image_classifier.ipynb containing the code to train a model on the [oxford_flowers102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) dataset
 * And a python script in the folder p2_image_classifier which can be run in the terminal with input arguments:
  * Location of the image of a flower which needs to be predicted
  * Location of model being used (in this project's case you can use my trained model from p1_image_classifier.ipynb which I have named model1.h5 and added in the same folder)
  * Optional Argument 1:The top k classes which were predicted (default value = 1)
  * Optional Argument 2:Location of the JSON file containg class names(flower names) against their indices
  an example of the input to be used in the terminal:
  python predict.py ./test_images/orange_dahlia.jpg ./model1.h5 --top_k 5 --category_names label_map.json
 # Caution:
 ## The model is trained with tensorflow 2.5.0 and other details specified in p1_image_classifer.ipynb, so the SAME requirements are needed for PART 2 of the project
 ## The Terminal Application i.e. p2_image_classifer was run on Jupyter Notebook's Terminal
 ## Don't Forget to change your directory to p2_image_classifier before running the python script
