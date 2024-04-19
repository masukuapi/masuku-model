# Masuku-Model
This repo holds the model for the Masuku Project

The model is based on a pre-trained YOLOv5 Image Classification Model
We used a model that was trained on a dataset of surgical masks and then applied transfer learning using our dataset
Our dataset contained images of people wearing various forms of face coverings like helmets, masks, bandanas, ski masks and so on

Firstly, we used the yolo_test.ipynb to train the image on our dataset. The model was trained for 50 epochs and came out with an accuracy of 95%

We exported the model as model.pt and model.onnx files. We decided to use model.onnx because of the ease of work
# Model-Working
  1. Pass an image as input
  2. The output returns a matrix of shape (1, 1, 25200, 7)
  3. From that matrix we were able to extract the coordinates, width and height of the bounding box and the confidence of the model
  4. Then we use OpenCV to draw the bounding box on the image
     
Now, this is not how the final version of the program will work. We plan to use live footage of a person and take the first frame of their appearance as the input image
Then we can say whether the person can use the ATM or No
