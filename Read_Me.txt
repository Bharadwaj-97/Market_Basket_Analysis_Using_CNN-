This folder includes the following

----------------------------------
1. MLProject.py
2. Trained_Objects.txt
3. Test_Image,Test_Image1
4. Market_Basket.csv
5. yolov3
6. yolov3.weights
7. Additional
8. Project Report
----------------------------------

1. MLProject.py contains the project file which amalgamates the object detection algorithm and FP Growth algorithm to provide recommendations.
2. Trained_Objects.txt contains the various objects the object detection algorithm must detect. This is the output layer of the CNN.
3. Test_Images - These are the images I have passed to test MLProject.py
4. Market_Basket.csv - This file contains a sample products purchased in 60 transactions. This is the source for FP growth algorithm.
5. yolov3 - This file contains the image annotations and labels of coco data set.
6. yolov3.weights - This file contains the weights and biases. These are pretrained weights and biases meaning they are can understand the various types of kernels.
7. Additional - As the aim of the project is to combine Object detection and FP growth algorithm, I could accomplish the task using coco dataset.I have additionally tried to 
                train the grocery dataset.  Usually a minimum of 100 epochs gives a better accuracy for my training dataset with a batch size of 32. However, its requires
                 a greater computational resources. The best possible training I could do was with a batch size of 16 and 5 epcohs. Each epoch took around 5 hrs to complete , 
                and the 5the epoch ran for 2 hrs and ran out of resources and training terminated. Taking a total 22 hrs. The yolov5/runs foldercontains the result of training.
                
8. Project Report - This contains the report of my Project.




