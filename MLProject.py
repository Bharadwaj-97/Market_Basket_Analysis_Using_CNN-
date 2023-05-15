import cv2
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt

# Test image
image = cv2.imread("Test_Image.PNG")
dataset = pd.read_csv("Market_Basket.csv",header=None)
result = []

categories = 'Trained_Objects.txt'
config = 'yolov3.cfg'
weights = 'yolov3.weights'

# CNN requires the input image boken into X and Y coordinates representing the height and width of the image
# So that it fetches the exact image and ignores the unwanted area
Height = image.shape[0]  # Height of Input Image
Width = image.shape[1]

#We cannot train CNN with large no.of CNN using same learning rate.
# Depending of the result of iterations CNN decides the learning rate
#scales is a coefficients at which learning_rate will be multiplied.
# Determines how the learning_rate will be changed during increasing number of iterations during training.
# Providing scale with large value wont recognise all the images.
scale = 0.004

# Here we are preparing the nodes in the final layer i.e, Total categories in the output layer
with open(categories, 'r') as f:
    categories = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)
# Blob from Image tries to identify the pixels in the images. It scales the image, mean substraction and channel swapping
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5   # How likely the bounded box contains the object
nms_threshold = 0.4  # Non Maximun Supression - It choses the ideal region of object from multiple proposals


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Here we draw rectangles around the images whose categories are identified.
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(categories[class_id])
    color = (255,0,0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

outputs = net.forward(get_output_layers(net))

for out in outputs:
    for detected in out:
        scores = detected[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detected[0] * Width)
            center_y = int(detected[1] * Height)
            w = int(detected[2] * Width)
            h = int(detected[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

for i in class_ids:
    if categories[i] not in result:
        result.append(categories[i])
print(result)

cv2.imshow("object detection", image)
cv2.waitKey()
# Saves the result
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()

transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i, j])

transaction = np.array(transaction)
df = pd.DataFrame(transaction, columns=["items"])
df["incident_count"] = 1
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
# creating tree map using plotly
fig1 = px.treemap(df_table, path=[ "items"], values='incident_count',
                  color=df_table["incident_count"], hover_data=['items'],
                 color_continuous_scale='Blues',
              )
fig1.show()

# Fetches contents of dataset into an array
transaction = []
for i in range(dataset.shape[0]):
    transaction.append([str(dataset.values[i, j]) for j in range(dataset.shape[1])])

transaction = np.array(transaction)

# initializing the transactionEncoder
# It converts the items in dataset to machine learning understandable format of boolean array
te = TransactionEncoder()
te_array = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_array, columns=te.columns_)

# Cleaning the null columns
dataset.drop('nan', axis=1, inplace=True)

# running the fpgrowth algorithm
fpresult = fpgrowth(dataset, min_support=0.07, use_colnames=True)

# creating asssociation rules
fpresult = association_rules(fpresult, metric="confidence", min_threshold=0.4)

# printing association rule

sort = fpresult.sort_values("confidence", ascending=False)

# Defroze the antecedents and consequents
a= [list(x) for x in sort.iloc[:,0]]
b=[list(x) for x in sort.iloc[:,1]]
sort['a']=a
sort['b']= b

indexName = sort[sort.iloc[:,9] == "nan"].index
sort.drop(indexName, inplace=True)

# I'm getting the index of rows that have object present so that based on the index i'll get those elements in next column
a_index=[]
for y in result:
 for idx,x in enumerate(sort.iloc[:,9]):
       if y in x:
           a_index.append(idx)

if len(a_index):
 for x in a_index:
    print("You may also want",sort.iloc[x,10] )
else:
    print("It is not a frequent pattern or Try adjusting the the threshold for Support and Confidence")

wow=sort[['a','b','support','confidence']]
print("completed")
wow.plot(x='a', y='support', kind="bar")
wow.plot(x='a', y='confidence', kind="bar")
plt.show()
