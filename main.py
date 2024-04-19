import numpy as np  
import cv2
import onnxruntime
import os

session = onnxruntime.InferenceSession('best(18.4).onnx')
class_names = ['covered', 'not-covered']
class_id = 0

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype('float32')
    image /= 255.0
    image = image[None]
    return image

# # Function to run inference
# def detect_face_cover(image_path):
#     global outputs, class_id
#     input_image = preprocess_image(image_path)
#     outputs = session.run(None, {'images': input_image})
#     # o1 = np.array(outputs)
    
#     outputs = outputs[0][0]
#     prob = outputs[:, 4].max()
#     print(prob)
#     confidence_threshold = 0.7
    
#     if prob > confidence_threshold:
#         result = class_names[0]
#         class_id = 0
#     else:
#         result = class_names[1]
#         class_id = 1
        
#     return result
 

def detect_face_cover(image_path):
    global outputs, class_id
    input_image = preprocess_image(image_path)
    output = session.run(None, {'images': input_image})
    o1 = np.array(output)
    # print(o1.shape)
    outputs = output[0][0]
    prob = outputs[:, 4].max()
    # print(prob)
    class_probabilities = output[0][0][:, -2:]
    # print(class_probabilities)
    # print(class_probabilities.shape)
    
    sumN = 0
    sumC = 0
    for i in range(len(class_probabilities)):
        confidence = o1[0, 0, i, 4]
        sumC += confidence * o1[0, 0, i, 5]
        sumN += confidence * o1[0, 0, i, 6]
    covered_probability = sumC
    not_covered_probability = sumN
    # print(covered_probability, not_covered_probability)
    if covered_probability > not_covered_probability:
            result = "covered"
            class_id = 0
    else:
            result = "not-covered"
            class_id = 1
            #break
    if covered_probability > not_covered_probability:
        result = "covered"
    else:
        result = "not-covered"
    
    return result

# image_path = 'test-images/test5.jpg'
# result = detect_face_cover(image_path)
# print(f"The face in the image is {result}.")

folder_path = 'test-images'
image_files = os.listdir(folder_path)

for image_file in image_files:
    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # check if file is an image
        image_path = os.path.join(folder_path, image_file)
        result = detect_face_cover(image_path)
        print(f"The face in the image {image_file} is {result}.")


# # Decoding the output
# input_img = cv2.imread(image_path)
# input_img = cv2.resize(input_img, (640, 640))
# box_x = int(outputs[:, 0].max())//2
# box_y = int(outputs[:, 1].max())//2
# box_w = int(outputs[:, 2].max())//10
# box_h = int(outputs[:, 3].max())//10
# scores = int(outputs[:, 4].max())*100
# #class_id = int(outputs[:, 5])
# #print(box_x, box_y, box_w, box_h, scores, class_id)
# cv2.rectangle(input_img, (box_w, box_h), (box_w + box_x, box_h + box_y), (0, 0, 255), 2)
# cv2.putText(input_img, class_names[class_id], (box_w, box_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# cv2.imshow('output', input_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()