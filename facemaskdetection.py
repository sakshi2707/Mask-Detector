
import uuid

import cv2
import json

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import VisualRecognitionV4
from ibm_watson.visual_recognition_v4 import FileWithMetadata, AnalyzeEnums
from matplotlib import pyplot as plt

'''cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = cap.read()
    imgname = './Images/No Mask/{}.jpg'.format(str(uuid.uuid1()))
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()
'''


apikey = 'eCYKkKadnaAmtAEbfaix3Kn7yJkl5M_32e--8HYe__jv'
url = 'https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/ed1210f9-9230-4af1-abf3-b466a0ef9555'
collection = 'e920e93b-ac86-4108-954a-49927663de8d'


authenticator = IAMAuthenticator(apikey)
service = VisualRecognitionV4('2018-03-19', authenticator=authenticator)
service.set_service_url(url)
#path = './Images/No Mask/download.jpg'
path = './Images/Mask/masked_woman.jpg'
with open(path, 'rb') as mask_img:
    analyze_images = service.analyze(collection_ids=[collection],
                                     features=[AnalyzeEnums.Features.OBJECTS.value],
                                    images_file=[FileWithMetadata(mask_img)]).get_result()
print(analyze_images)


obj = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object']
coords = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['location']

coords

img = cv2.imread(path)

font = cv2.FONT_HERSHEY_SIMPLEX
img=cv2.rectangle(img,(coords['left'],coords['top']),(coords['left']+coords['width'],coords['top']+coords['height']),(0,255,0),10)
img = cv2.putText(img, text=obj, org=(coords['left']+coords['width'], coords['top']+coords['height']), fontFace=font, fontScale=2, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
print(plt.imshow(img))
print(obj)
im=cv2.resize(img,(760,450))
cv2.imshow("Result",im)
cv2.waitKey(0)
