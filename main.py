import os
import numpy as np
from flask import Flask, request , jsonify , abort , send_from_directory , make_response
from werkzeug.utils import secure_filename
import tempfile
import cv2



app = Flask(__name__) 

#define config
HOST='0.0.0.0'
PORT='6000'
#trusted_ips = ('10.2.6.12', '10.2.4.7', '127.0.0.1', '172.17.0.1')

DR_MODEL_ONNEX = "classification_DR_model.onnx"

ROOT_PATH = os.getcwd()
USB_PATH = os.path.join(ROOT_PATH, "model")

ALLOWED_EXTENSIONS= set(['png', 'jpg', 'jpeg', 'BMP', 'bmp' , 'tif' ,'JPG','PNG','JPEG'])

app.config['ROOT_PATH'] = ROOT_PATH
app.config['USB_PATH'] = USB_PATH



"""
################Classification model Loading######################### 
model_name = DR_MODEL_ONNEX
Model_path = os.path.join(app.config['USB_PATH'] , model_name)
tax_name = model_name.split(".") 
try:
    txt_label = open(os.path.join(app.config['USB_PATH'], tax_name[0] + ".txt" ), 'r')
    labels =  [] 
    for element in txt_label.readlines():
        labels.append(element.strip())
        #labels = ['Abnormal', 'Normal']
    txt_label.close() 
except IOError:
    print("Label* File not accessible." )

opencv_net = cv2.dnn.readNetFromONNX(Model_path)
# Check Opencv got any GPU , if yes tell cv to use GPU
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Opencv neural network will use the CUDA backend if the DNN module supports
    opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

    #Now tells that all the neural network computations will happen on the GPU instead of the CPU
    opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #print("OpenCV runing on GPU (:)")
else:
    print ("OpenCV runing on CPU")  
##################################################################
"""



##############Detection Model loading################################ 
with open(os.path.join(app.config['USB_PATH'],'obj.names'), 'r') as f:
    classes = f.read().splitlines()
#print(classes)   

net = cv2.dnn.readNetFromDarknet(os.path.join(app.config['USB_PATH'],'yolov4-tiny-3l.cfg'), os.path.join(app.config['USB_PATH'],'yolov4-tiny-3l_best.weights'))
# Check Opencv got any GPU , if yes tell cv to use GPU
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Opencv neural network will use the CUDA backend if the DNN module supports
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

    #Now tells that all the neural network computations will happen on the GPU instead of the CPU
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #print("OpenCV runing on GPU (:)")
else:
    print ("OpenCV runing on CPU")
#Pass the loaded model to opencv detection dnn    
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(736, 736), swapRB=True) # 736 ,672 , 702
##################################################################


#Allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#Draw circle on image
def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;


"""
# Allowed IPs
@app.before_request
def limit_remote_addr():
    if request.remote_addr not in trusted_ips:
        abort(404)  # Not Found           
"""
"""
@app.route('/classification/', methods = ['GET', 'POST'])
def DR_Quick():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return make_response (jsonify("Uploaded file was empty."), 201)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return make_response (jsonify("Uploaded file name was empty."), 201)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            #Use temporry location
            with tempfile.TemporaryDirectory() as tmpdirname:

                file.save(os.path.join(tmpdirname, filename))
                included_extensions = ['png', 'jpg', 'jpeg', 'BMP', 'bmp' , 'tif' ,'JPG','PNG','JPEG']
                file_names = [fn for fn in os.listdir(tmpdirname) if any(fn.endswith(ext) for ext in included_extensions)]
                input_image = os.path.join(tmpdirname, file_names[0])
            
                try:
                    img = cv2.imread(input_image)                       
                    input_img = img.astype(np.float32)
                    input_img = cv2.resize(input_img, (256, 256))
                    # define preprocess parameters
                    mean = np.array([0.485, 0.456, 0.406]) * 255.0
                    scale = 1 / 255.0
                    std = [0.229, 0.224, 0.225]
                    # prepare input blob to fit the model input:
                    # 1. subtract mean
                    # 2. scale to set pixel values from 0 to 1
                    input_blob = cv2.dnn.blobFromImage(
                        image=input_img,
                        scalefactor=scale,
                        size=(224, 224),  # img target size
                        mean=mean,
                        swapRB=True,  # BGR -> RGB
                        crop=True  # center crop
                    )
                    # 3. divide by std
                    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

                    # set OpenCV DNN input
                    opencv_net.setInput(input_blob)
                    # OpenCV DNN inference
                    out = opencv_net.forward()
                    # get the predicted class ID
                    imagenet_class_id = np.argmax(out)
                    DR_output = str(labels[imagenet_class_id])
                    # get confidence
                    #confidence = out[0][imagenet_class_id]
                    #print("* class ID: {}, label: {}".format(imagenet_class_id, labels[imagenet_class_id]))
                    #print("* confidence: {:.4f}".format(confidence))
                    M_output = jsonify(DR=DR_output)                                                                 
                    return make_response(M_output, 200)
                except:
                    return make_response (jsonify(Error="Failed to Load AI model."), 202) 
"""




@app.route('/Detection_API/', methods = ['GET', 'POST'])
def Classification_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return make_response (jsonify("Uploaded file was empty."), 201)
        file = request.files['file']
        #password = request.form.get("password")
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return make_response (jsonify("Uploaded file name was empty."), 201)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            #Use temporry location
            with tempfile.TemporaryDirectory() as tmpdirname:

                file.save(os.path.join(tmpdirname, filename))
                included_extensions = ['png', 'jpg', 'jpeg', 'BMP', 'bmp' , 'tif' ,'JPG','PNG','JPEG']
                file_names = [fn for fn in os.listdir(tmpdirname) if any(fn.endswith(ext) for ext in included_extensions)]
                input_image = os.path.join(tmpdirname, file_names[0])                            
                try:

                    img = cv2.imread(input_image)
                    classIds, scores, boxes = model.detect(img, confThreshold=0.3, nmsThreshold=0.1)
                    
                    if len(classIds) == 0:
                        cv2.putText(img, "DR Negative", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3, color=(255, 255, 255), thickness=2)
                    else:
                        cv2.putText(img, "DR Positive", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3, color=(255, 255, 255), thickness=2)
                        for (classId, score, box) in zip(classIds, scores, boxes):
                            #cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),color=(0, 255, 0), thickness=2)
                            center_coordinate = (int(box[0] + (box[2]/2)) , int(box[1] + (box[3]/2)))
                            cv2.circle(img,center_coordinate,radius=box[3], color=(0, 255, 0), thickness=3)
                        
                            text = '%s' % (classes[classId[0]])
                            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        color=(0, 255, 0), thickness=2)
                            
    
                   
                    img_name= os.path.basename(input_image)
                    cv2.imwrite(os.path.join(tmpdirname, img_name), img)
                    #ret_path = os.path.join(tmpdirname, img_name)
                    return send_from_directory(tmpdirname, img_name , as_attachment=True ) #path/filename can get errror sometimes 
                    #return send_file(io.BytesIO(image_binary), mimetype='image/jpeg', as_attachment=True, attachment_filename='%s.jpg' % pid)
                except:# Exception as e:
                    #print(e)
                    return make_response (jsonify(Error="Failed to Load AI model."), 202) 


		
if __name__ == '__main__':
   app.run(debug = False , host=HOST, port=PORT, threaded=True)
