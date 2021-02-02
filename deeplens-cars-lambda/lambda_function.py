import os
import json
import time
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk
from display import LocalDisplay

def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return

# Create an IoT client for sending to messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

# Dimmensions of datasets
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    
    try:
        # Number of top classes to output
        num_top_k = 3

        
        model_type = 'classification'
        model_name = 'image-classification'
        
        with open('labels.txt', 'r') as f:
	        output_map = [l for l in f]

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # Optimize the model
        error, model_path = mo.optimize(model_name,INPUT_WIDTH,INPUT_HEIGHT)
        
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Model loaded')
        
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (INPUT_HEIGHT, INPUT_WIDTH))
           
            # Run the images through the inference engine and parse the results 
	    # using the parser API
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Get top k results with highest probabilities
            top_k = parsed_inference_results[model_type][0:num_top_k]
            # Add the classes of the top 2 results to the video.
            # image, text, origin, font face, font scale, color, thickness
            # color in BGR, not RGB
            output_text = '{} : {:.2f}'.format(output_map[top_k[0]['label']], top_k[0]['prob'])
            output_text2 = '{} : {:.2f}'.format(output_map[top_k[1]['label']], top_k[1]['prob'])
            cv2.putText(frame, output_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 20), 8)
            cv2.putText(frame, output_text2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 20), 8)
            
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send the top k results to the IoT console via MQTT
            cloud_output = {}
            for obj in top_k:
                cloud_output[output_map[obj['label']]] = obj['prob']
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
      	print('Error in lambda {}'.format(ex))
        client.publish(topic=iot_topic, payload='Error in lambda: {}'.format(ex))

infinite_infer_run()