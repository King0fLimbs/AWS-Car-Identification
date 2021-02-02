# AWS-Car-Identification
Identifies different types of cars using machine learning and Amazon's Lambda platform

The machine learning was done using Amazon Sagemaker to create a model that would be used to train the camera. The images used were a subset of a variety of different classes of cars taken from https://ai.stanford.edu/~jkrause/cars/car_dataset.html. 

The completed training model was then uploaded to the camera, Amazon's DeepLens Camera, which was then pointed at different images of cars to get detection results. These images were not used in the training of the model, so they were not seen by the camera before. The python detection code would run using Amazon's lambda functions on top of the camera feed. It would show its top 3 predictions for the type of car that it was seeing. For unique looking cars like a Lamborghini, it would have near perfect accuracy. But for similar looking cars like two different types of Toyota, it would tend to mix up the two. 

It's not a perfect detection software, but it was fun to work with this kind of technology and see how it could be used in the future as it continues to get more advanced.
