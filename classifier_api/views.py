from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import predict_image

class ImageClassifierView(APIView):
    """API endpoint that accepts an image and returns classification results"""
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        # Check if an image file was uploaded
        if 'image' not in request.FILES:
            return Response({'error': 'Please provide an image file'}, status=status.HTTP_400_BAD_REQUEST)
        
        image_file = request.FILES['image']
        
        # Read the image and make a prediction
        image_bytes = image_file.read()
        prediction = predict_image(image_bytes)
        
        # Check if prediction was successful
        if 'error' in prediction:
            return Response({'error': prediction['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(prediction)