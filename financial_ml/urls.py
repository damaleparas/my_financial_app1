# financial_ml/urls.py

from django.urls import path
from . import views

app_name = 'financial_ml' # Namespace for your app's URLs

urlpatterns = [
    path('', views.index, name='index'), # Root path for the app, serves the HTML page
    path('api/train/', views.train_model_api, name='train_model_api'), # API endpoint for training
    path('api/predict/', views.predict_api, name='predict_api'), # API endpoint for prediction
    path('api/test_prediction/', views.test_prediction_api, name='test_prediction_api'), # API endpoint for testing
]
