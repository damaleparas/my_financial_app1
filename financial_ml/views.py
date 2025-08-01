# financial_ml/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # <--- IMPORTANT: For simplicity, disable CSRF. In production, use proper CSRF tokens.
from django.conf import settings
import os
import uuid # For generating unique filenames for uploaded files

from .ml_integration import ml_service # Import the initialized MLService instance

def index(request):
    """Renders the main HTML page for the application."""
    return render(request, 'financial_ml/index.html')

@csrf_exempt # Disable CSRF for this API endpoint. Consider proper CSRF handling in production.
def train_model_api(request):
    """API endpoint to trigger model training."""
    if request.method == 'POST':
        train_type = request.POST.get('train_type')

        if train_type == 'sample':
            try:
                ml_service.train_model_sample_data()
                return JsonResponse({"status": "success", "message": "Model trained with sample data."})
            except Exception as e:
                return JsonResponse({"status": "error", "message": f"Training with sample data failed: {e}"}, status=500)
        elif train_type == 'custom':
            if 'csv_file' in request.FILES:
                csv_file = request.FILES['csv_file']
                # Save the uploaded file temporarily to MEDIA_ROOT
                file_name = f"custom_train_data_{uuid.uuid4()}.csv"
                file_path = os.path.join(settings.MEDIA_ROOT, file_name)
                
                try:
                    with open(file_path, 'wb+') as destination:
                        for chunk in csv_file.chunks():
                            destination.write(chunk)
                    
                    ml_service.train_model_custom_data(file_path)
                    return JsonResponse({"status": "success", "message": "Model trained with custom data."})
                except Exception as e:
                    return JsonResponse({"status": "error", "message": f"Training with custom data failed: {e}"}, status=500)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(file_path):
                        os.remove(file_path)
            else:
                return JsonResponse({"status": "error", "message": "No CSV file provided for custom training."}, status=400)
        else:
            return JsonResponse({"status": "error", "message": "Invalid training type."}, status=400)
    return JsonResponse({"status": "error", "message": "Only POST requests are allowed for training."}, status=405)

@csrf_exempt # Disable CSRF for this API endpoint. Consider proper CSRF handling in production.
def predict_api(request):
    """API endpoint to make a prediction from an uploaded balance sheet."""
    if request.method == 'POST':
        if 'balance_sheet_file' in request.FILES:
            balance_sheet_file = request.FILES['balance_sheet_file']
            # Save the uploaded file temporarily to MEDIA_ROOT
            file_name = f"balance_sheet_{uuid.uuid4()}.csv"
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            
            try:
                with open(file_path, 'wb+') as destination:
                    for chunk in balance_sheet_file.chunks():
                        destination.write(chunk)
                
                prediction_result = ml_service.make_prediction_from_balance_sheet(file_path)
                if "error" in prediction_result:
                    return JsonResponse({"status": "error", "message": prediction_result["error"]}, status=500)
                return JsonResponse({"status": "success", "data": prediction_result})
            except Exception as e:
                return JsonResponse({"status": "error", "message": f"Prediction failed: {e}"}, status=500)
            finally:
                # Clean up the temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            return JsonResponse({"status": "error", "message": "No balance sheet file provided."}, status=400)
    return JsonResponse({"status": "error", "message": "Only POST requests are allowed for prediction."}, status=405)

def test_prediction_api(request):
    """API endpoint to test prediction with a hardcoded sample."""
    if request.method == 'GET':
        try:
            test_result = ml_service.test_prediction_with_sample_data()
            return JsonResponse({"status": "success", "data": test_result})
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Test prediction failed: {e}"}, status=500)
    return JsonResponse({"status": "error", "message": "Only GET requests are allowed for test prediction."}, status=405)
