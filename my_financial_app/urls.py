# my_financial_app/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings # <--- ADD THIS LINE
from django.conf.urls.static import static # <--- ADD THIS LINE

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('financial_ml.urls')), # <--- ADD THIS LINE
]

# <--- ADD THESE LINES FOR SERVING MEDIA FILES IN DEVELOPMENT
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
