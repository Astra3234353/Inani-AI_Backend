from django.urls import path
from .views import UploadImageView, UploadVideoView

urlpatterns = [
    path('upload-video/', UploadVideoView.as_view(), name='upload-video'),
    path('upload-image/', UploadImageView.as_view(), name='upload-image'),
]