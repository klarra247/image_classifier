from django.urls import path
from .views import ImageClassifierView

urlpatterns = [
    path('classify/', ImageClassifierView.as_view(), name='classify'),
]