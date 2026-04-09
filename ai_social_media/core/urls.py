from django.urls import path
from .views import home, predict_performance, ai_caption, ai_hashtags

urlpatterns = [
    path('', home),
    path('predict/', predict_performance),
    path('ai-caption/', ai_caption),
    path('ai-hashtags/', ai_hashtags),
]