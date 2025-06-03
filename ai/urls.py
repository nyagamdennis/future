from django.urls import path

from .views import TriggerProphetTrainingView

urlpatterns = [
    path("api/trigger-prophet/", TriggerProphetTrainingView.as_view(), name="trigger-prophet"),
]