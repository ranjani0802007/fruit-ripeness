from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('predict/', views.predict_view, name='upload'),
    path('history/', views.history_view, name='history'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('delete/bulk/', views.bulk_delete_predictions, name='bulk_delete_predictions'),
]
