from django.urls import path
from . import views
from home.dash_apps.finished_apps import simple_example


urlpatterns = [
    path('', views.home, name = 'home'),
    path('charts/', views.charts, name = 'charts')

]