from django.contrib import admin

# Register your models here.
from .models import Problem,Student,Teacher,Class

admin.site.register(Problem)
admin.site.register(Student)
admin.site.register(Teacher)
admin.site.register(Class)
