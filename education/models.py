from django.db import models


from django.db import models


class QuizQuestion(models.Model):
    OPERATIONS = (
        ('+', 'Addition'),
        ('-', 'Subtraction'),
        ('*', 'Multiplication'),
        ('/', 'Division'),
    )
    operation = models.CharField(max_length=1, choices=OPERATIONS)
    num1 = models.IntegerField()
    num2 = models.IntegerField()
    answer = models.IntegerField()
    difficulty = models.IntegerField(default=1)
# Create your models here.
'''
class Teacher(models.Model):
    TeacherId = models.CharField(primary_key=True,max_length=20)
    TeacherName = models.CharField(max_length=100)
    PhotoFileName = models.CharField(max_length=100)

class Student(models.Model):
    StudentId = models.CharField(primary_key=True,max_length=20)
    TeacherName = models.CharField(max_length=100)
    PhotoFileName = models.CharField(max_length=100)
    
class Class(models.Model):
    ClassId =models.CharField(primary_key=True,max_length=20)
    ClassName = models.CharField(max_length=10)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)

class Problem(models.Model):
    ProblemId = models.CharField(primary_key=True,max_length=20)
    ProblemText = models.CharField(max_length=1000)
    Class = models.ForeignKey(Class, on_delete=models.CASCADE)
'''
#--------------------------------

# Import necessary modules
from django.db import models
from django.contrib.auth.models import AbstractUser

# Define Teacher model
class Teacher(models.Model):
    id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=100)
    email = models.EmailField()
    FirstName = models.CharField(max_length=100)
    LastName = models.CharField(max_length=100)
    picture = models.TextField() # Assumes usage of Django's built-in ImageField

    def __str__(self):
        return self.name

# Define Class model
class Class(models.Model):
    id = models.AutoField(primary_key=True) 
    code = models.CharField(max_length=100)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)

    def __str__(self):
        return f"Class {self.id}"

# Define Problem model
class Problem(models.Model):
    id = models.AutoField(primary_key=True)
    base64image = models.TextField()  # Assumes usage of base64-encoded images as text
    Class = models.ForeignKey(Class, on_delete=models.CASCADE, null=True, default=None)

    # Add other fields as needed

    def __str__(self):
        return f"Problem {self.id}"

# Define Student model
class Student(models.Model):
    id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=100)
    email = models.EmailField()
    FirstName = models.CharField(max_length=100)
    LastName = models.CharField(max_length=100)
    picture = models.TextField()  # Assumes usage of Django's built-in ImageField
    Class = models.ForeignKey(Class, on_delete=models.CASCADE, null=True, default=None)
    def __str__(self):
        return self.username
