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

class Teacher(models.Model):
    TeacherId = models.CharField(primary_key=True,max_length=20)
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