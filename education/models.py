from django.db import models

# Create your models here.


# class Question(models.Model):
#     operation = models.CharField(max_length=1, choices=[
#         ('+', 'Addition'),
#         ('-', 'Subtraction'),
#         ('*', 'Multiplication'),
#         ('/', 'Division'),
#     ])
#     operand1 = models.IntegerField()
#     operand2 = models.IntegerField()
#     answer = models.IntegerField()
    
#     def __str__(self):
#         return f"{self.operand1} {self.operation} {self.operand2} = {self.answer}"

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