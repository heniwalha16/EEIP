from django.db import models

# Create your models here.


class Question(models.Model):
    operation = models.CharField(max_length=1, choices=[
        ('+', 'Addition'),
        ('-', 'Subtraction'),
        ('*', 'Multiplication'),
        ('/', 'Division'),
    ])
    operand1 = models.IntegerField()
    operand2 = models.IntegerField()
    answer = models.IntegerField()
    
    def __str__(self):
        return f"{self.operand1} {self.operation} {self.operand2} = {self.answer}"