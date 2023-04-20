from rest_framework import serializers
from education.models import Class, Student, Teacher, Problem

class ClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = Class
        fields = ('id',
                  'code',
                  'teacher')

class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Teacher
        fields = ('id',
                  'password',
                  'email',
                  'FirstName',
                  'LastName',
                  'picture')
class ProblemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Problem
        fields = ('id',
                  'base64image',
                  'Class')

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = ('id',
                  'password',
                  'email',
                  'FirstName',
                  'LastName',
                  'picture',
                  'Class')