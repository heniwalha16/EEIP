from rest_framework import serializers
from education.models import Class, Teacher, Problem

class ClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = Class
        fields = ('ClassId',
                  'ClassName',
                  'teacher')

class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Teacher
        fields = ('TeacherId',
                  'TeacherName',
                  'PhotoFileName')
class ProblemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Problem
        fields = ('ProblemId',
                  'ProblemText',
                  'Class')