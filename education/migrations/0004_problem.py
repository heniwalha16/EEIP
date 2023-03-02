# Generated by Django 4.1.6 on 2023-02-25 14:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('education', '0003_class_teacher_delete_departments_delete_employees_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Problem',
            fields=[
                ('ProblemId', models.AutoField(primary_key=True, serialize=False)),
                ('ProblemText', models.CharField(max_length=1000)),
                ('Class', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='education.class')),
            ],
        ),
    ]