# Generated by Django 4.0.6 on 2023-04-20 15:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('education', '0003_rename_name_teacher_firstname_student_firstname_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='problem',
            name='Class',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='education.class'),
        ),
    ]
