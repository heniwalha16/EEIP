import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {TeacherComponent} from './teacher/teacher.component';
import {ClassComponent} from './class/class.component';
import {ProblemComponent} from './problem/problem.component';

const routes: Routes = [
  {path:'teacher',component:TeacherComponent},
  {path:'Class',component:ClassComponent},
  {path:'teacher/classes',component:ClassComponent},

  {path:'teacher/classes/:teacherId',component:ClassComponent},
  { path: 'teacher/classes/:teacherId/Problem/:ClassId', component: ProblemComponent },

  {path:'Problem',component:ProblemComponent}

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
