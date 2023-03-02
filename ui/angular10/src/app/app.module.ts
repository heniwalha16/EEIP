import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TeacherComponent } from './teacher/teacher.component';
import { ShowTechComponent } from './teacher/show-tech/show-tech.component';
import { SharedService } from './shared.service';
import {HttpClientModule} from '@angular/common/http';
import {FormsModule,ReactiveFormsModule} from '@angular/forms';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { ClassComponent } from './class/class.component';
import { ShowClassComponent } from './class/show-class/show-class.component';
import { AddEditTechComponent } from './teacher/add-edit-tech/add-edit-tech.component';
import { AddEditClassComponent } from './class/add-edit-class/add-edit-class.component';
import { ProblemComponent } from './problem/problem.component';
import { AddEditProblemComponent } from './problem/add-edit-problem/add-edit-problem.component';
import { ShowProblemComponent } from './problem/show-problem/show-problem.component';

@NgModule({
  declarations: [
    AppComponent,
    TeacherComponent,
    ShowTechComponent,
    ClassComponent,
    ShowClassComponent,
    AddEditTechComponent,
    AddEditClassComponent,
    ProblemComponent,
    AddEditProblemComponent,
    ShowProblemComponent
    ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    NgbModule
  ],
  providers: [SharedService],
  bootstrap: [AppComponent]
})
export class AppModule { }
