import { Component, Input, OnInit } from '@angular/core';
import { SharedService } from 'src/app/shared.service';

@Component({
  selector: 'app-add-edit-class',
  templateUrl: './add-edit-class.component.html',
  styleUrls: ['./add-edit-class.component.css']
})
export class AddEditClassComponent implements OnInit {

  constructor(private service:SharedService){}
  @Input() 
  Classroom :any;
  ClassId!: string;
  ClassName!: string;
  teacherId!:string;
  ngOnInit(): void {
    this.ClassId=this.Classroom.ClassId;
    this.ClassName=this.Classroom.ClassName;
    this.teacherId=this.Classroom.teacherId;

  }
  addClass(){
    var val = {ClassId:this.ClassId,
      ClassName:this.ClassName,
      teacher:this.teacherId
    };
this.service.addClass(val).subscribe(res=>{
alert(res.toString());
});
  }
  updateClass(){
    var val = {ClassId:this.ClassId,
      ClassName:this.ClassName,
      teacher:this.teacherId
    };
    this.service.updateClass(val).subscribe(res=>{
    alert(res.toString());
  });
  }
}

