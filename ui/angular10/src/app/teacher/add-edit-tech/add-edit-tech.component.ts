import { Component,OnInit ,Input} from '@angular/core';
import {SharedService} from 'src/app/shared.service';

@Component({
  selector: 'app-add-edit-tech',
  templateUrl: './add-edit-tech.component.html',
  styleUrls: ['./add-edit-tech.component.css']
})
export class AddEditTechComponent implements OnInit{
  constructor(private service:SharedService){}
  @Input() tech :any;
  TeacherId!: string;
  TeacherName!: string;
  PhotoFileName!:string;
  ngOnInit(): void {
    this.TeacherId=this.tech.TeacherId;
    this.TeacherName=this.tech.TeacherName;
    this.PhotoFileName=this.tech.PhotoFileName;
  }
  addTeacher(){
    var val = {TeacherId:this.TeacherId,
      TeacherName:this.TeacherName,
      PhotoFileName:this.PhotoFileName};
this.service.addTeacher(val).subscribe(res=>{
alert(res.toString());
});
  }
  updateTeacher(){
    var val = {TeacherId:this.TeacherId,
      TeacherName:this.TeacherName,
      PhotoFileName:this.PhotoFileName};
    this.service.updateTeacher(val).subscribe(res=>{
    alert(res.toString());
  });
  }
}
