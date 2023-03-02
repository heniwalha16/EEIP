import { Component,Input,OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {SharedService} from 'src/app/shared.service';

@Component({
  selector: 'app-show-class',
  templateUrl: './show-class.component.html',
  styleUrls: ['./show-class.component.css']
})
export class ShowClassComponent implements  OnInit{
  constructor(private service:SharedService,private route: ActivatedRoute){}
  ModalTitle!: string;
  ActivateAddEditClassComp:boolean=false;
  Classroom:any;
  ClassroomIdFilter:string="";
  ClassroomNameFilter:string="";
  ClassroomListWithoutFilter:any=[];
  ClassList:any=[];
  @Input()
  teacherId!: string;

  ngOnInit(): void {
    this.teacherId = this.route.snapshot.paramMap.get('teacherId') || 'default';

    this.refreshClassList();
    
  }
  refreshClassList(){
    this.service.getClassList(this.teacherId).subscribe(data=>{
      this.ClassList=data;
      this.ClassroomListWithoutFilter=data;
      
    });
}
addClick(){
  this.Classroom={
    teacherId:this.teacherId,
    ClassId:0,
    ClassName:""
    
  }
  this.ModalTitle="Add classroom";
  this.ActivateAddEditClassComp=true;

}
deleteClick(item: { ClassId: any; }){
  if(confirm('Are you sure??')){
    this.service.deleteClass(item.ClassId).subscribe(data=>{
      alert(data.toString());
      this.refreshClassList();
    })
  }
}

editClick(item: any){
  this.Classroom=item;
  this.Classroom.teacherId = this.teacherId;

  this.ModalTitle="Edit classroom";
  this.ActivateAddEditClassComp=true;
}

closeClick(){
  this.ActivateAddEditClassComp=false;
  this.refreshClassList();
}
FilterFn(){
  var TeacherIdFilter = this.ClassroomIdFilter;
  var TeacherNameFilter = this.ClassroomNameFilter;

  this.ClassList = this.ClassroomListWithoutFilter.filter(function (el: { ClassId: { toString: () => string; }; ClassName: { toString: () => string; }; }){
      return el.ClassId.toString().toLowerCase().includes(
        TeacherIdFilter.toString().trim().toLowerCase()
      )&&
      el.ClassName.toString().toLowerCase().includes(
        TeacherNameFilter.toString().trim().toLowerCase()
      )
  });
}

sortResult(prop: string | number,asc: any){
  this.ClassList = this.ClassroomListWithoutFilter.sort(function(a: { [x: string]: number; },b: { [x: string]: number; }){
    if(asc){
        return (a[prop]>b[prop])?1 : ((a[prop]<b[prop]) ?-1 :0);
    }else{
      return (b[prop]>a[prop])?1 : ((b[prop]<a[prop]) ?-1 :0);
    }
  })
}

}