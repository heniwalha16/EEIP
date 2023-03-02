import { Component,OnInit } from '@angular/core';
import {SharedService} from 'src/app/shared.service';

@Component({
  selector: 'app-show-tech',
  templateUrl: './show-tech.component.html',
  styleUrls: ['./show-tech.component.css']
})
export class ShowTechComponent   implements OnInit{

 constructor(private service:SharedService){}

  TeacherList:any=[];
  ModalTitle!: string;
  ActivateAddEditTechComp:boolean=false;
  tech:any;
 TeacherIdFilter:string="";
 TeacherNameFilter:string="";
 TeacherListWithoutFilter:any=[];
  ngOnInit(): void {
    this.refreshTechList();
  }
  refreshTechList(){
    this.service.getTeacherList().subscribe(data=>{
      this.TeacherList=data;
      this.TeacherListWithoutFilter=data;
    });
  }
  addClick(){
    this.tech={
      TeacherId:0,
      TeacherName:""
      
    }
    this.ModalTitle="Add Teacher";
    this.ActivateAddEditTechComp=true;

  }
  deleteClick(item: { TeacherId: any; }){
    if(confirm('Are you sure??')){
      this.service.deleteTeacher(item.TeacherId).subscribe(data=>{
        alert(data.toString());
        this.refreshTechList();
      })
    }
  }

  editClick(item: any){
    this.tech=item;
    this.ModalTitle="Edit Teacher";
    this.ActivateAddEditTechComp=true;
  }

  closeClick(){
    this.ActivateAddEditTechComp=false;
    this.refreshTechList();
  }
  FilterFn(){
    var TeacherIdFilter = this.TeacherIdFilter;
    var TeacherNameFilter = this.TeacherNameFilter;

    this.TeacherList = this.TeacherListWithoutFilter.filter(function (el: { TeacherId: { toString: () => string; }; TeacherName: { toString: () => string; }; }){
        return el.TeacherId.toString().toLowerCase().includes(
          TeacherIdFilter.toString().trim().toLowerCase()
        )&&
        el.TeacherName.toString().toLowerCase().includes(
          TeacherNameFilter.toString().trim().toLowerCase()
        )
    });
  }

  sortResult(prop: string | number,asc: any){
    this.TeacherList = this.TeacherListWithoutFilter.sort(function(a: { [x: string]: number; },b: { [x: string]: number; }){
      if(asc){
          return (a[prop]>b[prop])?1 : ((a[prop]<b[prop]) ?-1 :0);
      }else{
        return (b[prop]>a[prop])?1 : ((b[prop]<a[prop]) ?-1 :0);
      }
    })
  }

  
}
