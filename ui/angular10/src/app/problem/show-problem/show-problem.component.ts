import { Component,Input,OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {SharedService} from 'src/app/shared.service';

@Component({
  selector: 'app-show-problem',
  templateUrl: './show-problem.component.html',
  styleUrls: ['./show-problem.component.css']
})
export class ShowProblemComponent implements OnInit{

  constructor(private service:SharedService,private route: ActivatedRoute){}
  ModalTitle!: string;
  ActivateAddEditProblemComp:boolean=false;
  Problem:any;
  ProblemList:any=[];
  @Input()
  ID!: String;
  classId!: string;

  ngOnInit(): void {
    this.classId = this.route.snapshot.paramMap.get('ClassId') || 'default';

    this.refreshProblemList();
    
  }
  refreshProblemList(){
    this.service.getProblemList(this.classId).subscribe(data=>{
      this.ProblemList=data;
      
    });
}
addClick(){
  this.Problem={
    ProblemText:"",
    ProblemId:0,
    classId:this.classId,

    
  }
  this.ModalTitle="Add Problem";
  this.ActivateAddEditProblemComp=true;

}
deleteClick(item: { ProblemId: any; }){
  if(confirm('Are you sure??')){
    this.service.deleteProblem(item.ProblemId).subscribe(data=>{
      alert(data.toString());
      this.refreshProblemList();
    })
  }
}

editClick(item: any){
  this.Problem=item;
  this.Problem.classId = this.classId;

  this.ModalTitle="Edit classroom";
  this.ActivateAddEditProblemComp=true;
}
closeClick(){
  this.ActivateAddEditProblemComp=false;
  this.refreshProblemList();
}
}
