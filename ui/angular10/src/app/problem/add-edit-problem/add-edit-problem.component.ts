import { Component, Input, OnInit } from '@angular/core';
import { SharedService } from 'src/app/shared.service';

@Component({
  selector: 'app-add-edit-problem',
  templateUrl: './add-edit-problem.component.html',
  styleUrls: ['./add-edit-problem.component.css']
})
export class AddEditProblemComponent  implements OnInit{
  constructor(private service:SharedService){}

  @Input() 
  Problem :any;
  ProblemId!: string;
  ProblemText!: string;
  classId!:string;
  ngOnInit(): void {
    this.ProblemId=this.Problem.ProblemId;
    this.ProblemText=this.Problem.ProblemText;
    this.classId=this.Problem.classId;

  }
  addProblem(){
    var val = {ProblemId:this.ProblemId,
      ProblemText:this.ProblemText,
      Class:this.classId
    };
this.service.addProblem(val).subscribe(res=>{
alert(res.toString());
});
  }
  updateProblem(){
    var val = {
      ProblemId:this.ProblemId,
      ProblemText:this.ProblemText,
      Class:this.classId
    };
    this.service.updateProblem(val).subscribe(res=>{
    alert(res.toString());
  });
  }

}
