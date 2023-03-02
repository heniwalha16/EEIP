import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class SharedService {
  readonly APIUrl = "http://127.0.0.1:8000";

  constructor(private http:HttpClient) { }
  getTeacherList():Observable<any[]>{
    return this.http.get<any[]>(this.APIUrl + '/teacher/');
  }

  addTeacher(val:any){
    return this.http.post(this.APIUrl + '/teacher/',val);
  }

  updateTeacher(val:any){
    return this.http.put(this.APIUrl + '/teacher/',val);
  }

  deleteTeacher(val:any){
    return this.http.delete(this.APIUrl + '/teacher/'+val);
  }
  /*
  getClassList():Observable<any[]>{
    return this.http.get<any[]>(this.APIUrl + '/Class/');
  }*/
  getClassList(val:any){
    return this.http.get(this.APIUrl + '/Class/'+val);
  }

  addClass(val:any){
    return this.http.post(this.APIUrl + '/Class/',val);
  }

  updateClass(val:any){
    return this.http.put(this.APIUrl + '/Class/',val);
  }

  deleteClass(val:any){
    return this.http.delete(this.APIUrl + '/Class/'+val);
  }
  getProblemList(val:any){
    return this.http.get(this.APIUrl + '/Problem/'+val);
  }

  addProblem(val:any){
    return this.http.post(this.APIUrl + '/Problem/',val);
  }

  updateProblem(val:any){
    return this.http.put(this.APIUrl + '/Problem/',val);
  }

  deleteProblem(val:any){
    return this.http.delete(this.APIUrl + '/Problem/'+val);
  }
}
