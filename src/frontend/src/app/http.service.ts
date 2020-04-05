import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { IResponse, IActionRequest } from './rubiks/rubiks';

const urls = {
  cube: "http://127.0.0.1:5000/cube",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  public async getSolved() {
    return this.http.get<IResponse>(urls.cube).toPromise();
  }

  public async performAction(actionRequest: IActionRequest) {
    return this.http.post<IResponse>(urls.cube, actionRequest).toPromise();
  }
}
