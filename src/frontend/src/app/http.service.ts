import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Cube } from './rubiks/rubiks';

const urls = {
  cube: "http://127.0.0.1:5000/cube",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  public async getSolved() {
    return this.http.get<Cube>(urls.cube).toPromise();
  }

  public async performAction(action: number) {
    return this.http.post<Cube>(urls.cube, action).toPromise();
  }
}
