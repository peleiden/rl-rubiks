import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { IInfoResponse, ISolveRequest, ISolveResponse } from './rubiks';

const urls = {
  info: "info",
  solve: "solve",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  hosts = [
    { name: "Heroku", address: "https://rl-rubiks.herokuapp.com" },
    { name: "Local", address: "http://127.0.0.1:8000" },
  ];

  selectedHost = this.hosts[0];

  private getUrl(path: string) {
    return `${this.selectedHost.address}/${path}`;
  }

  public async getInfo() {
    return this.http.get<IInfoResponse>(this.getUrl(urls.info)).toPromise();
  }

  public async solve(solveRequest: ISolveRequest) {
    return this.http.post<ISolveResponse>(this.getUrl(urls.solve), solveRequest).toPromise();
  }
}
